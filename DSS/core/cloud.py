from typing import Union, Tuple, List
import torch
import torch.nn.functional as F
from pytorch3d.ops import eyes, padded_to_packed, packed_to_padded
from pytorch3d.structures import Pointclouds as PytorchPointClouds
from pytorch3d.structures import list_to_padded
from pytorch3d.transforms import Transform3d, Scale, Rotate, Translate
from pytorch3d.renderer.cameras import look_at_rotation
from pytorch3d.renderer.utils import TensorProperties, convert_to_tensors_and_broadcast
from ..utils.mathHelper import eps_denom
from ..utils import gather_batch_to_packed
from .. import logger_py


__all__ = ["PointClouds3D", "PointCloudsFilters"]


class PointClouds3D(PytorchPointClouds):
    """ PointClouds storing batches of point clouds in *object coordinate*.
    The point clouds are centered and isotropically resized to a unit cube,
    with up direction (0, 1, 0) and front direction (0, 0, -1)
    Overload of pytorch3d Pointclouds class

    Support named features, with a OrderedDict {name: dimensions}
    Attributes:
        normalized (bool): whether the point cloud is centered and normalized
        obj2world_mat (tensor): (B, 4, 4) object-to-world transformation for
            each (normalized and aligned) point cloud

    """

    def __init__(self, points, normals=None, features=None,
                 to_unit_sphere: bool = False,
                 to_unit_box: bool = False,
                 to_axis_aligned: bool = False,
                 up=((0, 1, 0),),
                 front=((0, 0, 1),)
                 ):
        """
        Args:
            points, normals: points in world coordinates
            (unnormalized and unaligned) Pointclouds in pytorch3d
            features: can be a dict {name: value} where value can be any acceptable
                form as the pytorch3d.Pointclouds
            to_unit_box (bool): transform to unit box (sidelength = 1)
            to_axis_aligned (bool): rotate the object using the up and front vectors
            up: the up direction in world coordinate (will be justified to object)
            front: front direction in the world coordinate (will be justified to z-axis)
        """
        super().__init__(points, normals=normals, features=features)
        self.obj2world_trans = Transform3d()

        # rotate object to have up direction (0, 1, 0)
        # and front direction (0, 0, -1)
        # (B,3,3) rotation to transform to axis-aligned point clouds
        if to_axis_aligned:
            self.obj2world_trans = Rotate(look_at_rotation(
                ((0, 0, 0),), at=front, up=up), device=self.device)
            world_to_obj_rotate_trans = self.obj2world_trans.inverse()

            # update points, normals
            self.update_points_(
                world_to_obj_rotate_trans.transform_points(self.points_packed()))
            normals_packed = self.normals_packed()
            if normals_packed is not None:
                self.update_normals_(
                    world_to_obj_rotate_trans.transform_normals(normals_packed))

        # normalize to unit box and update obj2world_trans
        if to_unit_box:
            normalizing_trans = self.normalize_to_box_()

        elif to_unit_sphere:
            normalizing_trans = self.normalize_to_sphere_()


    def update_points_(self, others_packed):
        points_packed = self.points_packed()
        if others_packed.shape != points_packed.shape:
            raise ValueError("update points must have dimension (all_p, 3).")
        self.offset_(others_packed - points_packed)

    def update_normals_(self, others_packed):
        """
        Update the point clouds normals. In place operation.

        Args:
            offsets_packed: A Tensor of the same shape as self.points_packed
                giving offsets to be added to all points.
        Returns:
            self.
        """
        if self.isempty():
            assert(others_packed.nelement() == 0), "Cannot update empty pointclouds with non-empty features"
            return self
        normals_packed = self.normals_packed()
        if normals_packed is not None:
            if others_packed.shape != normals_packed.shape:
                raise ValueError("update normals must have dimension (all_p, 3).")
        if normals_packed is None:
            self._normals_packed = others_packed
        else:
            normals_packed += (-normals_packed + others_packed)

        new_normals_list = list(
            self._normals_packed.split(self.num_points_per_cloud().tolist(), 0)
        )
        # Note that since _compute_packed() has been executed, points_list
        # cannot be None even if not provided during construction.
        self._normals_list = new_normals_list
        self._normals_padded = list_to_padded(new_normals_list)

        return self

    def update_features_(self, others_packed):
        """
        Update the point clouds features. In place operation.

        Args:
            offsets_packed: A Tensor of the same shape as self.points_packed
                giving offsets to be added to all points.
        Returns:
            self.
        """
        if self.isempty():
            assert(others_packed.nelement() == 0), "Cannot update empty pointclouds with non-empty features"
            return self
        features_packed = self.features_packed()
        if features_packed is not None:
            if others_packed.shape[0] != features_packed.shape[0]:
                raise ValueError("Offsets must have dimension (all_p, c).")
        if features_packed is None or features_packed.shape != others_packed.shape:
            self._features_packed = others_packed
            self._C = others_packed.shape[-1]
        else:
            features_packed += (-features_packed + others_packed)

        new_features_list = list(
            self._features_packed.split(
                self.num_points_per_cloud().tolist(), 0)
        )
        # Note that since _compute_packed() has been executed, points_list
        # cannot be None even if not provided during construction.
        self._features_list = new_features_list

        self._features_padded = list_to_padded(new_features_list)
        return self

    def normalize_to_sphere_(self):
        """
        Center and scale the point clouds to a unit sphere
        Returns: normalizing_trans (Transform3D)
        """
        # packed offset
        center = torch.stack([x.mean(dim=0) for x in self.points_list()], dim=0)
        center_packed = torch.repeat_interleave(-center,
                                               self.num_points_per_cloud(),
                                               dim=0)
        self.offset_(center_packed)
        # (P)
        norms = torch.norm(self.points_packed(), dim=-1)
        # List[(Pi)]
        norms = torch.split(norms, self.num_points_per_cloud())
        # (N)
        scale = torch.stack([x.max() for x in norms], dim=0)
        self.scale_(1/eps_denom(scale))
        normalizing_trans = Translate(-center).compose(Scale(1/eps_denom(scale))).to(device=self.device)
        self.obj2world_trans = normalizing_trans.inverse().compose(self.obj2world_trans)
        return normalizing_trans


    def normalize_to_box_(self):
        """
        center and scale the point clouds to a unit cube,
        Returns:
            normalizing_trans (Transform3D): Transform3D used to normalize the pointclouds
        """
        # (B,3,2)
        boxMinMax = self.get_bounding_boxes()
        boxCenter = boxMinMax.sum(dim=-1) / 2
        # (B,)
        boxRange, _ = (boxMinMax[:, :, 1] - boxMinMax[:, :, 0]).max(dim=-1)
        if boxRange == 0:
            boxRange = 1

        # center and scale the point clouds, likely faster than calling obj2world_trans directly?
        pointOffsets = torch.repeat_interleave(-boxCenter,
                                               self.num_points_per_cloud(),
                                               dim=0)
        self.offset_(pointOffsets)
        self.scale_(1 / boxRange)

        # update obj2world_trans
        normalizing_trans = Translate(-boxCenter).compose(
            Scale(1 / boxRange)).to(device=self.device)
        self.obj2world_trans = normalizing_trans.inverse().compose(self.obj2world_trans)
        return normalizing_trans

    def get_object_to_world_transformation(self, **kwargs):
        """
            Returns a Transform3d object from object to world
        """
        return self.obj2world_trans


true_tensor = torch.tensor([True], dtype=torch.bool).view(1, 1)


class PointCloudsFilters(TensorProperties):
    """ Filters are padded 2-D boolean mask (N, P_max) """
    def __init__(self, device='cpu',
                 inmask=true_tensor,
                 activation=true_tensor,
                 visibility=true_tensor,
                 **kwargs
                 ):
        super().__init__(device=device,
                         inmask=inmask,
                         activation=activation,
                         visibility=visibility,
                         **kwargs)

    def set_filter(self, **kwargs):
        """ filter should be 2-dim tensor (for padded values)"""
        my_filters = {}
        for k in dir(self):
            v = getattr(self, k)
            if torch.is_tensor(v):
                my_filters[k] = v
                self.device = v.device

        my_filters.update(kwargs)
        self.__init__(device=self.device, **my_filters)

    def filter(self, point_clouds: PointClouds3D):
        """ filter with all the existing filters """
        # CHECK
        names = [k for k in dir(self) if torch.is_tensor(getattr(self, k))]
        return self.filter_with(point_clouds, names)

    def filter_with(self, point_clouds: PointClouds3D, filter_names: Tuple[str]):
        """
        filter point clouds with all the specified filters,
        return the reduced point clouds
        """
        filters = [getattr(self, k)
                   for k in filter_names if torch.is_tensor(getattr(self, k))]
        # point_clouds N, filter 1
        matched_tensors = convert_to_tensors_and_broadcast(*filters,
                                                           point_clouds.points_padded(),
                                                           point_clouds.num_points_per_cloud(),
                                                           device=self.device)
        filters = matched_tensors[:-2]
        points = matched_tensors[-2]
        num_points_per_cloud = matched_tensors[-1]

        assert(all(x.ndim == 2 for x in filters))
        size1 = max([x.shape[1] for x in matched_tensors[:-1]])
        filters = [x.expand(-1, size1) for x in filters]

        # make sure that filters at the padded positions are 0
        filters = torch.stack(filters, dim=-1).all(dim=-1)
        for i, N in enumerate(num_points_per_cloud.cpu().tolist()):
            filters[i,N:] = False

        points_list = [points[b][filters[b]] for b in range(points.shape[0])]

        normals = point_clouds.normals_padded()
        if normals is not None:
            normals = normals.expand(points.shape[0], -1, -1)
            normals = [normals[b][filters[b]] for b in range(normals.shape[0])]

        features = point_clouds.features_padded()
        if features is not None:
            features = features.expand(points.shape[0], -1, -1)
            features = [features[b][filters[b]] for b in range(features.shape[0])]

        return PointClouds3D(points_list, normals=normals, features=features)
