from typing import Union, Tuple, List
import math
import torch
import torch.nn.functional as F
from pytorch3d.ops import (convert_pointclouds_to_tensor, is_pointclouds)
from pytorch3d.structures import Pointclouds as PytorchPointClouds
from pytorch3d.structures import list_to_padded, padded_to_list
from pytorch3d.transforms import Transform3d, Scale, Rotate, Translate
from pytorch3d.renderer.cameras import look_at_rotation
from pytorch3d.renderer.utils import (
    TensorProperties, convert_to_tensors_and_broadcast)
from pytorch3d.ops import knn_points, knn_gather
from pytorch3d.ops.knn import _KNN
import frnn
from ..utils.mathHelper import eps_denom, estimate_pointcloud_local_coord_frames, estimate_pointcloud_normals
from ..utils import mask_from_padding
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
            assert(others_packed.nelement() ==
                   0), "Cannot update empty pointclouds with non-empty features"
            return self
        normals_packed = self.normals_packed()
        if normals_packed is not None:
            if others_packed.shape != normals_packed.shape:
                raise ValueError(
                    "update normals must have dimension (all_p, 3).")
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
            assert(others_packed.nelement() ==
                   0), "Cannot update empty pointclouds with non-empty features"
            return self
        features_packed = self.features_packed()
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
        center = torch.stack([x.mean(dim=0)
                              for x in self.points_list()], dim=0)
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
        self.scale_(1 / eps_denom(scale))
        normalizing_trans = Translate(-center).compose(
            Scale(1 / eps_denom(scale))).to(device=self.device)
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

    def estimate_normals(
        self,
        neighborhood_size: int = 50,
        disambiguate_directions: bool = True,
        assign_to_self: bool = False,
    ):
        """
        Estimates the normals of each point in each cloud and assigns
        them to the internal tensors `self._normals_list` and `self._normals_padded`

        The function uses `ops.estimate_pointcloud_local_coord_frames`
        to estimate the normals. Please refer to this function for more
        detailed information about the implemented algorithm.

        Args:
        **neighborhood_size**: The size of the neighborhood used to estimate the
            geometry around each point.
        **disambiguate_directions**: If `True`, uses the algorithm from [1] to
            ensure sign consistency of the normals of neigboring points.
        **normals**: A tensor of normals for each input point
            of shape `(minibatch, num_point, 3)`.
            If `pointclouds` are of `Pointclouds` class, returns a padded tensor.
        **assign_to_self**: If `True`, assigns the computed normals to the
            internal buffers overwriting any previously stored normals.

        References:
          [1] Tombari, Salti, Di Stefano: Unique Signatures of Histograms for
          Local Surface Description, ECCV 2010.
        """

        # estimate the normals
        normals_est = estimate_pointcloud_normals(
            self,
            neighborhood_size=neighborhood_size,
            disambiguate_directions=disambiguate_directions,
        )

        # assign to self
        if assign_to_self:
            _, self._normals_padded, _ = self._parse_auxiliary_input(normals_est)
            self._normals_list, self._normals_packed = None, None
            if self._points_list is not None:
                # update self._normals_list
                self.normals_list()
            if self._points_packed is not None:
                # update self._normals_packed
                self._normals_packed = torch.cat(self._normals_list, dim=0)

        return normals_est

    def subsample_randomly(self, ratio):
        if not isinstance(ratio, torch.Tensor):
            ratio = torch.full((len(self),), ratio, device=self.device)
        assert ratio.nelement() == len(self)

        points_list = self.points_list()
        normals_list = self.normals_list()
        features_list = self.features_list()
        for b, pts in enumerate(points_list):
            idx = torch.randperm(pts.shape[0])[:int(ratio[b]*pts.shape[0])]
            points_list[b] = pts[idx]
            if features_list is not None:
                features_list[b] = features_list[b][idx]
            if normals_list is not None:
                normals_list[b] = normals_list[b][idx]

        other = self.__class__(
            points=points_list, normals=normals_list, features=features_list
        )
        return other


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
        points_padded, num_points = convert_pointclouds_to_tensor(point_clouds)
        # point_clouds N, filter 1

        matched_tensors = convert_to_tensors_and_broadcast(*filters,
                                                           points_padded,
                                                           num_points,
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
            filters[i, N:] = False

        points_list = [points[b][filters[b]] for b in range(points.shape[0])]
        if not is_pointclouds(point_clouds):
            return PointClouds3D(points_list)

        normals = point_clouds.normals_padded()
        if normals is not None:
            normals = normals.expand(points.shape[0], -1, -1)
            normals = [normals[b][filters[b]] for b in range(normals.shape[0])]

        features = point_clouds.features_padded()
        if features is not None:
            features = features.expand(points.shape[0], -1, -1)
            features = [features[b][filters[b]]
                        for b in range(features.shape[0])]

        return PointClouds3D(points_list, normals=normals, features=features)


def remove_outliers(pointclouds, neighborhood_size=16, tolerance=0.05):
    """
    Identify a point as outlier if the ratio of the smallest and largest
    variance is > than a threshold
    """
    points, num_points = convert_pointclouds_to_tensor(pointclouds)
    mask_padding = mask_from_padding(num_points)
    variance, local_frame = estimate_pointcloud_local_coord_frames(
        points, neighborhood_size=neighborhood_size)
    # thres = variance[..., -1].median(dim=1)[0] * 16
    # largest
    mask = (variance[...,0] / torch.sum(variance, dim=-1)) < tolerance
    # mask = variance[...,-1] < thres
    pointclouds_filtered = PointCloudsFilters(
        device=pointclouds.device, activation=mask & mask_padding).filter(pointclouds)
    return pointclouds_filtered


def resample_uniformly(pointclouds, neighborhood_size=8, iters=1, knn=None, normals=None, reproject=False, repulsion_mu=1.0):
    """ resample sample_iters times """
    import math
    import frnn
    points_init, num_points = convert_pointclouds_to_tensor(pointclouds)
    batch_size = num_points.shape[0]
    # knn_result = knn_points(
    #     points_init, points_init, num_points, num_points, K=neighborhood_size + 1, return_nn=True)
    diag = (points_init.view(-1, 3).max(dim=0).values -
            points_init.view(-1, 3).min(0).values).norm().item()
    avg_spacing = math.sqrt(diag / points_init.shape[1])
    search_radius = min(
        4 * avg_spacing * neighborhood_size, 0.2)
    if knn is None:
        dists, idxs, _, grid = frnn.frnn_grid_points(points_init, points_init,
                                                    num_points, num_points, K=neighborhood_size + 1, r=search_radius, grid=None, return_nn=False)
        knn = _KNN(dists=dists[...,1:], idx=idxs[...,1:], knn=None)

    # estimate normals
    if isinstance(pointclouds, torch.Tensor):
        normals = normals
    else:
        normals = pointclouds.normals_padded()

    if normals is None:
        normals = estimate_pointcloud_normals(points_init, neighborhood_size=neighborhood_size, disambiguate_directions=False)
    else:
        normals = F.normalize(normals, dim=-1)

    points = points_init
    for i in range(iters):
        if reproject:
            normals = denoise_normals(points, normals, num_points, knn_result=knn)
            points = project_to_latent_surface(points, normals, max_proj_iters=2, max_est_iter=3)
        if i >0 and i % 3 == 0:
            dists, idxs, _, grid = frnn.frnn_grid_points(points_init, points_init,
                                                    num_points, num_points, K=neighborhood_size + 1, r=search_radius, grid=None, return_nn=False)
            knn = _KNN(dists=dists[...,1:], idx=idxs[...,1:], knn=None)
        nn = frnn.frnn_gather(points, knn.idx, num_points)
        pts_diff = points.unsqueeze(-2) -nn
        dists = torch.sum(pts_diff**2, dim=-1)
        knn_result = _KNN(dists=dists, idx=knn.idx, knn=nn)
        deltap = knn_result.dists
        inv_sigma_spatial = num_points / 2.0 / 16
        spatial_w = torch.exp(-deltap * inv_sigma_spatial)
        spatial_w[knn_result.idx < 0] = 0
        # density_w = torch.sum(spatial_w, dim=-1) + 1.0
        # 0.5 * derivative of (-r)exp(-r^2*inv)
        density = frnn.frnn_gather(spatial_w.sum(-1, keepdim=True) + 1.0, knn.idx, num_points)
        nn_normals = frnn.frnn_gather(normals, knn_result.idx, num_points)
        pts_diff_proj = pts_diff - (pts_diff*nn_normals).sum(dim=-1, keepdim=True)*nn_normals
        # move = 0.5 * torch.sum(density*spatial_w[..., None] * pts_diff_proj, dim=-2) / torch.sum(density.view_as(spatial_w)*spatial_w, dim=-1).unsqueeze(-1)
        # move = F.normalize(move, dim=-1) * move.norm(dim=-1, keepdim=True).clamp_max(2*avg_spacing)
        move = repulsion_mu * avg_spacing * torch.mean(density*spatial_w[..., None] * F.normalize(pts_diff_proj, dim=-1), dim=-2)
        points = points + move
        # then project to latent surface again

    if is_pointclouds(pointclouds):
        return pointclouds.update_padded(points)
    return points

def project_to_latent_surface(points, normals, sharpness_angle=60, neighborhood_size=31, max_proj_iters=10, max_est_iter=5):
    """
    RIMLS
    """
    points, num_points = convert_pointclouds_to_tensor(points)
    normals = F.normalize(normals, dim=-1)
    sharpness_sigma = 1 - math.cos(sharpness_angle / 180 * math.pi)
    diag = (points.max(dim=-2)[0] - points.min(dim=-2)[0]).norm(dim=-1)
    avg_spacing = math.sqrt(diag / points.shape[1])
    search_radius = min(16 * avg_spacing * neighborhood_size, 0.2)

    dists, idxs, _, grid = frnn.frnn_grid_points(points, points,
                                                num_points, num_points, K=neighborhood_size + 1, r=search_radius, grid=None, return_nn=False)
    knn_result = _KNN(dists=dists[...,1:], idx=idxs[...,1:], knn=None)

    # knn_normals = knn_gather(normals, knn_result.idx, num_points)
    knn_normals = frnn.frnn_gather(normals, knn_result.idx, num_points)

    inv_sigma_spatial = 1/knn_result.dists[...,0]/16
    # spatial_dist = 16 / inv_sigma_spatial
    not_converged = torch.full(points.shape[:-1], True, device=points.device, dtype=torch.bool)
    itt = 0
    it = 0
    while True:
        knn_pts = frnn.frnn_gather(points, knn_result.idx, num_points)
        pts_diff = points[not_converged].unsqueeze(-2) - knn_pts[not_converged]
        fx = torch.sum(pts_diff*knn_normals[not_converged], dim=-1)
        not_converged_1 = torch.full(fx.shape[:-1], True, dtype=torch.bool, device=fx.device)
        knn_normals_1 = knn_normals[not_converged]
        inv_sigma_spatial_1 = inv_sigma_spatial[not_converged]
        f = points.new_zeros(points[not_converged].shape[:-1], device=points.device)
        grad_f = points.new_zeros(points[not_converged].shape, device=points.device)
        alpha = torch.ones_like(fx)
        for itt in range(max_est_iter):
            if itt > 0:
                alpha_old = alpha
                weights_n = ((knn_normals_1[not_converged_1] - grad_f[not_converged_1].unsqueeze(-2)).norm(dim=-1) / 0.5)**2
                weights_n = torch.exp(-weights_n)
                weights_p = torch.exp(-((fx[not_converged_1] - f[not_converged_1].unsqueeze(-1))**2*inv_sigma_spatial_1[not_converged_1].unsqueeze(-1)/4))
                alpha[not_converged_1] = weights_n * weights_p
                not_converged_1[not_converged_1] = (alpha[not_converged_1] - alpha_old[not_converged_1]).abs().max(dim=-1)[0] < 1e-4
                if not not_converged_1.any():
                    break

            deltap = torch.sum(pts_diff[not_converged_1] * pts_diff[not_converged_1], dim=-1)
            phi = torch.exp(-deltap * inv_sigma_spatial_1[not_converged_1].unsqueeze(-1))
            # phi[deltap > spatial_dist] = 0
            dphi = inv_sigma_spatial_1[not_converged_1].unsqueeze(-1)*phi

            weights = phi * alpha[not_converged_1]
            grad_weights = 2*pts_diff*(dphi * weights).unsqueeze(-1)

            sum_grad_weights = torch.sum(grad_weights, dim=-2)
            sum_weight = torch.sum(weights, dim=-1)
            sum_f = torch.sum(fx[not_converged_1] * weights, dim=-1)
            sum_Gf = torch.sum(grad_weights*fx[not_converged_1].unsqueeze(-1), dim=-2)
            sum_N = torch.sum(weights.unsqueeze(-1) * knn_normals_1[not_converged_1], dim=-2)

            tmp_f = sum_f / eps_denom(sum_weight)
            tmp_grad_f = (sum_Gf - tmp_f.unsqueeze(-1)*sum_grad_weights + sum_N) / eps_denom(sum_weight).unsqueeze(-1)
            grad_f[not_converged_1] = tmp_grad_f
            f[not_converged_1] = tmp_f

        move = f.unsqueeze(-1) * grad_f
        points[not_converged] = points[not_converged]-move
        mask = move.norm(dim=-1) > 5e-4
        not_converged[not_converged] = mask
        it = it + 1
        if not not_converged.any() or it >= max_proj_iters:
            break

    return points

def denoise_normals(points, normals, num_points, sharpness_sigma=30, knn_result=None, neighborhood_size=16):
    """
    Weights exp(-(1-<n, n_i>)/(1-cos(sharpness_sigma))), for i in a local neighborhood
    """
    points, num_points = convert_pointclouds_to_tensor(points)
    normals = F.normalize(normals, dim=-1)
    if knn_result is None:
        diag = (points.max(dim=-2)[0] - points.min(dim=-2)[0]).norm(dim=-1)
        avg_spacing = math.sqrt(diag / points.shape[1])
        search_radius = min(
            4 * avg_spacing * neighborhood_size, 0.2)

        dists, idxs, _, grid = frnn.frnn_grid_points(points, points,
                                                    num_points, num_points, K=neighborhood_size + 1, r=search_radius, grid=None, return_nn=True)
        knn_result = _KNN(dists=dists[...,1:], idx=idxs[...,1:], knn=None)
    if knn_result.knn is None:
        knn = frnn.frnn_gather(points, knn_result.idx, num_points)
        knn_result = _KNN(idx=knn_result.idx, knn=knn, dists=knn_result.dists)

    # filter out
    knn_normals = frnn.frnn_gather(normals, knn_result.idx, num_points)
    # knn_normals = frnn.frnn_gather(normals, self._knn_idx, num_points)
    weights_n = ((1 - torch.sum(knn_normals *
                                normals[:, :, None, :], dim=-1)) / sharpness_sigma)**2
    weights_n = torch.exp(-weights_n)

    inv_sigma_spatial = num_points / 2.0
    spatial_dist = 16 / inv_sigma_spatial
    deltap = knn - points[:, :, None, :]
    deltap = torch.sum(deltap * deltap, dim=-1)
    weights_p = torch.exp(-deltap * inv_sigma_spatial)
    weights_p[deltap > spatial_dist] = 0
    weights = weights_p * weights_n
    # weights[self._knn_idx < 0] = 0
    normals_denoised = torch.sum(knn_normals * weights[:, :, :, None], dim=-2) / \
        eps_denom(torch.sum(weights, dim=-1, keepdim=True))
    normals_denoised = F.normalize(normals_denoised, dim=-1)
    return normals_denoised.view_as(normals)


def upsample(points, n_points: Union[int, torch.Tensor], num_points=None, neighborhood_size=16, knn_result=None):
    """
    Args:
        points (N, P, 3)
        n_points (tensor of [N] or integer): target number of points per cloud

    """
    batch_size = points.shape[0]
    knn_k = neighborhood_size
    if num_points is None:
        num_points = torch.tensor([points.shape[1]] * points.shape[0],
                                  device=points.device, dtype=torch.long)
    if not ((num_points - num_points[0]) == 0).all():
        logger_py.warn(
            "May encounter unexpected behavior for heterogeneous batches")
    if num_points.sum() == 0:
        return points, num_points
    n_remaining = n_points - num_points
    if (n_remaining == 0).all():
        return points, num_points

    point_cloud_diag = (points.max(dim=-2)[0] - points.min(dim=-2)[0]).norm(dim=-1)
    inv_sigma_spatial = num_points / point_cloud_diag
    spatial_dist = 16 / inv_sigma_spatial

    if knn_result is None:
        knn_result = knn_points(
            points, points, num_points, num_points,
            K=knn_k + 1, return_nn=True, return_sorted=True)

        knn_result = _KNN(dists=knn_result.dists[..., 1:], idx=knn_result.idx[..., 1:], knn=knn_result.knn[..., 1:, :])

    while True:
        if (n_remaining == 0).all():
            break
        # half of the points per batch
        sparse_pts = points
        sparse_dists = knn_result.dists
        sparse_knn = knn_result.knn
        batch_size, P, _ = sparse_pts.shape
        max_P = (P // 10)
        # sparse_knn_normals = frnn.frnn_gather(
        #     normals_init, knn_result.idx, num_points)[:, 1:]
        # get all mid points
        mid_points = (sparse_knn + 2 * sparse_pts[..., None, :]) / 3
        # N,P,K,K,3
        mid_nn_diff = mid_points.unsqueeze(-2) - sparse_knn.unsqueeze(-3)
        # minimize among all the neighbors
        min_dist2 = torch.norm(mid_nn_diff, dim=-1)  # N,P,K,K
        min_dist2 = min_dist2.min(dim=-1)[0]  # N,P,K
        father_sparsity, father_nb = min_dist2.max(dim=-1)  # N,P
        # neighborhood to insert
        sparsity_sorted = father_sparsity.sort(dim=1).indices
        n_new_points = n_remaining.clone()
        n_new_points[n_new_points > max_P] = max_P
        sparsity_sorted = sparsity_sorted[:, -max_P:]
        new_pts = torch.gather(mid_points[torch.arange(mid_points.shape[0]), torch.arange(mid_points.shape[1]), father_nb], 1,
                               sparsity_sorted.unsqueeze(-1).expand(-1, -1, 3))

        from DSS.utils.io import save_ply
        sparse_selected = torch.gather(sparse_pts, 1, sparsity_sorted.unsqueeze(-1).expand(-1, -1, 3))
        # save_ply('tests/outputs/test_uniform_projection/init.ply', sparse_pts.view(-1,3).cpu())
        # save_ply('tests/outputs/test_uniform_projection/sparse.ply', sparse_selected[0].cpu())
        # save_ply('tests/outputs/test_uniform_projection/new_pts.ply', new_pts.view(-1,3).cpu().detach())
        # import pdb; pdb.set_trace()
        total_pts_list = []
        for b, pts_batch in enumerate(padded_to_list(points, num_points.tolist())):
            total_pts_list.append(
                torch.cat([new_pts[b][-n_new_points[b]:], pts_batch], dim=0))

        points = list_to_padded(total_pts_list)
        n_remaining = n_remaining - n_new_points
        num_points = n_new_points + num_points
        knn_result = knn_points(
            points, points, num_points, num_points, K=knn_k + 1, return_nn=True)
        knn_result = _KNN(dists=knn_result.dists[..., 1:], idx=knn_result.idx[..., 1:], knn=knn_result.knn[..., 1:, :])

    return points, num_points

def upsample_ear(points,  normals, n_points: Union[int, torch.Tensor], num_points=None, neighborhood_size=16, repulsion_mu=0.4, edge_sensitivity=1.0):
    """
    Args:
        points (N, P, 3)
        n_points (tensor of [N] or integer): target number of points per cloud

    """
    batch_size = points.shape[0]
    knn_k = neighborhood_size
    if num_points is None:
        num_points = torch.tensor([points.shape[1]] * points.shape[0],
                                  device=points.device, dtype=torch.long)
    if not ((num_points - num_points[0]) == 0).all():
        logger_py.warn(
            "May encounter unexpected behavior for heterogeneous batches")
    if num_points.sum() == 0:
        return points, num_points

    point_cloud_diag = (points.max(dim=-2)[0] - points.min(dim=-2)[0]).norm(dim=-1)
    inv_sigma_spatial = num_points / point_cloud_diag
    spatial_dist = 16 / inv_sigma_spatial

    knn_result = knn_points(
        points, points, num_points, num_points,
        K=knn_k + 1, return_nn=True, return_sorted=True)
    # dists, idxs, nn, grid = frnn.frnn_grid_points(points_proj, points_proj, num_points, num_points, K=self.knn_k + 1,
    #                                               r=torch.sqrt(spatial_dist), return_nn=True)
    # knn_result = _KNN(dists=dists, idx=idxs, knn=nn)
    _knn_idx = knn_result.idx[..., 1:]
    _knn_dists = knn_result.dists[..., 1:]
    _knn_nn = knn_result.knn[..., 1:, :]
    move_clip = knn_result.dists[..., 1].mean().sqrt()

    # 2. LOP projection
    if denoise_normals:
        normals_denoised, weights_p, weights_n = denoise_normals(
            points, normals, num_points, knn_result=knn_result)
        normals = normals_denoised

    # (optional) search knn in the original points
    # e(-(<n, p-pi>)^2/sigma_p)
    weight_lop = torch.exp(-torch.sum(normals[:, :, None, :] *
                                        (points[:, :, None, :] - _knn_nn), dim=-1)**2 * inv_sigma_spatial)
    weight_lop[_knn_dists > spatial_dist] = 0
        # weight_lop[self._knn_idx < 0] = 0

    # spatial weight
    deltap = _knn_dists
    spatial_w = torch.exp(-deltap * inv_sigma_spatial)
    spatial_w[deltap > spatial_dist] = 0
    # spatial_w[self._knn_idx[..., 1:] < 0] = 0
    density_w = torch.sum(spatial_w, dim=-1) + 1.0
    move_data = torch.sum(
        weight_lop[..., None] * (points[:, :, None, :] - _knn_nn), dim=-2) / \
        eps_denom(torch.sum(weight_lop, dim=-1, keepdim=True))
    move_repul = repulsion_mu * density_w[..., None] * torch.sum(spatial_w[..., None] * (
        knn_result.knn[:, :, 1:, :] - points[:, :, None, :]), dim=-2) / \
        eps_denom(torch.sum(spatial_w, dim=-1, keepdim=True))
    move_repul = F.normalize(
        move_repul) * move_repul.norm(dim=-1, keepdim=True).clamp_max(move_clip)
    move_data = F.normalize(
        move_data) * move_data.norm(dim=-1, keepdim=True).clamp_max(move_clip)
    move = move_data + move_repul
    points = points - move

    n_remaining = n_points - num_points
    while True:
        if (n_remaining == 0).all():
            break
        # half of the points per batch
        sparse_pts = points
        sparse_dists = _knn_dists
        sparse_knn = _knn_nn
        batch_size, P, _ = sparse_pts.shape
        max_P = (P // 10)
        # sparse_knn_normals = frnn.frnn_gather(
        #     normals_init, knn_result.idx, num_points)[:, 1:]
        # get all mid points
        mid_points = (sparse_knn + 2 * sparse_pts[..., None, :]) / 3
        # N,P,K,K,3
        mid_nn_diff = mid_points.unsqueeze(-2) - sparse_knn.unsqueeze(-3)
        # minimize among all the neighbors
        min_dist2 = torch.norm(mid_nn_diff, dim=-1)  # N,P,K,K
        min_dist2 = min_dist2.min(dim=-1)[0]  # N,P,K
        father_sparsity, father_nb = min_dist2.max(dim=-1)  # N,P
        # neighborhood to insert
        sparsity_sorted = father_sparsity.sort(dim=1).indices
        n_new_points = n_remaining.clone()
        n_new_points[n_new_points > max_P] = max_P
        sparsity_sorted = sparsity_sorted[:, -max_P:]
        # N, P//2, 3, sparsest at the end
        new_pts = torch.gather(mid_points[torch.arange(mid_points.shape[0]), torch.arange(mid_points.shape[1]), father_nb], 1,
                               sparsity_sorted.unsqueeze(-1).expand(-1, -1, 3))
        total_pts_list = []
        for b, pts_batch in enumerate(padded_to_list(points, num_points.tolist())):
            total_pts_list.append(
                torch.cat([new_pts[b][-n_new_points[b]:], pts_batch], dim=0))

        points_proj = list_to_padded(total_pts_list)
        n_remaining = n_remaining - n_new_points
        num_points = n_new_points + num_points
        knn_result = knn_points(
            points_proj, points_proj, num_points, num_points, K=knn_k + 1, return_nn=True)
        _knn_idx = knn_result.idx[..., 1:]
        _knn_dists = knn_result.dists[..., 1:]
        _knn_nn = knn_result.knn[..., 1:, :]

    return points_proj, num_points
