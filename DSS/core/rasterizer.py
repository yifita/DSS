"""
Surface Splatting Rasterizer

"""
from typing import NamedTuple, Optional

import torch
import torch.autograd as autograd
import torch.nn.functional as F
import numpy as np

import pytorch3d.ops as ops3d
from pytorch3d.renderer import (PointsRasterizationSettings,
                                PointsRasterizer,
                                )
from pytorch3d.renderer.points.rasterize_points import kMaxPointsPerBin
import frnn
from .cloud import PointClouds3D
from ..utils import gather_with_neg_idx, gather_batch_to_packed, get_per_point_visibility_mask, num_points_2_cloud_to_packed_first_idx
from ..utils.mathHelper import eps_denom, eps_sqrt, to_homogen, estimate_pointcloud_local_coord_frames
from .. import _C, logger_py

"""
A rasterizer takes a point cloud as input and outputs
"PointFragments" which contains all the information one needs
to draw the points on the screen space
"""
# Class to store the outputs of point rasterization


class PointFragments(NamedTuple):
    idx: torch.Tensor
    zbuf: torch.Tensor
    qvalue: torch.Tensor
    scaler: torch.Tensor
    occupancy: torch.Tensor


class PointsRasterizationSettings:
    """
    Attributes:
    TODO: complete documentations
        cutoff_threshold (float): deciding whether inside the splat based on Q < cutoff_threshold
        backface_culling (bool): render only the front-facing faces
        depth_merging_threshold (float): depth threshold for determine zbuffer
        Vrk_invariant: use 3D spheres of a uniform size to represent a point
        Vrk_isotropic: use isotropic gaussian in the source space, otherwise compute anisotropic
            gaussian variance
        radii_backward_scaler (float): the scaler used to increase the splat radius
            for backward pass, used for OccRBFBackward and OccBackward. If zero,
            then use WeightBackward, which contains gradients only for points
            rendered at the pixel.
        points_per_pixels (int): rasterize maximum of K points per pixel
        bin_size (int): used for faster forward-pass
        clip_pts_grad (float): clip per-point gradient using the gradient norm
        antialiasing_sigma (float): gaussian sigma for anti-aliasing
    """
    __slots__ = [
        "cutoff_threshold",
        "backface_culling",
        "depth_merging_threshold",
        "Vrk_invariant",
        "Vrk_isotropic",
        "radii_backward_scaler",
        "image_size",
        "points_per_pixel",
        "bin_size",
        "max_points_per_bin",
        "clip_pts_grad",
        "antialiasing_sigma",
    ]

    def __init__(
        self,
        backface_culling: bool = True,
        cutoff_threshold: float = 1,
        depth_merging_threshold: float = 0.05,
        Vrk_invariant: bool = False,
        Vrk_isotropic: bool = True,
        radii_backward_scaler: float = 10,
        image_size: int = 256,
        points_per_pixel: int = 8,
        bin_size: Optional[int] = 0,
        max_points_per_bin: Optional[int] = None,
        clip_pts_grad: Optional[float] = -1,
        antialiasing_sigma: Optional[float] = 1.0,
    ):
        self.cutoff_threshold = cutoff_threshold
        self.backface_culling = backface_culling
        self.depth_merging_threshold = depth_merging_threshold
        self.Vrk_invariant = Vrk_invariant
        self.Vrk_isotropic = Vrk_isotropic
        self.radii_backward_scaler = radii_backward_scaler
        self.image_size = image_size
        self.points_per_pixel = points_per_pixel
        self.bin_size = bin_size
        self.max_points_per_bin = max_points_per_bin
        self.clip_pts_grad = clip_pts_grad
        self.antialiasing_sigma = antialiasing_sigma


class SurfaceSplatting(PointsRasterizer):
    """
    SurfaceSplatting rasterizer Differentiable Surface Splatting
    Outputs per point:
        1. the screen space extent (screen range (-1, 1))
        2. center of projection (screen range (-1, 1))
    """

    def __init__(self, cameras=None, raster_settings=None, frnn_radius=0.2):
        """
        cameras: A cameras object which has a  `transform_points` method
            which returns the transformed points after applying the
            world-to-view and view-to-screen transformations.
        compositor: Use SurfaceSplatting compositor by default unless user overwrites it, in that case,
            issue a warning
        raster_settings: the parameters for rasterization. This should be a
            named tuple.

        All these initial settings can be overridden by passing keyword
        arguments to the forward function.
        """
        if raster_settings is None:
            raster_settings = PointsRasterizationSettings()
        super().__init__(cameras=cameras, raster_settings=raster_settings)

        self.raster_settings = raster_settings
        self.frnn_radius = frnn_radius

    def transform_normals(self, point_clouds, **kwargs):
        """ Return normals in view coordinates (padded) """
        self.cameras = kwargs.get("cameras", self.cameras)
        if self.cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of PointsRasterizer"
            raise ValueError(msg)

        view_transform = self.cameras.get_world_to_view_transform()

        # calling normals_padded() will also create points_padded,
        # so must enable grad
        with torch.autograd.enable_grad():
            normals = point_clouds.normals_padded()

        normals_in_view = view_transform.transform_normals(normals)
        return normals_in_view

    def _filter_backface_points(self, point_clouds, **kwargs):
        with torch.autograd.no_grad():
            normals_view = self.transform_normals(point_clouds, **kwargs)
            # mask = (normals_view[:, :, 2] < 1e-3)  # error buffer
            mask = normals_view[:, :, 2] < 0

        lengths = point_clouds.num_points_per_cloud()
        mask_packed = ops3d.padded_to_packed(mask.float(), point_clouds.cloud_to_packed_first_idx(),
                                             lengths.sum().item()).bool()
        if torch.all(mask_packed):
            return point_clouds, mask_packed

        # create point clouds again from packed
        points_padded = point_clouds.points_padded()
        normals_padded = point_clouds.normals_padded()
        features_padded = point_clouds.features_padded()

        # tuple to list, since pointclouds class doesn't accept tuples
        points_list = [points_padded[b][mask[b]]
                       for b in range(points_padded.shape[0])]
        normals_list = [normals_padded[b][mask[b]]
                        for b in range(normals_padded.shape[0])]
        if features_padded is not None:
            features_list = [features_padded[b][mask[b]]
                             for b in range(features_padded.shape[0])]
            new_point_clouds = point_clouds.__class__(
                points=points_list, normals=normals_list, features=features_list)
        else:
            new_point_clouds = point_clouds.__class__(
                points=points_list, normals=normals_list)

        # for k in per_point_info:
        #     per_point_info[k] = per_point_info[k][mask_packed]
        return new_point_clouds, mask_packed

    def _filter_points_with_invalid_depth(self, point_clouds, **kwargs):
        self.cameras = kwargs.get('cameras', self.cameras)
        lengths = point_clouds.num_points_per_cloud()
        points_padded = point_clouds.points_padded()
        with torch.autograd.no_grad():
            to_view = self.cameras.get_world_to_view_transform()
            points = to_view.transform_points(points_padded)
            znear = getattr(self.cameras, 'znear', kwargs.get('znear', 1.0))
            zfar = getattr(self.cameras, 'zfar', kwargs.get('zfar', 100.0))
            mask = (points[..., 2] >= znear) & (points[..., 2] <= zfar)

        mask_packed = ops3d.padded_to_packed(mask.float(
        ), point_clouds.cloud_to_packed_first_idx(), lengths.sum().item()).bool()
        if torch.all(mask_packed):
            return point_clouds, mask_packed

        # create point clouds again from packed
        points_padded = point_clouds.points_padded()
        normals_padded = point_clouds.normals_padded()
        features_padded = point_clouds.features_padded()
        # tuple to list, since pointclouds class doesn't accept tuples
        points_list = [points_padded[b][mask[b]]
                       for b in range(points_padded.shape[0])]
        normals_list = features_list = None
        if normals_padded is not None:
            normals_list = [normals_padded[b][mask[b]]
                            for b in range(normals_padded.shape[0])]
        if features_padded is not None:
            features_list = [features_padded[b][mask[b]]
                             for b in range(features_padded.shape[0])]
        new_point_clouds = point_clouds.__class__(
            points=points_list, normals=normals_list, features=features_list)
        # for k in per_point_info:
        #     per_point_info[k] = per_point_info[k][mask_packed]
        return new_point_clouds, mask_packed

    def filter_renderable(self, point_clouds, point_clouds_filter=None, **kwargs):
        raster_settings = kwargs.get("raster_settings", self.raster_settings)
        cameras = kwargs.get('cameras', self.cameras)
        if point_clouds.isempty():
            return None, None, None, None

        P = point_clouds.num_points_per_cloud().sum().item()
        max_P = point_clouds.num_points_per_cloud().max().item()
        first_idx = point_clouds.cloud_to_packed_first_idx()
        batch_size = len(point_clouds)

        if point_clouds_filter is not None:  # activation filter
            point_clouds_filter.set_filter(visibility=torch.full(
                (batch_size, max_P), False, dtype=torch.bool, device=point_clouds_filter.device))
            point_clouds = point_clouds_filter.filter_with(
                point_clouds, ('activation',))

        if cameras.R.shape[0] != len(point_clouds):
            logger_py.warning('Detected unequal number of cameras and pointclouds. '
                              'Call point_clouds.extend(len(cameras)) outside this function, '
                              'otherwise inplace modification to pointclouds will not take effects.')
            point_clouds = point_clouds.extend(cameras.R.shape[0])

        point_clouds, valid_depth_mask = self._filter_points_with_invalid_depth(
            point_clouds, **kwargs)

        if point_clouds.isempty():
            return point_clouds, None, None, None

        # new point clouds containing only points facing towards the camera
        if raster_settings.backface_culling:
            point_clouds, frontface_mask = self._filter_backface_points(
                point_clouds, **kwargs)
            valid_depth_mask[valid_depth_mask] = frontface_mask

        return point_clouds, valid_depth_mask

    def _compute_anisotropic_Vrk(self, pointclouds, **kwargs):
        """
        determine the variance in the local surface frame h * Sk.T @ Sk based on curvature.
        Args:
            points_packed (Pointclouds3D): point clouds in object coordinates
        Returns:
            Vr (N, 3, 3): V_k^r matrix packed
        """
        with torch.autograd.enable_grad():
            points_padded = pointclouds.points_padded()

        num_points = pointclouds.num_points_per_cloud()

        # eigenvectors (=principal directions) in an ascending order of their
        # corresponding eigenvalues, while the smallest eigenvalue's eigenvector
        # corresponds to the normal direction
        curvatures, local_frame = estimate_pointcloud_local_coord_frames(
            pointclouds, neighborhood_size=8, disambiguate_directions=False)

        local_frame = ops3d.padded_to_packed(local_frame.reshape(local_frame.shape[:2] + (-1,)),
                                             pointclouds.cloud_to_packed_first_idx(),
                                             num_points.sum().item())
        curvatures = ops3d.padded_to_packed(curvatures,
                                            pointclouds.cloud_to_packed_first_idx(
                                            ), num_points.sum().item())

        local_frame = local_frame.view(-1, 3, 3)[:, :, 1:]
        # curvature only determines ratio of the two principle axis
        # the actual size is based on a global max_size
        curvatures = curvatures.view(-1, 3)[:, 1:]
        curvature_ratios = curvatures / curvatures[:, -1:]
        # TODO: compute density
        curvatures = curvatures
        Vr = local_frame @ torch.diag_embed(
            curvatures) @ local_frame.transpose(1, 2)
        return Vr, local_frame.transpose(1, 2)

    def _compute_global_Vrk(self, pointclouds, refresh=True, **kwargs):
        """
        determine variance scaler used in globally (see _compute_isotropic_Vrk)
        Args:
            pointclouds: pointclouds in object coorindates
        Returns:
            h_k: scaler
            S_k: local frame
        """
        if not refresh and self._Vrk_h is not None:
            h_k = self._Vrk_h
        else:
            # compute average density
            with torch.autograd.enable_grad():
                pts_world = pointclouds.points_padded()

            num_points_per_cloud = pointclouds.num_points_per_cloud()
            if self.frnn_radius <= 0:
                # use knn here
                # logger_py.info("vrk knn points")
                sq_dist, _, _ = ops3d.knn_points(pts_world, pts_world,
                                                 num_points_per_cloud, num_points_per_cloud,
                                                 K=7)
            else:
                sq_dist, _, _, _ = frnn.frnn_grid_points(pts_world, pts_world,
                                                         num_points_per_cloud, num_points_per_cloud,
                                                         K=7, r=self.frnn_radius)
            # logger_py.info("frnn and knn dist close: {}".format(torch.allclose(sq_dist, sq_dist2)))
            sq_dist = sq_dist[:, :, 1:]
            # knn search is unreliable, set sq_dist manually
            sq_dist[num_points_per_cloud < 7] = 1e-3
            h_k = 0.5 * sq_dist.max(dim=-1, keepdim=True)[0]
            # prevent some outlier rendered be too large, or too small
            h_k = h_k.mean(dim=1, keepdim=True).clamp(5e-5, 1e-3)
            Vrk_h = gather_batch_to_packed(
                h_k, pointclouds.packed_to_cloud_idx())

        # Sk, a transformation from 2D local surface frame to 3D world frame
        # Because isometry, two axis are equivalent, we can simply
        # find two 3d vectors perpendicular to the point normals
        # (totalP, 2, 3)
        with torch.autograd.enable_grad():
            normals = pointclouds.normals_packed()

        u0 = F.normalize(torch.cross(normals,
                                     normals + torch.rand_like(normals)), dim=-1)
        u1 = F.normalize(torch.cross(normals, u0), dim=-1)
        Sk = torch.stack([u0, u1], dim=1)
        Vrk = Vrk_h.view(-1, 1, 1) * Sk.transpose(1, 2) @ Sk
        return Vrk, Sk

    def _compute_isotropic_Vrk(self, pointclouds, refresh=True, **kwargs):
        """
        determine the variance in the local surface frame h * Sk.T @ Sk,
        where Sk is 2x3 local surface coordinate to world coordinate.
        determine the h_k in V_k^r = h_k*Id using nearest neighbor
        heuristically h_k = mean(dist between points in a small neighbor)
        The larger h_k is, the larger the splat is
        NOTE: h_k in inverse to the definition in the paper, the larger h_k, the
            larger the splats
        Args:
            pointclouds: pointcloud in object coordinate
        Returns:
            h_k: [N,3,3] tensor for each point
            S_k: [N,2,3] local frame
        """
        if not refresh and self._Vrk_h is not None and \
                pointclouds.num_points_per_cloud().sum() == self._Vrk_h.shape[0]:
            pass
        else:
            with torch.autograd.enable_grad():
                pts_world = pointclouds.points_padded()

            num_points_per_cloud = pointclouds.num_points_per_cloud()
            if self.frnn_radius <= 0:
                # logger_py.info("vrk knn points")
                sq_dist, _, _ = ops3d.knn_points(pts_world, pts_world,
                                                 num_points_per_cloud, num_points_per_cloud,
                                                 K=7)
            else:
                sq_dist, _, _, _ = frnn.frnn_grid_points(pts_world, pts_world,
                                                         num_points_per_cloud, num_points_per_cloud,
                                                         K=7, r=self.frnn_radius)

            sq_dist = sq_dist[:, :, 1:]
            # knn search is unreliable, set sq_dist manually
            sq_dist[num_points_per_cloud < 7] = 1e-3
            # (totalP, knnK)
            sq_dist = ops3d.padded_to_packed(sq_dist,
                                             pointclouds.cloud_to_packed_first_idx(
                                             ), num_points_per_cloud.sum().item())
            # [totalP, ]
            h_k = 0.5 * sq_dist.max(dim=-1, keepdim=True)[0]

            # prevent some outlier rendered be too large, or too small
            self._Vrk_h = h_k.clamp(5e-5, 0.01)

        # Sk, a transformation from 2D local surface frame to 3D world frame
        # Because isometry, two axis are equivalent, we can simply
        # find two 3d vectors perpendicular to the point normals
        # (totalP, 2, 3)
        with torch.autograd.enable_grad():
            normals = pointclouds.normals_packed()

        u0 = F.normalize(torch.cross(normals,
                                     normals + torch.rand_like(normals)), dim=-1)
        u1 = F.normalize(torch.cross(normals, u0), dim=-1)
        Sk = torch.stack([u0, u1], dim=1)
        Vrk = self._Vrk_h.view(-1, 1, 1) * Sk.transpose(1, 2) @ Sk
        return Vrk, Sk

    def _compute_variance_and_detMk(self, pointclouds, **kwargs):
        """
        Compute the projected kernel variance Vk'+I Eq.(35) in [2],
        J V_k^r J^T + I Eq.(7) in [1]
        Args:
            pointclouds (PointClouds3D): point clouds in object coordinates
        Returns:
            variance (tensor): (N, 2, 2)
            detMk (tensor): (N, 1) determinant of Mk
        """
        raster_settings = kwargs.get("raster_settings", self.raster_settings)

        WJk = self._compute_WJk(pointclouds, **kwargs)
        totalP = WJk.shape[0]

        if raster_settings.Vrk_invariant:
            Vrk, Sk = self._compute_global_Vrk(pointclouds, **kwargs)
        elif raster_settings.Vrk_isotropic:
            # (N, 3, 3)
            Vrk, Sk = self._compute_isotropic_Vrk(
                pointclouds, **kwargs)
        else:
            Vrk, Sk = self._compute_anisotropic_Vrk(pointclouds)

        Mk = Sk @ WJk
        Vk = WJk.transpose(1, 2) @ Vrk @ WJk

        # low-pass filter +sigma*I
        # NOTE: [2] is in pixel space, but we are in NDC space, so the variance should be
        # scaled by pixel_size
        pixel_size = 2.0 / raster_settings.image_size
        variance = Vk + raster_settings.antialiasing_sigma * \
            ops3d.eyes(2, totalP, device=Vk.device,
                       dtype=Vk.dtype) * (pixel_size**2)

        detMk = torch.det(Mk)

        return variance, detMk

    def _compute_WJk(self, point_clouds, **kwargs):
        """
        Compute the Jacobian of the projection matrix
        which is the projection from camera -> view space (NDC)
        This function should accomodate different camera models, so we use
        cameras.get_full_projection_transform()
        Mk consists of
        W  (3x4) world coordinates to view coordinates (NDC)
        Jk (4x2) projection jacobian
        Args:
            point_clouds (PointClouds3D)
            cameras
        Returns:
            Mk (tensor): (N, 3, 2) Jacobian screen xy wrt world (xyz)
        """
        self.cameras = kwargs.get("cameras", self.cameras)
        with torch.autograd.enable_grad():
            pts_packed = point_clouds.points_packed()

        num_pts = pts_packed.shape[0]

        # the Jacobian/affine transformation doesn't need the last row
        cam_proj_trans = self.cameras.get_full_projection_transform()
        M44 = cam_proj_trans.get_matrix()
        W = M44[..., :3, :]
        W = gather_batch_to_packed(W, point_clouds.packed_to_cloud_idx())
        # the projection jacobian (P,3,2)
        # is always x,y,z dividing the 4th column
        # [2] Eq.(34)
        # 1/t Id + (-1/t/t)M[:, 3], where t = (x,y,z,1)*M[:, 3]
        pts_packed_hom = to_homogen(pts_packed, dim=-1)
        denom = pts_packed_hom[:, None, :] @ gather_batch_to_packed(
            M44[..., 3:], point_clouds.packed_to_cloud_idx())  # P,1,1
        denom = denom.view(-1)

        # TODO: numerical instability! denom can have small absolute value
        # leads to even smaller denom**2, yielding extremely large Jk, detMk
        # variance Vk, inverseVk (used to compute ellipse parameters), eventually
        # result in extremely large radii
        denom_sqr = eps_denom(denom ** 2)
        Jk = pts_packed_hom.new_zeros([num_pts, 4, 2])
        denom = eps_denom(denom)
        Jk[:, 0, 0] = 1 / denom
        Jk[:, 1, 1] = 1 / denom
        # from padded (N,4,2) to (P,4,2)
        Wk = gather_batch_to_packed(
            M44[..., :2], point_clouds.packed_to_cloud_idx())
        xy_view = pts_packed_hom[:, None, :] @ Wk  # (P,1,2)
        Jk[:, 3, 0] = -1 / denom_sqr * xy_view[:, :, 0].view(-1)
        Jk[:, 3, 1] = -1 / denom_sqr * xy_view[:, :, 1].view(-1)

        Mk = W @ Jk

        return Mk

    def _get_ellipse_axis_aligned_radius(self, cutoffC, ellipseParams,
                                         min_radii=-float('inf'), max_radii=float('inf')):
        """
        Compute the axis-aligned radii around each points in screen space given the
        elliptical parameters a, b, c in ax^2 + cy^2 + bxy <= cutoffC
        Args:
            cutoffC (scalar): cutoff threshold
            ellipseParams (tensor): (N,3) coefficients a, b, c of the elliptical function
        Returns:
            radii (tensor): (N,2) splat radius in x and y direction (in NDC)
        """
        with torch.autograd.no_grad():
            a = ellipseParams[..., 0]
            b = ellipseParams[..., 1]
            c = ellipseParams[..., 2]
            # N,2
            b2 = b**2
            ac4 = 4 * a * c
            denom = eps_denom(ac4 - b2)
            y = torch.sqrt(eps_sqrt(4 * a * cutoffC / denom))
            x = torch.sqrt(eps_sqrt(4 * c * cutoffC / denom))
            radii = torch.stack([x, y], dim=-1)
            # radii = radii.clamp(min=min_radii, max=max_radii)
            # cutoffC = torch.max(x**2 * denom / 4 / c, y**2 * denom / 4 / a)
            cutoffC = torch.full_like(a, cutoffC)
        return radii, cutoffC

    def _get_per_point_info(self, pointclouds, **kwargs):
        """
        Compute necessary per-point information to rasterize
        NOTE: alternatively return the max radii, reduces memory, but
        coule be inaccurate because falsely large points could cover
        actually points that are in the back.
        Returns:
            radii (tensor): (N, 2) splat radii in x and y
            ellipse_params (tensor): (N, 3) ax^2 + bxy + cy^2
            scalar (tensor): (N) scalar for Screen space EWA filter
        """
        raster_settings = kwargs.get("raster_settings", self.raster_settings)
        cutoff_thres = raster_settings.cutoff_threshold

        # GV = M_k V_k^T M_k^T + V^h
        GVs, detMk = self._compute_variance_and_detMk(pointclouds, **kwargs)

        GVdets = torch.det(GVs)
        GVinvs = torch.inverse(GVs)
        totalP = detMk.shape[0]

        # compute ellipse radii (N, 2) to prepare the rasterization
        # ellipseParams (a,b,c) = ax^2+cxy+by^2
        ellipseParams = GVs.new_empty(totalP, 3)
        ellipseParams[..., 0] = GVinvs[..., 0, 0]
        ellipseParams[..., 1] = GVinvs[..., 0, 1] + GVinvs[..., 1, 0]
        ellipseParams[..., 2] = GVinvs[..., 1, 1]

        # NOTE: make radii at least one pixel height/width
        pixel_size = 2 / (raster_settings.image_size - 1)
        radii, cutoff_thres = self._get_ellipse_axis_aligned_radius(
            cutoff_thres, ellipseParams, min_radii=pixel_size, max_radii=20 * pixel_size)
        # gaussian normalization term 2pi|V|^{1/2} (N,1)
        scalerk = torch.sqrt(eps_sqrt(GVdets * 4 * np.pi * np.pi))
        scalerk = detMk.abs() / eps_denom(scalerk)

        # radii = radii.clamp(min=0.5*pixel_size, max=5*pixel_size)
        return {"radii": radii.detach(),
                "ellipse_params": ellipseParams.detach(),
                "cutoff_threshold": cutoff_thres.detach(),
                "scaler": scalerk.detach()}

    def _empty_fragments(self, batch_size, **kwargs):
        """
        Templates for an empty rasterization output
        """
        raster_settings = kwargs.get("raster_settings", self.raster_settings)
        S = raster_settings.image_size
        P = raster_settings.points_per_pixel
        idx = torch.full((batch_size, S, S, P), -1,
                         dtype=torch.long, device=self.device)
        zbuf = torch.full((batch_size, S, S, P), -1.0,
                          dtype=torch.float, device=self.device)
        qvalue_map = torch.full(
            (batch_size, S, S, P), -1.0, dtype=torch.float, device=self.device)
        occ_map = torch.full((batch_size, S, S), 0,
                             dtype=torch.float, device=self.device)
        return PointFragments(idx=idx, zbuf=zbuf, qvalue=qvalue_map, occupancy=occ_map)

    def forward(self, point_clouds, point_clouds_filter=None, **kwargs) -> PointFragments:
        """
        Args:
            point_clouds (Pointclouds3D): a set of point clouds with coordinates.
            per_point_info (dict):
                radii_packed: (N,2) axis-aligned radii in packed form
                ellipse_params_packed: (N,3) ellipse parameters in packed form
        Returns:
            PointFragments: Rasterization outputs as a named tuple.
        """
        raster_settings = kwargs.get("raster_settings", self.raster_settings)
        cameras = kwargs.get('cameras', self.cameras)
        max_P = point_clouds.num_points_per_cloud().max().item()
        total_P = point_clouds.num_points_per_cloud().sum().item()

        point_clouds_filtered, mask_filtered = self.filter_renderable(
            point_clouds, point_clouds_filter, **kwargs)

        if point_clouds_filtered.isempty():
            return self._empty_fragments(cameras.R.shape[0], **kwargs)

        # compute per-point features for elliptical gaussian weights
        with torch.autograd.no_grad():
            per_point_info = self._get_per_point_info(
                point_clouds_filtered, **kwargs)


        _tmp = point_clouds_filtered.points_padded()
        if _tmp.requires_grad:
            _tmp.register_hook(lambda x: _check_grad(x, 'transform'))
        pcls_screen = self.transform(point_clouds_filtered, **kwargs)

        idx, zbuf, qvalue_map, occ_map = rasterize_elliptical_points(
            pcls_screen,
            per_point_info["ellipse_params"],
            per_point_info['cutoff_threshold'],
            per_point_info["radii"],
            depth_merging_threshold=raster_settings.depth_merging_threshold,
            image_size=raster_settings.image_size,
            points_per_pixel=raster_settings.points_per_pixel,
            bin_size=raster_settings.bin_size,
            max_points_per_bin=raster_settings.max_points_per_bin,
            radii_backward_scaler=raster_settings.radii_backward_scaler,
            clip_pts_grad=raster_settings.clip_pts_grad
        )

        # compute weight: scalar*exp(-0.5Q)
        frag_scaler = gather_with_neg_idx(
            per_point_info['scaler'], 0, idx.view(-1).long())
        frag_scaler = frag_scaler.view_as(qvalue_map)

        fragments = PointFragments(
            idx=idx, zbuf=zbuf, qvalue=qvalue_map, scaler=frag_scaler, occupancy=occ_map)

        # returns (P,) boolean mask for visibility
        visibility_mask = get_per_point_visibility_mask(
            point_clouds_filtered, fragments)
        mask_filtered[mask_filtered] = visibility_mask

        if point_clouds_filter is not None:
            # update point_clouds visibility filter
            # we use this information in projection loss
            # put all_depth_visibility_mask (num_active) to original_visibility_mask (P,)
            # transform to padded
            # original_visibility_mask = ops3d.packed_to_padded(
            #     valid_depth_mask.float(), first_idx, max_P).bool()
            # lixin
            original_visibility_mask = ops3d.packed_to_padded(
                mask_filtered.float(), point_clouds.cloud_to_packed_first_idx(), max_P).bool()
            point_clouds_filter.set_filter(visibility=original_visibility_mask)

        if kwargs.get('verbose', False):
            # use scatter to get per point info of the original
            original_per_point_info = {}
            for k in per_point_info:
                original_per_point_info[k] = per_point_info[k].new_zeros(
                    (total_P, ) + per_point_info[k].shape[1:])
                original_per_point_info[k][mask_filtered] = per_point_info[k]

            return fragments, point_clouds_filtered, original_per_point_info
        return fragments, point_clouds_filtered


def _clip_grad(value=0.1):
    def func(grad):
        scaler = grad.norm(dim=-1, keepdim=True).clamp(0, value)
        grad = torch.nn.functional.normalize(grad, dim=-1) * scaler
        # grad.clamp_(-value, value)
        return grad
    return func

def _check_grad(x, msg):
    from ..utils import valid_value_mask
    if not valid_value_mask(x).all():
        print(msg)
    return x

def rasterize_elliptical_points(pcls_screen, ellipse_params,
                                cutoff_threshold, radii,
                                depth_merging_threshold: float = 0.05,
                                image_size: int = 512,
                                points_per_pixel: int = 5,
                                bin_size: Optional[int] = None,
                                max_points_per_bin: Optional[int] = None,
                                radii_backward_scaler: float = 10.0,
                                clip_pts_grad: float = -1.0):
    """
    Similar to default point rasterizer, with following differences:
    0. use per-point axis-aligned radii to check bin
    1. check_inside(pixel, point) by a) axis-aligned radii and
        the ellipse funciton f(x)<cutoff_threshold
    2. output f(x) for compositor function, since f_x(x) is analytically derivable
    Args:
        pcls_screen (tensor): (N, 3) pts in screen space (NDC) coordinates
        ellipse_params (tensor): (N, 3) ellipse parameters per splat,
            the (a,b,c) in ax^2 + bxy + cy^2
        radii (tensor): (N, 2) axis-aligned radius
    Returns:
        PointFragments containing [B,C,H,W] maps of
            idx (C=1), f(x) (C=1), zbuf (C=1)
    """
    points_packed = pcls_screen.points_packed()
    cloud_to_packed_first_idx = pcls_screen.cloud_to_packed_first_idx()
    num_points_per_cloud = pcls_screen.num_points_per_cloud()

    cutoff_threshold = cutoff_threshold.expand(points_packed.shape[0])
    # Binning part is from pytorch3d, it creates a local pixel-to-point search
    # list, so that when we iterate over the pixels in the bins, we don't need to iterate *all*
    # the points, but only those that are inside the bin.
    if bin_size is None:
        # TODO: These heuristics are not well-thought out!
        if image_size <= 64:
            bin_size = 8
        elif image_size <= 256:
            bin_size = 16
        elif image_size <= 512:
            bin_size = 32
        elif image_size <= 1024:
            bin_size = 64
    if bin_size != 0:
        # There is a limit on the number of points per bin in the cuda kernel.
        num_bins = 1 + (image_size - 1) // bin_size
        if num_bins >= kMaxPointsPerBin:
            raise ValueError(
                "bin_size too small, number of bins must be less than %d; got %d"
                % (kMaxPointsPerBin, num_bins)
            )

    if max_points_per_bin is None:
        max_points_per_bin = int(max(10000, num_points_per_cloud.max()))

    if points_packed.requires_grad and clip_pts_grad > 0:
        points_packed.register_hook(_clip_grad(clip_pts_grad))
        points_packed.register_hook(lambda x: _check_grad(x, 'elliptical_rasterizer'))

    idx, zbuf, qvalue_map, occ_map = EllipticalRasterizer.apply(
        points_packed, ellipse_params, cutoff_threshold, radii,
        cloud_to_packed_first_idx, num_points_per_cloud,
        depth_merging_threshold, image_size, points_per_pixel, bin_size, max_points_per_bin,
        radii_backward_scaler)
    return idx, zbuf, qvalue_map, occ_map


class EllipticalRasterizer(autograd.Function):
    @staticmethod
    def forward(ctx, pts_screen, ellipse_param, cutoff_threshold, radii,
                cloud_to_packed_first_idx, num_points_per_cloud,
                depth_merging_threshold,
                image_size,
                points_per_pixel,
                bin_size: int = 0,
                max_points_per_bin: int = 0,
                radii_backward_scaler: float = 10.0,
                ):
        """
        TODO: save if bin_points if bin_size is not 0, and reuse in the backward pass
        """
        args = (
            pts_screen,
            ellipse_param,
            cutoff_threshold,
            radii,
            cloud_to_packed_first_idx,
            num_points_per_cloud,
            depth_merging_threshold,
            image_size,
            points_per_pixel,
            bin_size,
            max_points_per_bin,
        )
        idx, zbuf, qvalue_map, occ_map = _C.splat_points(*args)

        ctx.radii_backward_scaler = radii_backward_scaler
        ctx.depth_merging_threshold = depth_merging_threshold
        if radii_backward_scaler == 0:
            ctx.save_for_backward(
                pts_screen, idx, ellipse_param, cutoff_threshold)
        else:
            zbuf0 = zbuf[..., 0].clone()
            ctx.save_for_backward(pts_screen, ellipse_param, cutoff_threshold, radii, idx, zbuf0,
                                  cloud_to_packed_first_idx, num_points_per_cloud)
        return idx, zbuf, qvalue_map, occ_map

    @staticmethod
    def backward(ctx, idx_grad, zbuf_grad, qvalue_grad, occ_grad):
        # idx_grad and zbuf_grad are None (unless maybe we make weights depend on z? i.e. volumetric splatting)

        grad_radii = None
        grad_cloud_to_packed_first_idx = None
        grad_num_points_per_cloud = None
        grad_cutoff_thres = None
        grad_depth_merging_thres = None
        grad_image_size = None
        grad_points_per_pixel = None
        grad_bin_size = None
        grad_max_points_per_bin = None
        grad_radii_s = None
        grad_backward_rbf = None

        grads = (grad_cutoff_thres, grad_radii, grad_cloud_to_packed_first_idx,
                 grad_num_points_per_cloud,
                 grad_depth_merging_thres, grad_image_size, grad_points_per_pixel,
                 grad_bin_size, grad_max_points_per_bin, grad_radii_s, grad_backward_rbf)

        radii_s = ctx.radii_backward_scaler

        # either use OccRBFBackward or use OccBackward
        pts_screen, ellipse_param, cutoff_threshold, radii, idx, zbuf0, \
            cloud_to_packed_first_idx, num_points_per_cloud, \
            = ctx.saved_tensors
        depth_merging_threshold = ctx.depth_merging_threshold

        backward_occ_fast = True
        if not backward_occ_fast:
            device = pts_screen.device
            grads_input_xy = pts_screen.new_zeros((pts_screen.shape[0], 2))
            grads_input_z = pts_screen.new_zeros((pts_screen.shape[0], 1))
            mask = (idx[..., 0] >= 0).bool()  # float
            pts_visibility = torch.full(
                (pts_screen.shape[0],), False, dtype=torch.bool, device=pts_screen.device)
            # all rendered points (indices in packed points)
            visible_idx = idx[mask].unique().long().view(-1)
            visible_idx = visible_idx[visible_idx >= 0]
            pts_visibility[visible_idx] = True
            num_points_per_cloud = torch.stack([x.sum() for x in torch.split(
                pts_visibility, num_points_per_cloud.tolist(), dim=0)])
            cloud_to_packed_first_idx = num_points_2_cloud_to_packed_first_idx(
                num_points_per_cloud)

            pts_screen = pts_screen[pts_visibility]
            radii = radii[pts_visibility]
            grad_visible = _C._splat_points_occ_backward(pts_screen, radii, occ_grad,
                                                        cloud_to_packed_first_idx, num_points_per_cloud,
                                                        radii_s, depth_merging_threshold)
            if torch.isnan(grad_visible).any() or not torch.isfinite(grad_visible).all():
                print('invalid grad_visible')
            assert(pts_visibility.sum() == grad_visible.shape[0])
            grads_input_xy[pts_visibility] = grad_visible
            _C._backward_zbuf(idx, zbuf_grad, grads_input_z)
            # TODO necessary to concatenate
            grads_input = torch.cat([grads_input_xy, grads_input_z], dim=-1)
        else:
            """
            We only care about rasterized points (visible points)
            1. Filter [P,*] data to [P_visible,*] data
            2. Fast backward cuda
                2a. call FRNN insertion
                2b. count_sort
            """
            device = pts_screen.device
            mask = (idx[..., 0] >= 0).bool()  # float
            pts_visibility = torch.full(
                (pts_screen.shape[0],), False, dtype=torch.bool, device=pts_screen.device)
            # all rendered points (indices in packed points)
            visible_idx = idx[mask].unique().long().view(-1)
            visible_idx = visible_idx[visible_idx >= 0]
            pts_visibility[visible_idx] = True
            num_points_per_cloud = torch.stack([x.sum() for x in torch.split(
                pts_visibility, num_points_per_cloud.tolist(), dim=0)])
            cloud_to_packed_first_idx = num_points_2_cloud_to_packed_first_idx(
                num_points_per_cloud)

            pts_screen_visible = pts_screen[pts_visibility]
            radii_visible = radii[pts_visibility]

            #####################################
            #  2a. call FRNN insertion
            #####################################
            N = num_points_per_cloud.shape[0]
            P = pts_screen_visible.shape[0]
            assert(num_points_per_cloud.sum().item()==P)
            # from frnn.frnn import GRID_PARAMS_SIZE, MAX_RES, prefix_sum_cuda
            # imported from
            from prefix_sum import prefix_sum_cuda
            GRID_2D_PARAMS_SIZE = 6
            GRID_2D_MAX_RES = 1024
            GRID_2D_DELTA = 2
            GRID_2D_TOTAL = 5
            RADIUS_CELL_RATIO = 2
            # first convert to padded
            max_P = num_points_per_cloud.max().item()
            pts_padded = ops3d.packed_to_padded(pts_screen_visible, cloud_to_packed_first_idx, max_P)
            radii_padded = ops3d.packed_to_padded(radii_visible, cloud_to_packed_first_idx, max_P)
            # determine search radius as max(radii)*radii_s
            search_radius = torch.tensor([radii_padded[i, :num_points_per_cloud[i]].median() * radii_s for i in range(N)], dtype=torch.float, device=device)
            # create grid from scratch
            # setup grid params
            grid_params_cuda = torch.zeros((N, GRID_2D_PARAMS_SIZE), dtype=torch.float, device=pts_padded.device)
            G = -1
            pts_padded_2D = pts_padded[:, :, :2].clone().contiguous()
            for i in range(N):
                # 0-2 grid_min; 3 grid_delta; 4-6 grid_res; 7 grid_total
                grid_min = pts_padded_2D[i, :num_points_per_cloud[i]].min(dim=0)[0]
                grid_max = pts_padded_2D[i, :num_points_per_cloud[i]].max(dim=0)[0]
                grid_params_cuda[i, :GRID_2D_DELTA] = grid_min
                grid_size = grid_max - grid_min
                cell_size = search_radius[i].item() / RADIUS_CELL_RATIO
                if cell_size < grid_size.min()/GRID_2D_MAX_RES:
                    cell_size = grid_size.min() / GRID_2D_MAX_RES
                grid_params_cuda[i, GRID_2D_DELTA] = 1 / cell_size
                grid_params_cuda[i, GRID_2D_DELTA+1:GRID_2D_TOTAL] = torch.floor(grid_size / cell_size) + 1
                grid_params_cuda[i, GRID_2D_TOTAL] = torch.prod(grid_params_cuda[i, GRID_2D_DELTA+1:GRID_2D_TOTAL])
                if G < grid_params_cuda[i, GRID_2D_TOTAL]:
                    G = int(grid_params_cuda[i, GRID_2D_TOTAL].item())

            # insert points into the grid
            pc_grid_cnt = torch.zeros((N, G), dtype=torch.int, device=device)
            pc_grid_cell = torch.full((N, max_P), -1, dtype=torch.int, device=device)
            pc_grid_idx = torch.full((N, max_P), -1, dtype=torch.int, device=device)
            frnn._C.insert_points_cuda(pts_padded_2D, num_points_per_cloud, grid_params_cuda, pc_grid_cnt, pc_grid_cell, pc_grid_idx, G)

            # use prefix_sum from Matt Dean
            grid_params = grid_params_cuda.cpu()
            pc_grid_off = torch.full((N, G), 0, dtype=torch.int, device=device)
            for i in range(N):
                prefix_sum_cuda(pc_grid_cnt[i], grid_params[i, GRID_2D_TOTAL], pc_grid_off[i])

            # sort points according to their grid positions and insertion orders
            # sort based on x, y first. Then we will use points_sorted_idxs to recover the points_sorted with Z
            points_sorted = torch.zeros((N, max_P, 2), dtype=torch.float, device=device)
            points_sorted_idxs = torch.full((N, max_P), -1, dtype=torch.int, device=device)
            frnn._C.counting_sort_cuda(
                pts_padded_2D,
                num_points_per_cloud,
                pc_grid_cell,
                pc_grid_idx,
                pc_grid_off,
                points_sorted,      # (N,P,2)
                points_sorted_idxs  # (N,P)
            )
            new_points_sorted = torch.zeros_like(pts_padded)
            for i in range(N):
                points_sorted_idxs_i = points_sorted_idxs[i, :num_points_per_cloud[i]].long().unsqueeze(1).expand(-1, 3)
                new_points_sorted[i, :num_points_per_cloud[i]] = torch.gather(pts_padded[i], 0, points_sorted_idxs_i)
                # print(points_sorted[i, :10])
                # print(new_points_sorted[i, :10])
            # new_points_sorted = torch.gather(pts_padded, 1, points_sorted_idxs.long().unsqueeze(2).expand(-1, -1, 3))

            assert(new_points_sorted is not None and pc_grid_off is not None and points_sorted_idxs is not None and grid_params_cuda is not None)
            # convert sorted_points and sorted_points_idxs to packed (P, )
            points_sorted = ops3d.padded_to_packed(new_points_sorted, cloud_to_packed_first_idx, P)
            # padded_to_packed only supports torch.float32...
            shifted_points_sorted_idxs = points_sorted_idxs+cloud_to_packed_first_idx.float().unsqueeze(1)
            points_sorted_idxs = ops3d.padded_to_packed(shifted_points_sorted_idxs, cloud_to_packed_first_idx, P)
            points_sorted_idxs_2D = points_sorted_idxs.long().unsqueeze(1).expand(-1, 2)
            radii_sorted = torch.gather(radii_visible, 0, points_sorted_idxs_2D)
            pc_grid_off += cloud_to_packed_first_idx.unsqueeze(1)
            grad_sorted = _C._splat_points_occ_fast_cuda_backward(points_sorted, radii_sorted, search_radius, occ_grad,
                num_points_per_cloud, cloud_to_packed_first_idx, pc_grid_off, grid_params_cuda)
            # grad_sorted_slow = _C._splat_points_occ_backward(points_sorted, radii_sorted,
            #                                             occ_grad, cloud_to_packed_first_idx, num_points_per_cloud,
            #                                             radii_s, depth_merging_threshold)
            # breakpoint()
            # points_sorted_idxs_3D = points_sorted_idxs.long().unsqueeze(1).expand(-1, 3)
            # print(points_sorted_idxs_3D.max(), grad_sorted.shape[0])
            grad_visible = torch.zeros_like(grad_sorted).scatter_(0, points_sorted_idxs_2D, grad_sorted)
            # grad_visible_slow = _C._splat_points_occ_backward(pts_screen[pts_visibility], radii[pts_visibility],
            #                                             occ_grad, cloud_to_packed_first_idx, num_points_per_cloud,
            #                                             radii_s, depth_merging_threshold)
            # breakpoint()
            if torch.isnan(grad_visible).any() or not torch.isfinite(grad_visible).all():
                print('invalid grad_visible')
            assert(pts_visibility.sum() == grad_visible.shape[0])
            grads_input_xy = pts_screen.new_zeros(pts_screen.shape[0], 2)
            grads_input_z = pts_screen.new_zeros(pts_screen.shape[0], 1)
            # print("1")
            grads_input_xy[pts_visibility] = grad_visible
            _C._backward_zbuf(idx, zbuf_grad, grads_input_z)
            grads_input = torch.cat([grads_input_xy, grads_input_z], dim=-1)
            # print("2")

        pts_grad = grads_input

        return (pts_grad, None) + grads


__all__ = [k for k in globals().keys() if not k.startswith("_")]
