import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch3d.ops as ops3d
from pytorch3d.renderer.cameras import (FoVOrthographicCameras,
                                        FoVPerspectiveCameras,
                                        PerspectiveCameras,
                                        OrthographicCameras)
from pytorch3d.renderer import PointsRenderer, NormWeightedCompositor
from pytorch3d.renderer.compositing import weighted_sum
from ..utils import gather_with_neg_idx, gather_batch_to_packed, get_per_point_visibility_mask
from ..utils.mathHelper import eps_denom, eps_sqrt, to_homogen, estimate_pointcloud_local_coord_frames
from .rasterizer import SurfaceSplatting
from .. import logger_py
from .. import get_debugging_mode, get_debugging_tensor
import frnn

import time


__all__ = ['SurfaceSplattingRenderer']

"""
Returns a 4-Channel image for RGBA
"""


class SurfaceSplattingRenderer(PointsRenderer):

    def __init__(self, rasterizer, compositor, antialiasing_sigma: float = 1.0,
                 density: float = 1e-4, backface_culling=True, frnn_radius=-1):
        super().__init__(rasterizer, compositor)

        self.cameras = self.rasterizer.cameras
        self._Vrk_h = None
        # screen space low pass filter
        self.antialiasing_sigma = antialiasing_sigma
        self.backface_culling = backface_culling
        # average of squared distance to the nearest neighbors
        self.density = density

        if self.compositor is None:
            logger_py.info('Composite with weighted sum.')
        elif not isinstance(self.compositor, NormWeightedCompositor):
            logger_py.warning('Expect a NormWeightedCompositor, but initialized with {}'.format(
                self.compositor.__class__.__name__))

        self.frnn_radius = frnn_radius
        logger_py.error("frnn_radius: {}".format(frnn_radius))

    def to(self, device):
        super().to(device)
        self.cameras = self.cameras.to(device)
        return self

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
        curvatures = curvature_ratios * self.density
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
            # (totalP, knnK)
            sq_dist = ops3d.padded_to_packed(sq_dist,
                                             pointclouds.cloud_to_packed_first_idx(
                                             ), num_points_per_cloud.sum().item())
            h_k = 0.5 * sq_dist.max(dim=-1, keepdim=True)[0]
            self._Vrk_h = h_k = torch.mean(h_k).item()
            # prevent some outlier rendered be too large, or too small
            self._Vrk_h = h_k.clamp(5e-5, 1e-3)

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
        Vrk = self._Vrk_h * Sk.transpose(1, 2) @ Sk
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
                print(pts_world.shape)
                sq_dist, _, _, _ = frnn.frnn_grid_points(pts_world, pts_world,
                                                 num_points_per_cloud, num_points_per_cloud,
                                                 K=7, r=self.frnn_radius)
            # sq_dist_gt = sq_dist.clone().detach()
            # sq_dist_gt[sq_dist_gt > 0.5*0.5] = -1
            # for i in range(sq_dist2.shape[0]):
            #     if not torch.allclose(sq_dist2[i, :num_points_per_cloud[i]], sq_dist_gt[i, :num_points_per_cloud[i]]):
            #         # logger_py.info(sq_dist2[i, :num_points_per_cloud[i]])
            #         # logger_py.info(sq_dist_gt[i, :num_points_per_cloud[i]])
            #         logger_py.info(i)
            #         logger_py.info(num_points_per_cloud[i])
            #         exit(0)
            # logger_py.info("frnn and knn dist close: {}".format(torch.allclose(sq_dist_gt, sq_dist2)))
            # num_diffs = torch.sum(torch.abs(sq_dist2 - sq_dist_gt) > 1e-08 + torch.abs(sq_dist_gt) * 1e-5)
            # logger_py.info(float(num_diffs.item()) / float(sq_dist_gt.numel()))
            # if not torch.allclose(sq_dist_gt, sq_dist2):
                # torch.save(pts_world, "pts_world.pt")
                # torch.save(num_points_per_cloud, "num_points_per_cloud.pt")
                # logger_py.info(sq_dist_gt[0])
            #    exit(0)
            sq_dist = sq_dist[:, :, 1:]
            # knn search is unreliable, set sq_dist manually
            sq_dist[num_points_per_cloud < 7] = 1e-3
            # (totalP, knnK)
            sq_dist = ops3d.padded_to_packed(sq_dist,
                                             pointclouds.cloud_to_packed_first_idx(
                                             ), num_points_per_cloud.sum().item())
            # [totalP, ]
            # h_k = 0.5 * sq_dist.mean(dim=-1, keepdim=True)
            # h_k = 0.5 * sq_dist.median(dim=-1, keepdim=True)[0]
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
        raster_settings = kwargs.get(
            "raster_settings", self.rasterizer.raster_settings)

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
        variance = Vk + self.antialiasing_sigma * \
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
        self.cameras = self.rasterizer.cameras = kwargs.get(
            "cameras", self.cameras)
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

        # Below is also valid, but less readable
        # cam_proj_trans = cameras.get_full_projection_transform()
        # M44 = cam_proj_trans.get_matrix()
        # W = M44[..., :3, :3]
        # W = gather_batch_to_packed(W, point_clouds.packed_to_cloud_idx())

        # # the normalizing (projection) jacobian (P,3,2)
        # # is always x,y,z dividing the 4th column
        # # [2] Eq.(34)
        # # 1/t Id + (-1/t/t)M[:, 3], where t = (x,y,z,1)*M[:, 3]
        # M4 = M44[..., 3:]
        # pts_packed_hom = to_homogen(pts_packed, dim=-1)
        # denom = pts_packed_hom[:, None, :] @ M4
        # denom = eps_denom(denom)
        # denom_sqr = eps_denom(denom ** 2)
        # # from padded (N,3,1) to (P,3)
        # M4k = gather_batch_to_packed(M4, point_clouds.packed_to_cloud_idx())
        # Jk = - 1 / denom_sqr * (M4k[:, :3, :] @ pts_packed[:, None, :2])
        # Jk[:, :2, :2] = Jk[:, :2, :2] + \
        #     1 / denom * ops3d.eyes(2, num_pts,
        #                            device=denom.device, dtype=denom.dtype)

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
        raster_settings = kwargs.get(
            "raster_settings", self.rasterizer.raster_settings)
        cutoff_thres = raster_settings.cutoff_threshold

        # GV = M_k V_k^T M_k^T + V^h
        GVs, detMk = self._compute_variance_and_detMk(pointclouds, **kwargs)
        # TODO det dangerously low for float due to Vrk too small
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

    def transform_normals(self, point_clouds, **kwargs):
        """ Return normals in view coordinates (padded) """
        self.cameras = self.rasterizer.cameras = kwargs.get(
            "cameras", self.cameras)
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
            mask = (normals_view[:, :, 2] < 0)

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
        features_list = [features_padded[b][mask[b]]
                         for b in range(features_padded.shape[0])]
        new_point_clouds = point_clouds.__class__(
            points=points_list, normals=normals_list, features=features_list)

        # for k in per_point_info:
        #     per_point_info[k] = per_point_info[k][mask_packed]
        return new_point_clouds, mask_packed

    def _filter_points_with_invalid_depth(self, point_clouds, **kwargs):
        self.cameras = self.rasterizer.cameras = kwargs.get(
            'cameras', self.cameras)
        lengths = point_clouds.num_points_per_cloud()
        points_padded = point_clouds.points_padded()
        with torch.autograd.no_grad():
            to_view = self.cameras.get_world_to_view_transform()
            points = to_view.transform_points(points_padded)
            # z < znear or z < focallength
            if isinstance(self.cameras, (FoVOrthographicCameras, FoVPerspectiveCameras)):
                znear = self.cameras.znear
                zfar = self.cameras.zfar
                mask = points[..., 2] > znear.view(-1, 1)
                mask = mask & (points[..., 2] < zfar.view(-1, 1))
            elif isinstance(self.cameras, (OrthographicCameras, PerspectiveCameras)):
                znear = self.cameras.focal_length
                mask = points[..., 2] < znear.view(-1, 1)
            else:
                znear = torch.tensor(1.0).to(device=points.device)
                mask = points[..., 2] < znear.view(-1, 1)

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
        features_list = [features_padded[b][mask[b]]
                         for b in range(features_padded.shape[0])]
        new_point_clouds = point_clouds.__class__(
            points=points_list, normals=normals_list, features=features_list)
        # for k in per_point_info:
        #     per_point_info[k] = per_point_info[k][mask_packed]
        return new_point_clouds, mask_packed

    def filter_renderable(self, point_clouds, cameras, point_clouds_filter, **kwargs):
        if point_clouds.isempty():
            return None

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

        # render points whose depth is > focal_length or znear
        point_clouds, valid_depth_mask = self._filter_points_with_invalid_depth(
            point_clouds, **kwargs)

        # new point clouds containing only points facing towards the camera
        if self.backface_culling:
            point_clouds, frontface_mask = self._filter_backface_points(
                point_clouds, **kwargs)
        else:
            lengths = point_clouds.num_points_per_cloud()
            frontface_mask = torch.full((lengths.sum().item(),),
                                        True, dtype=torch.bool,
                                        device=point_clouds.device)

        return point_clouds, valid_depth_mask, frontface_mask

    def forward(self, point_clouds, point_clouds_filter=None, verbose=False, cutoff_thres_alpha=1.0,
                **kwargs) -> torch.Tensor:
        """
        point_clouds_filter: used to get activation mask and update visibility mask
        cutoff_threshold
        """

        if point_clouds.isempty():
            return None

        self.cameras = self.rasterizer.cameras = kwargs.get(
            'cameras', self.cameras)
        cameras = self.cameras
        self.backface_culling = kwargs.get(
            'backface_culling', self.backface_culling)

        P = point_clouds.num_points_per_cloud().sum().item()
        max_P = point_clouds.num_points_per_cloud().max().item()
        first_idx = point_clouds.cloud_to_packed_first_idx()
        batch_size = len(point_clouds)

        point_clouds, valid_depth_mask, frontface_mask = self.filter_renderable(
            point_clouds, cameras, point_clouds_filter)

        if point_clouds.isempty():
            return None

        # compute per-point features for elliptical gaussian weights
        with torch.autograd.no_grad():
            per_point_info = self._get_per_point_info(
                point_clouds, **kwargs)

        per_point_info['cutoff_threshold'] = cutoff_thres_alpha * \
            per_point_info['cutoff_threshold']

        # rasterize
        fragments = self.rasterizer(
            point_clouds, per_point_info, **kwargs)

        # compute weight: scalar*exp(-0.5Q)
        frag_scaler = gather_with_neg_idx(
            per_point_info['scaler'], 0, fragments.idx.view(-1).long())
        frag_scaler = frag_scaler.view_as(fragments.qvalue)
        weights = torch.exp(-0.5 * fragments.qvalue) * frag_scaler
        weights = weights.permute(0, 3, 1, 2)

        # from fragments to rgba
        pts_rgb = point_clouds.features_packed()[:, :3]

        if self.compositor is None:
            # NOTE: weight _splat_points_weights_backward, weighted sum will return
            # zero gradient for the weights.
            images = weighted_sum(fragments.idx.long().permute(0, 3, 1, 2),
                                  weights,
                                  pts_rgb.permute(1, 0),
                                  **kwargs)
        else:
            images = self.compositor(
                fragments.idx.long().permute(0, 3, 1, 2),
                weights,
                pts_rgb.permute(1, 0),
                **kwargs
            )
        images = images.clamp(0, 1)

        # permute so image comes at the end
        images = images.permute(0, 2, 3, 1)
        mask = fragments.occupancy

        images = torch.cat([images, mask.unsqueeze(-1)], dim=-1)

        with torch.autograd.no_grad():
            frontface_visibility_mask = get_per_point_visibility_mask(
                point_clouds, fragments)
            # put frontface_mask (num_frontfacing) to valid_depthmask (num_alldepth = num_active)
            frontface_mask[frontface_mask] = frontface_visibility_mask
            valid_depth_mask[valid_depth_mask] = frontface_mask

        if point_clouds_filter is not None:
            # update point_clouds visibility filter
            # we use this information in projection loss
            # put all_depth_visibility_mask (num_active) to original_visibility_mask (P,)
            # transform to padded
            original_visibility_mask = ops3d.packed_to_padded(
                valid_depth_mask.float(), first_idx, max_P).bool()
            point_clouds_filter.set_filter(
                visibility=original_visibility_mask)


        if verbose:
            # use scatter to get per point info of the original
            original_per_point_info = {}
            for k in per_point_info:
                original_per_point_info[k] = per_point_info[k].new_zeros(
                    valid_depth_mask.shape + per_point_info[k].shape[1:])
                original_per_point_info[k][valid_depth_mask] = per_point_info[k]

            return images, original_per_point_info, fragments
       return images


class SurfaceDiscRenderer(SurfaceSplattingRenderer):
    """
    Simplified version of SurfaceSplattingRenderer, which omits perspective projections.
    In other words, the splats are rendered as discs, the same way as Pytorch3D's defaul
    PointRendere, except that the disc size is still a function of the point cloud density.
    The backward is the same as SurfaceSplattingRenderer, except that there's no gradient
    for the ellipse parameters
    """
    pass
