"""
Surface Splatting Rasterizer

"""
from typing import NamedTuple, Optional

import torch
import torch.autograd as autograd
import numpy as np

import pytorch3d.ops as ops3d
from pytorch3d.renderer import (PointsRasterizationSettings,
                                PointsRasterizer,
                                FoVPerspectiveCameras,
                                )
from pytorch3d.renderer.cameras import get_world_to_view_transform
from pytorch3d.renderer.points.rasterize_points import kMaxPointsPerBin

from .cloud import PointClouds3D
from ..utils.mathHelper import eps_denom, eps_sqrt
from .. import _C, get_debugging_mode, get_debugging_tensor

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
    occupancy: torch.Tensor


class PointsRasterizationSettings:
    """
    Attributes:
    TODO: complete documentations
        cutoff_threshold (float): deciding whether inside the splat based on Q < cutoff_threshold
        radii_backward_scaler (float): the scaler used to increase the splat radius
            for backward pass, used for OccRBFBackward and OccBackward. If zero,
            then use WeightBackward, which contains gradients only for points
            rendered at the pixel.
        backward_rbf: use radial based gradients from exp(-Q) for x and y, otherwise use 1/d^2*dx
        Vrk_isotropic: use isotropic gaussian in the source space, otherwise compute an per-point
            gaussian variance (TODO)
    """
    __slots__ = [
        "cutoff_threshold",
        "depth_merging_threshold",
        "Vrk_invariant",
        "Vrk_isotropic",
        "radii_backward_scaler",
        "image_size",
        "points_per_pixel",
        "bin_size",
        "max_points_per_bin",
        "backward_rbf",
        "clip_pts_grad",
    ]

    def __init__(
        self,
        cutoff_threshold: float = 1,
        depth_merging_threshold: float = 0.05,
        Vrk_invariant: bool = False,
        Vrk_isotropic: bool = True,
        radii_backward_scaler: float = 10,
        image_size: int = 256,
        points_per_pixel: int = 8,
        bin_size: Optional[int] = None,
        max_points_per_bin: Optional[int] = None,
        backward_rbf: bool = False,
        clip_pts_grad: Optional[float] = -1
    ):
        self.cutoff_threshold = cutoff_threshold
        self.depth_merging_threshold = depth_merging_threshold
        self.Vrk_invariant = Vrk_invariant
        self.Vrk_isotropic = Vrk_isotropic
        self.radii_backward_scaler = radii_backward_scaler
        self.image_size = image_size
        self.points_per_pixel = points_per_pixel
        self.bin_size = bin_size
        self.max_points_per_bin = max_points_per_bin
        self.backward_rbf = backward_rbf
        self.clip_pts_grad = clip_pts_grad


class SurfaceSplatting(PointsRasterizer):
    """
    SurfaceSplatting rasterizer Differentiable Surface Splatting
    Outputs per point:
        1. the screen space extent (screen range (-1, 1))
        2. center of projection (screen range (-1, 1))
    """

    def __init__(self, device="cpu", cameras=None, compositor=None, raster_settings=None):
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
        self.cameras = (
            cameras if cameras is not None else FoVPerspectiveCameras(
                device=device)
        )
        super().__init__(cameras=self.cameras)
        if raster_settings is None:
            raster_settings = PointsRasterizationSettings()

        self.raster_settings = raster_settings

    def forward(self, point_clouds, per_point_info, **kwargs) -> PointFragments:
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

        pcls_screen = self.transform(point_clouds, **kwargs)

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
            backward_rbf=raster_settings.backward_rbf,
            clip_pts_grad=raster_settings.clip_pts_grad
        )
        return PointFragments(idx=idx, zbuf=zbuf, qvalue=qvalue_map, occupancy=occ_map)


def _clip_grad(value=0.1):
    def func(grad):
        scaler = grad.norm(dim=-1, keepdim=True).clamp(0, value)
        grad = torch.nn.functional.normalize(grad, dim=-1) * scaler
        # grad.clamp_(-value, value)
        return grad
    return func


def rasterize_elliptical_points(pcls_screen, ellipse_params,
                                cutoff_threshold, radii,
                                depth_merging_threshold: float = 0.05,
                                image_size: int = 512,
                                points_per_pixel: int = 5,
                                bin_size: Optional[int] = None,
                                max_points_per_bin: Optional[int] = None,
                                radii_backward_scaler: float = 10.0,
                                backward_rbf: bool = False,
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
        if not points_packed.is_cuda:
            # Binned CPU rasterization not fully implemented
            bin_size = 0
        else:
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
                "bin_size too small, number of points per bin must be less than %d; got %d"
                % (kMaxPointsPerBin, num_bins)
            )

    if max_points_per_bin is None:
        max_points_per_bin = int(max(10000, num_points_per_cloud.max()))

    if points_packed.requires_grad and clip_pts_grad > 0:
        points_packed.register_hook(_clip_grad(clip_pts_grad))

    idx, zbuf, qvalue_map, occ_map = EllipticalRasterizer.apply(
        points_packed, ellipse_params, cutoff_threshold, radii,
        cloud_to_packed_first_idx, num_points_per_cloud,
        depth_merging_threshold, image_size, points_per_pixel, bin_size, max_points_per_bin,
        radii_backward_scaler, backward_rbf)
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
                backward_rbf: bool = False):
        """
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
        ctx.backward_rbf = backward_rbf
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

        grads = (grad_radii, grad_cloud_to_packed_first_idx,
                 grad_num_points_per_cloud,
                 grad_depth_merging_thres, grad_image_size, grad_points_per_pixel,
                 grad_bin_size, grad_max_points_per_bin, grad_radii_s, grad_backward_rbf)

        saved_tensors = ctx.saved_tensors
        radii_s = ctx.radii_backward_scaler

        if radii_s == 0:
            pts_screen, idx, ellipse_params, cutoff_threshold = ctx.saved_tensors
            grads_input = _C._splat_points_weights_backward(
                pts_screen, ellipse_params, cutoff_threshold, idx, zbuf_grad, qvalue_grad)

        else:
            # either use OccRBFBackward or use OccBackward
            pts_screen, ellipse_param, cutoff_threshold, radii, idx, zbuf0, \
                cloud_to_packed_first_idx, num_points_per_cloud, \
                = ctx.saved_tensors
            depth_merging_threshold = ctx.depth_merging_threshold
            if ctx.backward_rbf:
                grads_input = _C._splat_points_occ_rbf_backward(pts_screen, ellipse_param, cutoff_threshold, radii,
                                                                idx, zbuf0, occ_grad, zbuf_grad,
                                                                cloud_to_packed_first_idx, num_points_per_cloud,
                                                                radii_s, depth_merging_threshold)
            else:
                # print("use occ backward")
                grads_input = _C._splat_points_occ_backward(pts_screen, ellipse_param, cutoff_threshold, radii,
                                                            idx, zbuf0, occ_grad, zbuf_grad,
                                                            cloud_to_packed_first_idx, num_points_per_cloud,
                                                            radii_s, depth_merging_threshold)
        pts_grad, grad_cutoff_thres = grads_input
        grad_cutoff_thres = grad_cutoff_thres / grad_cutoff_thres.nelement()

        # _dbg_tensor['raster_pts_grad'] = pts_grad.detach()
        # _dbg_tensor['raster_ellipse_'] = ellipse_params_grad.detach()
        return (pts_grad, None, grad_cutoff_thres) + grads


# class DiscRasterizer(autograd.Function):
#     @staticmethod
#     def forward(ctx, pts_screen, radii, cutoff_threshold,
#                 cloud_to_packed_first_idx, num_points_per_cloud,
#                 depth_merging_threshold,
#                 image_size,
#                 points_per_pixel,
#                 bin_size: int = 0,
#                 max_points_per_bin: int = 0,
#                 radii_backward_scaler: float = 10.0):
#         """
#         """
#         args = (
#             pts_screen,
#             radii,
#             cloud_to_packed_first_idx,
#             num_points_per_cloud,
#             cutoff_threshold,
#             depth_merging_threshold,
#             image_size,
#             points_per_pixel,
#             bin_size,
#             max_points_per_bin,
#         )
#         idx, zbuf, qvalue_map, occ_map = _C.splat_points(*args)

#         ctx.cutoff_threshold = cutoff_threshold
#         ctx.radii_backward_scaler = radii_backward_scaler
#         ctx.depth_merging_threshold = depth_merging_threshold
#         if radii_backward_scaler == 0:
#             ctx.save_for_backward(pts_screen, idx)
#         else:
#             zbuf0 = zbuf[..., 0].clone()
#             ctx.save_for_backward(pts_screen, radii, idx, zbuf0,
#                                   cloud_to_packed_first_idx, num_points_per_cloud)
#         return idx, zbuf, qvalue_map, occ_map

#     @staticmethod
#     def backward(ctx, idx_grad, zbuf_grad, qvalue_grad, occ_grad):
#         # idx_grad and zbuf_grad are None (unless maybe we make weights depend on z? i.e. volumetric splatting)

#         # NOTE(yifan): should we pass gradients to ellipse_params and radii? I think it's not a good idea -
#         # would cause optimization difficulty
#         grad_radii = None
#         grad_cloud_to_packed_first_idx = None
#         grad_num_points_per_cloud = None
#         grad_cutoff_thres = None
#         grad_depth_merging_thres = None
#         grad_image_size = None
#         grad_points_per_pixel = None
#         grad_bin_size = None
#         grad_max_points_per_bin = None
#         grad_radii_s = None
#         grad_backward_rbf = None

#         grads = (grad_radii, grad_cloud_to_packed_first_idx,
#                  grad_num_points_per_cloud, grad_cutoff_thres,
#                  grad_depth_merging_thres, grad_image_size, grad_points_per_pixel,
#                  grad_bin_size, grad_max_points_per_bin, grad_radii_s, grad_backward_rbf)

#         saved_tensors = ctx.saved_tensors
#         radii_s = ctx.radii_backward_scaler

#         # use OccBackward
#         pts_screen, radii, idx, zbuf0, \
#             cloud_to_packed_first_idx, num_points_per_cloud, \
#             = ctx.saved_tensors
#         depth_merging_threshold = ctx.depth_merging_threshold
#         cutoff_threshold = ctx.cutoff_threshold

#         pts_grad = _C._splat_disc_points_occ_backward(pts_screen, radii,
#                                                       idx, zbuf0, occ_grad, zbuf_grad,
#                                                       cloud_to_packed_first_idx, num_points_per_cloud,
#                                                       radii_s, cutoff_threshold, depth_merging_threshold)
#         # _dbg_tensor['raster_pts_grad'] = pts_grad.detach()
#         # _dbg_tensor['raster_ellipse_'] = ellipse_params_grad.detach()
#         return (pts_grad) + grads


__all__ = [k for k in globals().keys() if not k.startswith("_")]
