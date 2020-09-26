from typing import Optional, Tuple, Union
import torch
import torch.autograd as autograd
from pytorch3d.ops import knn_points, knn_gather, convert_pointclouds_to_tensor
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.renderer.utils import convert_to_tensors_and_broadcast
from ..core.cloud import PointClouds3D
from .. import logger_py
from .common import approximate_gradient
from ..core.cloud import PointClouds3D
from ..utils import gather_batch_to_packed
from ..utils.mathHelper import eps_denom, pinverse
import time
import math

"""
Various functions for iso-surface projection
"""


def _convert_batched_to_packed_args(*args, latent=None):
    """
    Convert possible batched inputs to packed
    Args:
          list of broadcastable (N, *, cj) tensors and latent: (N, C)
    Returns:
          list of reshaped (M, cj) 2-dim inputs and latent (M, C)
    """
    if len(args) == 0:
        if latent is not None:
            latent.squeeze_()
        return args + (latent,)
    device = args[0].device
    args = convert_to_tensors_and_broadcast(*args, device=device)
    assert(all([x.shape[0] == args[0].shape[0] for x in args]))

    if latent is not None:
        latent.squeeze_()
        assert(latent.ndim == 2)
        if args[0].ndim > 2:
            batch_size = args[0].shape[0]
            num_intermediate = math.prod(args[0].shape[1:-1])
            args = [x.view(-1, x.shape[-1]) for x in args]

            first_idx = torch.repeat_interleave(
                torch.arange(batch_size, device=device),
                num_intermediate,
                dim=0)
            latent = gather_batch_to_packed(latent, first_idx)

    args = [x.view(-1, x.shape[-1]) for x in args]
    return args + [latent]


class LevelSetProjection(object):

    def __init__(self, proj_max_iters=10, proj_tolerance=5.0e-5, max_points_per_pass=120000,
                 exact_gradient=True, approx_grad_step=1e-2):
        self.proj_max_iters = proj_max_iters
        self.proj_tolerance = proj_tolerance
        self.max_points_per_pass = max_points_per_pass
        self.exact_gradient = exact_gradient
        self.approx_grad_step = approx_grad_step

    def project_points(self, points_init, network, latent, levelset):
        raise NotImplementedError


class UniformProjection(LevelSetProjection):
    """
    Project sampled points uniformly to the current zero-levelset
    First repulse points, then project to iso-surface `proj_max_iters` times.
    Repeat `sample_iters` times.
    Attributes:
        knn_k: used to find neighborhood
        sigma_p: variance weight spatial kernel (the larger the smoother)
        sigma_n: variance for normal kernel (the smaller the more feature-focused)
    """

    def __init__(self, proj_max_iters=10, proj_tolerance=5e-5,
                 max_points_per_pass=120000, exact_gradient=True,
                 approx_grad_step=1e-2,
                 sample_iters=5,
                 knn_k=12, r=2.5, alpha=1.0,
                 sigma_p=0.75, sigma_n=1.0, **kwargs):
        """
        Args:
            sigma_p, sigma_n: used to compound a feature x = [p/sigma_p, n/sigma_n]
        """
        super().__init__(proj_max_iters=proj_max_iters,
                         proj_tolerance=proj_tolerance,
                         max_points_per_pass=max_points_per_pass,
                         exact_gradient=exact_gradient,
                         approx_grad_step=approx_grad_step)
        self.knn_k = knn_k
        self.sample_iters = sample_iters
        self.sigma_p = sigma_p
        self.sigma_n = sigma_n
        self.alpha = alpha
        self.r = r

    def _create_tree(self, points_padded: torch.Tensor, refresh_tree=True, num_points_per_cloud=None):
        """
        create a data structure, per-point cache knn-neighbor
        Args:
            points_padded (N,P,D)
            num_points_per_cloud list
        """
        if not refresh_tree and hasattr(self, '_knn_idx') and self._knn_idx is not None:
            return self._knn_idx
        assert(points_padded.ndim == 3)
        if num_points_per_cloud is None:
            num_points_per_cloud = torch.tensor([points_padded.shape[1]] * points_padded.shape[0],
                                                device=points_padded.device, dtype=torch.long)
        knn_result = knn_points(
            points_padded, points_padded, num_points_per_cloud, num_points_per_cloud,
            K=self.knn_k + 1, return_nn=False, return_sorted=True)
        self._knn_idx = knn_result.idx[..., 1:]
        return self._knn_idx

    def _compute_sdf_and_grad(self, points, model, latent=None, **forward_kwargs) -> Tuple[torch.Tensor]:
        """
        Evalute sdf and compute grad in splits
        Args:
            points: (N, *, 3)
        Returns sdf (N, *) and grad (N,*,D)
        """
        shp = points.shape
        D = points.shape[-1]
        points_packed = points.view(-1, 3)
        if latent is not None:
            latent_packed = latent.view(
                points_packed.shape[0], latent.shape[-1])
        else:
            latent_packed = torch.empty(
                (points_packed.shape[0], 0), device=points_packed.device)
        grad_packed = []
        eval_packed = []
        for sub_points, sub_latent in zip(torch.split(points_packed, self.max_points_per_pass, dim=0),
                                          torch.split(latent_packed, self.max_points_per_pass, dim=0)):
            net_input = sub_points
            if self.exact_gradient:
                with autograd.enable_grad():
                    net_input.detach_().requires_grad_(True)
                    network_eval = model.forward(
                        net_input, c=sub_latent, **forward_kwargs)[..., :1]
                    input_grad = autograd.grad([network_eval], [net_input], torch.ones_like(
                        network_eval), retain_graph=True)[0][..., (-D):]
            else:
                network_eval = model.forward(net_input, c=sub_latent)[..., :1]
                input_grad = approximate_gradient(
                    sub_points, model, c=sub_latent, h=self.approx_grad_step)

            grad_packed.append(input_grad)
            eval_packed.append(network_eval)

        grad_packed = torch.cat(grad_packed, dim=0).view(shp)
        eval_packed = torch.cat(eval_packed, dim=0).view(shp[:-1])
        return eval_packed, grad_packed

    def project_points(self, point_clouds: Union[PointClouds3D, torch.Tensor],
                       model: nn.Module,
                       latent: Optional[torch.Tensor] = None,
                       normals_init: Optional[torch.Tensor] = None,
                       refresh_tree: bool = True,
                       **forward_kwargs):
        """
        TODO: change points_init to point_clouds
        repulse and project points, no early convergence because the repulsion term measures
        Args:
            point_clouds: (N,P,D) padded points
            model: nn.Module
            latent: (N,C) latent code in minibatches or None
        Args:
            levelset_points                  projected iso-points (N, P, D)
            levelset_points_Dx               gradient of the projected iso-points (N, P, D)
            network_eval_on_levelset_points  sdf value (N, P)
            mask                             iso-points is valid (sdf < threshold)
        """
        points_init, num_points = convert_pointclouds_to_tensor(point_clouds)
        shp = points_init.shape

        # from collections import defaultdict
        # times = defaultdict(lambda: 0.0)
        with autograd.no_grad():
            points_projected = points_init
            for sample_iter in range(self.sample_iters):
                # 1. compute the normals
                if normals_init is None:
                    # torch.cuda.synchronize()
                    t0 = time.time()
                    sdf_padded, grad_padded = self._compute_sdf_and_grad(
                        points_init, model, latent, **forward_kwargs)
                    # torch.cuda.synchronize()
                    # t1 = time.time()
                    # times['sdf_and_grad'] = (t1 - t0)
                    normals_init = torch.nn.functional.normalize(
                        grad_padded, dim=-1)
                else:
                    normals_init = torch.nn.functional.normalize(
                        normals_init, dim=-1)

                # 2. construct tree with compound features
                features = torch.cat(
                    [points_init / self.sigma_p, normals_init / self.sigma_n], dim=-1)

                # 3. move all the points
                # compute kernels (N,P,K) k(x,xi), xi \in Neighbor(x)
                knn_idx = self._create_tree(points_init, refresh_tree=(
                    sample_iter == 0) or refresh_tree, num_points_per_cloud=num_points)

                features_nb = knn_gather(features, knn_idx)
                # (N,P,K,D)
                features_diff = features.unsqueeze(2) - features_nb
                features_dist = torch.sum(features_diff**2, dim=-1)
                kernels = torch.exp(-features_dist)
                # CHECK: purpose is to remove outlier influence
                # N,P,K,1,D
                features_diff_ij = features_nb[:, :, :,
                                               None, :] - features_nb[:, :, None, :, :]
                features_dist_ij = torch.sum(features_diff_ij**2, dim=-1)
                kernels[features_dist > self.r] = 0
                kernel_matrices = torch.exp(-features_dist_ij)
                kernel_matrices[features_dist_ij > self.r] = 0
                kernel_matrices_inv = pinverse(kernel_matrices)
                # (N,P,K,3)
                points_cross_diff = features_diff[:, :, :, None, :3] + \
                    features_diff[:, :, None, :, :3]

                move = 1 / self.sigma_p / self.sigma_p * \
                    torch.sum(points_cross_diff * (kernels[:, :, None, :] *
                                                   kernels[:, :, :, None] *
                                                   kernel_matrices_inv).unsqueeze(-1), dim=[2, 3])
                move = F.normalize(move, dim=-1, eps=1e-15) * \
                    move.norm(dim=-1, keepdim=True).clamp_max(0.01)
                # move = torch.sum(
                #     features_diff[..., :3] * kernels.unsqueeze(-1), dim=-2)

                points_projected = move + points_init
                points_projected = points_projected.reshape(-1, 3)

                not_converged = points_projected.new_full(
                    points_projected.shape[:-1], True, dtype=torch.bool)

                for it in range(self.proj_max_iters):
                    # 4. project points to iso surface
                    # recompute normal grad and project
                    # torch.cuda.synchronize()
                    # t0 = time.time()
                    curr_points = points_projected[not_converged]
                    if latent is not None:
                        curr_latent = latent[not_converged]
                    else:
                        curr_latent = latent

                    curr_sdf, curr_grad = self._compute_sdf_and_grad(
                        curr_points, model, curr_latent, **forward_kwargs)

                    # torch.cuda.synchronize()
                    # t1 = time.time()
                    # times['sdf_and_grad'] = (
                    #     (t1 - t0) + it * times['sdf_and_grad']) / (it + 1)
                    curr_not_converged = curr_sdf.squeeze(
                        -1).abs() > self.proj_tolerance
                    not_converged[not_converged] = curr_not_converged
                    if (not_converged).any():
                        # Eq.4
                        active_grad = curr_grad[curr_not_converged]
                        active_sdf = curr_sdf[curr_not_converged]
                        active_pts = curr_points[curr_not_converged]
                        sum_square_grad = torch.sum(
                            active_grad ** 2, dim=-1, keepdim=True)
                        move = self.alpha * active_sdf.view(active_pts.shape[:-1] + (1,)) * \
                            (active_grad / eps_denom(sum_square_grad, 1.0e-17))
                        # move = F.normalize(move, dim=-1, eps=1e-15) * \
                        #     move.norm(dim=-1, keepdim=True).clamp_max(0.05)
                        points_projected[not_converged] = active_pts - move
                    else:
                        break

                if (sample_iter + 1) < self.sample_iters:
                    points_init = points_projected.view(shp)
                    normals_init = None  # force to recompute normals
                    # features = torch.cat(
                    #     [points_init / self.sigma_p, normals_init / self.sigma_n], dim=-1)

            # After the final projection, re-evaluate sdf and grad
            sdf_padded, grad_padded = self._compute_sdf_and_grad(
                points_projected, model, latent=None, **forward_kwargs)
            valid_projection = sdf_padded.abs() < self.proj_tolerance
            if (not valid_projection.all() and self.proj_max_iters > 0):
                abs_sdf = sdf_padded.abs()
                logger_py.info("[UnifProj] success rate: {:.2f}% ,  max : {:.3g} , min {:.3g} , mean {:.3g} , std {:.3g}"
                               .format(valid_projection.sum().item() * 100 / valid_projection.nelement(),
                                       torch.max(torch.abs(abs_sdf)),
                                       torch.min(torch.abs(abs_sdf)),
                                       torch.mean(abs_sdf),
                                       torch.std(abs_sdf)))
            # print(times)

            return {'levelset_points': points_projected.view(shp),
                    'levelset_points_Dx': grad_padded.view(shp),
                    'network_eval_on_levelset_points': sdf_padded.view(shp[:-1]),
                    'mask': valid_projection.view(shp[:-1])
                    }


class SphereTracing(LevelSetProjection):
    def __init__(self, proj_max_iters=10, proj_tolerance=5e-5,
                 max_points_per_pass=12000, exact_gradient=True,
                 alpha=1.0, radius=1.0, padding=0.1, **kwargs):
        """
        Args:
            sigma_p, sigma_n: used to compound a feature x = [p/sigma_p, n/sigma_n]
        """
        super().__init__(proj_max_iters=proj_max_iters,
                         proj_tolerance=proj_tolerance,
                         max_points_per_pass=max_points_per_pass,
                         exact_gradient=exact_gradient,
                         )
        self.alpha = alpha
        self.radius = radius
        self.padding = padding

    def project_points(self, ray0: torch.Tensor, ray_direction: torch.Tensor,
                       model: nn.Module,
                       latent: Optional[torch.Tensor] = None,
                       **forward_kwargs):
        """
        Args:
            ray0: (N,*,D) points
            ray_direction: (N,*,D) normalized ray direction
            model: nn.Module
            latent: (N, C) latent code or None

        Returns:
            {
            'levelset_points':      projected levelset points
            'levelset_points_Dx':   moore-penrose pseudo inverse (will be used in the sampling layer again)
            'network_eval_on_levelset_points':  network prediction of the projected levelset points
            'mask':                 mask of valid projection (SDF value< threshold) (N,P)
            }
        """
        shp = ray0.shape
        # to packed form
        ray0, ray_direction, latent = _convert_batched_to_packed_args(
            ray0, ray_direction, latent=latent)

        N, D = ray0.shape

        ray0_list = torch.split(ray0, self.max_points_per_pass, dim=0)
        ray_direction_list = torch.split(
            ray_direction, self.max_points_per_pass, dim=0)

        # change c from batch to per point
        if latent is not None and latent.nelement() > 0:
            latent_list = torch.split(latent, self.max_points_per_pass, dim=0)
        else:
            latent_list = [None] * len(ray0_list)

        levelset_points = []
        levelset_points_Dx = []
        eval_on_levelset_points = []

        with autograd.no_grad():
            # process points in sub-batches
            for sub_points_init, sub_ray_direction, sub_latent in zip(
                    ray0_list, ray_direction_list, latent_list):
                curr_projection = sub_points_init.clone()
                num_points, points_dim = sub_points_init.shape[:2]
                active_mask = sub_points_init.new_full(
                    [num_points, ], True, dtype=torch.bool)

                trials = 0
                model.eval()

                # mask the points that are still inside a unit sphere *after*
                # ray-marching
                inside_sphere = torch.full(
                    sub_ray_direction.shape[:-1], True, dtype=torch.bool,
                    device=sub_ray_direction.device)
                while True:
                    # evaluate sdf
                    net_input = curr_projection[active_mask]
                    if sub_latent is not None:
                        current_latent = sub_latent[active_mask]
                    else:
                        current_latent = sub_latent
                    with autograd.enable_grad():
                        if trials == 0:
                            if self.exact_gradient:
                                net_input.detach_().requires_grad_(True)
                                network_eval = model.forward(
                                    net_input, c=current_latent, **forward_kwargs)[..., :1]
                                grad = autograd.grad(
                                    network_eval, net_input, torch.ones_like(network_eval), retain_graph=True
                                )[0].detach()
                            else:
                                network_eval = model.forward(
                                    net_input, c=current_latent, **forward_kwargs)[..., :1]
                                grad = approximate_gradient(
                                    net_input, model, c=current_latent, h=self.approx_grad_step, **forward_kwargs)

                            network_eval = network_eval.detach()  # c in Eq.9
                        else:
                            if self.exact_gradient:
                                net_input.detach_().requires_grad_(True)
                                network_eval_active = model.forward(
                                    net_input, c=current_latent, **forward_kwargs)[..., :1]
                                grad_active = autograd.grad([network_eval_active], [net_input], torch.ones_like(
                                    network_eval_active), retain_graph=True)[0]
                            else:
                                network_eval_active = model.forward(
                                    net_input, c=current_latent, **forward_kwargs)[..., :1]
                                grad = approximate_gradient(
                                    net_input, model, c=current_latent, h=self.approx_grad_step, **forward_kwargs)

                            grad[active_mask] = grad_active.detach()
                            network_eval[active_mask] = network_eval_active.detach()

                    active_mask = (network_eval.abs() > self.proj_tolerance).squeeze(1) \
                        & inside_sphere

                    # project not converged points to iso-surface
                    if ((active_mask).any() and trials < self.proj_max_iters):
                        network_eval_active = network_eval[active_mask]
                        points_active = curr_projection[active_mask]

                        # Advance by alpha*sdf
                        move = self.alpha * network_eval_active * \
                            sub_ray_direction[active_mask]
                        move = F.normalize(move, dim=-1, eps=1e-15) * \
                            move.norm(dim=-1, keepdim=True).clamp_max(0.01)
                        points_active += move
                        inside_sphere[active_mask] = (points_active.norm(
                            dim=-1) < (self.padding + self.radius))
                        curr_projection[active_mask &
                                        inside_sphere] = points_active[inside_sphere[active_mask]]
                    else:
                        break

                    trials = trials + 1

                curr_projection.detach_()
                levelset_points.append(curr_projection)
                eval_on_levelset_points.append(network_eval.detach())
                levelset_points_Dx.append(grad.detach())

        levelset_points = torch.cat(levelset_points, dim=0)
        eval_on_levelset_points = torch.cat(eval_on_levelset_points, dim=0)
        levelset_points_Dx = torch.cat(levelset_points_Dx, dim=0)

        valid_projection = eval_on_levelset_points.abs() <= self.proj_tolerance

        # if (not valid_projection.all() and self.proj_max_iters > 0):
        #     abs_sdf = eval_on_levelset_points.abs()
        #     logger_py.info("[SphereTracing] success rate: {:.2f}% ,  max : {:.3g} , min {:.3g} , mean {:.3g} , std {:.3g}"
        #                    .format(valid_projection.sum().item() * 100 / valid_projection.nelement(),
        #                            torch.max(torch.abs(abs_sdf)),
        #                            torch.min(torch.abs(abs_sdf)),
        #                            torch.mean(abs_sdf),
        #                            torch.std(abs_sdf)))

        return {'levelset_points': levelset_points.view(shp),
                'network_eval_on_levelset_points': eval_on_levelset_points.view(shp[:-1]),
                'levelset_points_Dx': levelset_points.view(shp),
                'mask': valid_projection.view(shp[:-1])}


class GenNewtonProjection(LevelSetProjection):
    """ General Newton Projection:
    Project an initial sample point to the current zero-levelset.
    """

    def project_points(self, point_clouds: Union[torch.Tensor, PointClouds3D],
                       model: nn.Module,
                       latent: Optional[torch.Tensor] = None):
        """
        Args:
            point_clouds: (N,*,D) padded points or PointClouds
            model: nn.Module
            latent: (N,*,C) gathered padded latent code or None

        Returns:
            {
                'levelset_points':      projected levelset points
                'levelset_points_Dx':   moore-penrose pseudo inverse (will be used in the sampling layer again)
                'network_eval_on_levelset_points':  network prediction of the projected levelset points
            }
        """
        points_init, num_points_per_cloud = convert_pointclouds_to_tensor(
            point_clouds)
        shp = points_init.shape
        points_init, latent = _convert_batched_to_packed_args(
            points_init, latent=latent)
        # project points to the iso-surface
        # change c from batch to per point
        if latent is not None and latent.nelement() > 0:
            first_idx = torch.repeat_interleave(
                torch.arange(points_init.shape[0], device=self.device),
                points_init.shape[1],
                dim=0)
            latent = gather_batch_to_packed(latent, first_idx)

        levelset_points = []
        levelset_points_Dx = []
        eval_on_levelset_points = []

        with autograd.no_grad():
            for sub_points_init in torch.split(points_init, self.max_points_per_pass, dim=0):
                curr_projection = sub_points_init.clone()
                num_points, points_dim = sub_points_init.shape[:2]
                not_converged = sub_points_init.new_full(
                    [num_points, ], True, dtype=torch.bool)

                trials = 0
                model.eval()
                while True:
                    # Do
                    # {repulse, evaluate sdf, project not converged points}
                    # while
                    # {not all the projected points are converged}
                    net_input = curr_projection[not_converged]
                    with autograd.enable_grad():
                        if latent is not None:
                            curr_latent = latent[not_converged]
                        else:
                            curr_latent = latent

                        if trials == 0:
                            if self.exact_gradient:
                                net_input.detach_().requires_grad_(True)
                                network_eval = model.forward(
                                    net_input, c=curr_latent)[..., :1]
                                grad = autograd.grad([network_eval], [net_input], torch.ones_like(
                                    network_eval), retain_graph=True)[0][:, (-points_dim):]
                            else:
                                network_eval = model.forward(
                                    net_input, c=current_latent)[..., :1]
                                grad = approximate_gradient(
                                    net_input, model, c=current_latent, h=self.approx_grad_step)

                            network_eval = network_eval.detach()  # c in Eq.9
                        else:
                            if self.exact_gradient:
                                net_input.detach_().requires_grad_(True)
                                network_eval_active = model.forward(
                                    net_input, c=curr_latent)[..., :1]
                                grad_active = autograd.grad([network_eval_active], [net_input], torch.ones_like(
                                    network_eval_active), retain_graph=True)[0][:, (-points_dim):]
                            else:
                                network_eval_active = model.forward(
                                    net_input, c=current_latent)[..., :1]
                                grad_active = approximate_gradient(
                                    net_input, model, c=current_latent, h=self.approx_grad_step)

                            grad[not_converged] = grad_active.detach()
                            # c in Eq.9
                            network_eval[not_converged] = network_eval_active.detach(
                            )

                    not_converged = network_eval.squeeze(
                        1).abs() > self.proj_tolerance

                    # project not converged points to iso-surface
                    if (not_converged.any() and trials < self.proj_max_iters):
                        num_not_converged = not_converged.sum()
                        network_eval_active = network_eval[not_converged]
                        grad_active = grad[not_converged]
                        points_active = curr_projection[not_converged]

                        # demonimator in Eq.5
                        sum_square_grad = torch.sum(
                            grad_active ** 2, dim=-1, keepdim=True)
                        # Eq.4
                        points_active = points_active - network_eval_active.view([num_not_converged] + [1] * (
                            len(grad_active.shape) - 1)) * (grad_active / sum_square_grad.clamp_min(1.0e-8))

                        curr_projection[not_converged] = points_active
                    else:
                        break

                    trials = trials + 1

                curr_projection.detach_()
                levelset_points.append(curr_projection)
                levelset_points_Dx.append(grad.detach())
                eval_on_levelset_points.append(network_eval.detach())

        levelset_points = torch.cat(levelset_points, dim=0)
        levelset_points_Dx = torch.cat(levelset_points_Dx, dim=0)
        eval_on_levelset_points = torch.cat(eval_on_levelset_points, dim=0)

        valid_projection = eval_on_levelset_points.abs() <= self.proj_tolerance
        if (not valid_projection.all() and self.proj_max_iters > 0):
            abs_sdf = eval_on_levelset_points.abs()
            logger_py.info("[NewtonProj] success rate: {:.2f}% ,  max : {:.3g} , min {:.3g} , mean {:.3g} , std {:.3g}"
                           .format(valid_projection.sum().item() * 100 / valid_projection.nelement(),
                                   torch.max(torch.abs(abs_sdf)),
                                   torch.min(torch.abs(abs_sdf)),
                                   torch.mean(abs_sdf),
                                   torch.std(abs_sdf)))

        return {'levelset_points': levelset_points.view(shp),
                'levelset_points_Dx': levelset_points_Dx.view(shp),
                'network_eval_on_levelset_points': eval_on_levelset_points.view(shp[:-1]),
                'mask': valid_projection.view(shp[:-1])}


class SampleNetwork(nn.Module):
    """
    Eq.13 in the paper
    """

    def forward(self, network: nn.Module,
                levelset_points: torch.Tensor,
                levelset_points_Dx: torch.Tensor,
                network_eval_on_levelset_points: torch.Tensor,
                c: Optional[torch.Tensor] = None,
                return_eval: bool = False):
        """
        Args:
            levelset_points: (n, *, d) leaf nodes on the level set (from projection), packed points
            levelset_points_Dx: (n, *, d) grad(network, levelset_points)
            network_eval_on_levelset_points: (n, *, 1) the SDF value of the levelset points
        """
        # is it necessary to pass in network_eval_on_levelset_points?
        network_eval = network.forward(levelset_points).view_as(
            network_eval_on_levelset_points)
        sum_square_grad = torch.sum(
            levelset_points_Dx ** 2, dim=-1, keepdim=True)

        # network_eval_on_levelset_points (bxnxl)   := c, independent of theta (constant)
        # network_eval                    (bxnxl)   := F(p; theta)
        # levelset_points_Dx              (bxlxnxd) := D_xF(p;theta_0)^{+} moore-penrose pseudo-inverse Eq.5
        sampled_points = levelset_points - (
            network_eval - network_eval_on_levelset_points).view(levelset_points.shape[:-1] + (1,)) * (
            levelset_points_Dx / sum_square_grad.clamp_min(1.0e-8))
        if return_eval:
            return sampled_points, network_eval
        return sampled_points


def find_zero_crossing_between_point_pairs(p0: Optional[torch.Tensor], p1: Optional[torch.Tensor],
                                           network: torch.nn.Module,
                                           n_secant_steps=8,
                                           n_steps=100,
                                           is_occupancy=True,
                                           max_points=120000,
                                           c: Optional[torch.Tensor] = None,
                                           **forward_kwargs):
    '''
    Args:
        p0 (tensor): (N, *, 3)
        p1 (tensor): (N, *, 3)
        network (nn.Module): sdf evaluator
        n_steps (int): number of evaluation steps; if the difference between
            n_steps[0] and n_steps[1] is larger then 1, the value is sampled
            in the range
        n_secant_steps (int): number of secant refinement steps
        max_points (int): max number of points loaded to GPU memory
        c (tensor): (N,C)
    Returns:
        pt_pred (tensor): (N, *, 3)
        mask (tensor): (N, *) boolean tensor mask valid zero crossing (sign change
            & from outside to inside & doesn't start from being inside )
    '''
    def _compare_func(is_occupancy, tau_logit=0.0):
        def less_than(data):
            return data < tau_logit

        def greater_than(data):
            return data > tau_logit
        if is_occupancy:
            return less_than
        else:
            return greater_than

    compare_func = _compare_func(is_occupancy)
    device = p0.device
    shp = p0.shape
    p0, p1, c = _convert_batched_to_packed_args(p0, p1, latent=c)
    n_pts, D = p0.shape

    # Prepare d_proposal and p_proposal in form (b_size, n_pts, n_steps, 3)
    # d_proposal are "proposal" depth values and p_proposal the
    # corresponding "proposal" 3D points
    ray_direction = torch.nn.functional.normalize(
        p1 - p0, p=2, dim=-1, eps=1e-10)

    d_proposal = torch.linspace(0, 1, steps=n_steps).view(
        1, n_steps).to(device) * torch.norm(p1 - p0, p=2, dim=-1).unsqueeze(-1)

    p_proposal = p0.unsqueeze(-2) + \
        ray_direction.unsqueeze(-2) * d_proposal.unsqueeze(-1)

    # Evaluate all proposal points in parallel
    with torch.no_grad():
        p_proposal = p_proposal.view(-1, 3)
        p_proposal_list = torch.split(p_proposal, max_points, dim=0)
        if c is not None:
            c = c.view(n_pts, 1, c.shape[-1]).expand(n_pts,
                                                     p_proposal.shape[1], c.shape[-1]).view(-1, c.shape[-1])
            c_list = torch.split(c, max_points, dim=0)
        else:
            c_list = [None]*len(p_proposal_list)

        val = torch.cat([network.forward(p_split, c=c_split, **forward_kwargs)[..., :1]
                         for p_split, c_split in zip(p_proposal_list, c_list)],
                        dim=0).view(n_pts, n_steps)

    # Create mask for valid points where the first point is not occupied
    mask_0_not_occupied = compare_func(val[..., 0])

    # Calculate if sign change occurred and concat 1 (no sign change) in
    # last dimension
    sign_matrix = torch.cat([torch.sign(val[..., :-1] * val[..., 1:]),
                             torch.ones(n_pts, 1).to(device)],
                            dim=-1)
    cost_matrix = sign_matrix * torch.arange(n_steps, 0, -1).float().to(device)

    # Get first sign change and mask for values where
    # a.) a sign changed occurred and
    # b.) no a neg to pos sign change occurred (meaning from inside surface to outside)
    # NOTE: for sdf value b.) becomes from pos to neg
    values, indices = torch.min(cost_matrix, -1)
    mask_sign_change = values < 0
    mask_out_to_in = compare_func(val[torch.arange(n_pts), indices])

    # Define mask where a valid depth value is found
    mask = mask_sign_change & mask_out_to_in

    # Get depth values and function values for the interval
    # to which we want to apply the Secant method
    # Again, for SDF decoder d_low is actually d_high
    d_start = d_proposal[torch.arange(n_pts), indices.view(n_pts)][mask]
    f_start = val[torch.arange(n_pts), indices.view(n_pts)][mask]
    indices = torch.clamp(indices + 1, max=n_steps - 1)
    d_end = d_proposal[torch.arange(n_pts), indices.view(n_pts)][mask]
    f_end = val[torch.arange(n_pts), indices.view(n_pts)][mask]

    p0_masked = p0[mask]
    ray_direction_masked = ray_direction[mask]

    # write c in pointwise format
    if c is not None and c.shape[-1] != 0:
        c = c.unsqueeze(1).repeat(1, n_pts, 1).view(-1, c.shape[-1])[mask]

    # Apply surface depth refinement step (e.g. Secant method)
    p_pred = run_Secant_method(
        f_start, f_end, d_start, d_end, n_secant_steps, p0_masked,
        ray_direction_masked, network, c, compare_func, **forward_kwargs)

    # for sanity
    pt_pred = torch.ones(mask.shape + (3,)).to(device)
    pt_pred[mask] = p_pred
    pt_pred = pt_pred.view(shp)
    return pt_pred, mask.view(shp[:-1])


def run_Secant_method(f_start, f_end, d_start, d_end, n_secant_steps,
                      p0_masked, ray_direction_masked, decoder, c,
                      compare_func, **forward_kwargs):
    ''' Runs the secant method for interval [d_start, d_end].

    Args:
        f_start(tensor): (N, *)
        f_end(tensor): (N, *)
        d_start (tensor): (N,*) start values for the interval
        d_end (tensor): (N,*) end values for the interval
        n_secant_steps (int): number of steps
        p0_masked (tensor): masked ray start points
        ray_direction_masked (tensor): masked ray direction vectors
        decoder (nn.Module): decoder model to evaluate point occupancies
        c (tensor): latent conditioned code c
    '''
    d_pred = - f_start * (d_end - d_start) / (f_end - f_start) + d_start
    for i in range(n_secant_steps):
        p_mid = p0_masked + d_pred.unsqueeze(-1) * ray_direction_masked
        with torch.no_grad():
            f_mid = decoder.forward(p_mid, c, **forward_kwargs)[..., :1]
            f_mid = f_mid.squeeze(-1)
        # ind_start masks f_mid has the same sign as d_start
        # if decoder outputs sdf, d_start (start) is > 0,
        ind_start = compare_func(f_mid)
        if ind_start.sum() > 0:
            d_start[ind_start] = d_pred[ind_start]
            f_start[ind_start] = f_mid[ind_start]
        if (ind_start == 0).sum() > 0:
            d_end[ind_start == 0] = d_pred[ind_start == 0]
            f_end[ind_start == 0] = f_mid[ind_start == 0]

        d_pred = - f_start * (d_end - d_start) / (f_end - f_start) + d_start

    p_pred = p0_masked + \
        d_pred.unsqueeze(-1) * ray_direction_masked
    return p_pred


class DirectionalSamplingNetwork(SampleNetwork):
    def forward(self, network, iso_points, iso_points_Dx, ray, c=None, return_eval: bool = False):
        network_eval = network.forward(iso_points, c=c)[..., :1]

        sampled_points = iso_points - ray / \
            eps_denom(torch.sum(iso_points_Dx * ray, dim=-1, keepdim=True), 1e-17) *\
            network_eval

        if return_eval:
            return sampled_points, network_eval
        return sampled_points
