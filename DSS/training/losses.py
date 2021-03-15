"""
[1] Feature Preserving Point Set Surfaces based on Non-Linear Kernel Regression
Cengiz Oztireli, Gaël Guennebaud, Markus Gross
[2] Consolidation of Unorganized Point Clouds for Surface Reconstruction
Hui Huang, Dan Li, Hao Zhang, Uri Ascher Daniel Cohen-Or
[3] Differentiable Surface Splatting for Point-based Geometry Processing
Wang Yifan, Felice Serena, Shihao Wu, Cengiz Oeztireli, Olga Sorkine-Hornung
"""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d import ops
from pytorch3d.ops import padded_to_packed
from pytorch3d.ops.knn import _KNN as KNN
from DSS.core.cloud import PointClouds3D
from DSS.utils.mathHelper import eps_denom, estimate_pointcloud_normals
from DSS import get_debugging_mode, get_debugging_tensor
from DSS import get_logger

logger_py = get_logger(__name__)


class BaseLoss(nn.Module):
    """
    Attributes:
        reduce (str): 'mean' | 'sum' | 'none'
        channel_dim (int): if not None, average this dimension before
            reduction
    """

    def __init__(self, reduction: str = 'mean', channel_dim: int = -1):
        super().__init__()
        self.reduction = reduction
        self.channel_dim = channel_dim
        self.hooks = []

    def compute(self, *args):
        raise NotImplementedError

    def _reduce(self, loss, reduction=None):
        reduction = reduction or self.reduction
        if reduction == 'none':
            return loss
        if reduction == 'sum':
            loss = torch.sum(loss)
        elif reduction == 'mean':
            loss = torch.mean(loss)
        else:
            raise ValueError(
                'Invalid reduction method ({})'.format(self.reduction))
        return loss

    def forward(self, *args, **kwargs):
        reduction = kwargs.pop('reduction', self.reduction)
        self.channel_dim = kwargs.pop('channel_dim', self.channel_dim)
        loss = self.compute(*args, **kwargs)
        if self.channel_dim is not None:
            loss = torch.sum(loss, dim=self.channel_dim)
        loss = self._reduce(loss, reduction=reduction)
        return loss

    def debug(self, is_debug, **kwargs):
        if is_debug:
            # nothing to do
            pass
        else:
            for hook in self.hooks:
                hook.remove()
            self.hooks.clear()


class NormalLengthLoss(BaseLoss):
    """enforce normal length to be 1"""

    def __init__(self, reduction: str = 'mean', **kwargs):
        super().__init__(reduction=reduction, channel_dim=None)

    def compute(self, normals, *args, **kwargs):
        assert(normals.shape[-1] == 3)
        loss = (normals.norm(p=2, dim=-1) - 1)**2
        return loss


class NormalLoss(BaseLoss):
    """ Normal same in the neighborhood """
    def __init__(self, reduction: str = 'mean', neighborhood_size=16):
        super().__init__(reduction=reduction, channel_dim=None)
        self.neighborhood_size = neighborhood_size

    def compute(self, pointclouds: PointClouds3D, *args, neighborhood_size=None, **kwargs):
        if neighborhood_size is None:
            neighborhood_size = self.neighborhood_size
        num_points = pointclouds.num_points_per_cloud()
        normals_packed = pointclouds.normals_packed()
        assert(normals_packed is not None)
        normals_ref = estimate_pointcloud_normals(pointclouds, neighborhood_size=neighborhood_size)
        normals_ref = padded_to_packed(
            normals_ref, pointclouds.cloud_to_packed_first_idx(), num_points.sum().item())

        return 1-F.cosine_similarity(normals_packed, normals_ref).abs()


class CosSimilarityLoss(BaseLoss):

    def __init__(self, reduction: str = 'mean'):
        super().__init__(reduction=reduction, channel_dim=None)

    def compute(self, vec1, vec2, **kwargs):
        """
        F.cosine_similarity
        """
        return 1 - F.cosine_similarity(vec1, vec2)


class SmapeLoss(BaseLoss):
    """
    relative L1 norm
    http://drz.disneyresearch.com/~jnovak/publications/KPAL/KPAL.pdf eq(2)
    """

    def compute(self, x, y, mask=None, eps=1e-8, **kwargs):
        """ if reduce is true, return a 1-channel tensor, i.e. compute mean over the last dimension """
        lossImg = torch.abs(x - y) / (torch.abs(x) + torch.abs(y) + eps)
        if mask is not None:
            lossImg = lossImg[mask]
        return lossImg


class L1Loss(BaseLoss):
    def compute(self, x, y, weights=None, mask=None, **kwargs):
        lossImg = torch.abs(x - y)
        if weights is not None:
            lossImg = lossImg * weights
        if mask is not None:
            lossImg = lossImg[mask]
        return lossImg

class L2Loss(BaseLoss):
    def compute(self, x, y, weights=None, mask=None, **kwargs):
        lossImg = (x - y)**2
        if weights is not None:
            lossImg = lossImg * weights
        if mask is not None:
            lossImg = lossImg[mask]
        return lossImg

class SurfaceLoss(BaseLoss):
    def __init__(self, reduction='mean', knn_k: int = 33, filter_scale: float = 1.0, sharpness_sigma: float = 0.75):
        super().__init__(reduction=reduction, channel_dim=None)
        self.knn_tree = None
        self.knn_k = knn_k
        self.knn_mask = None
        self.filter_scale = filter_scale
        self.sharpness_sigma = sharpness_sigma

    def _build_knn(self, point_clouds):
        """
        search for KNN again set knn_tree and knn_mask attributes
        TODO(yifan): use a real Kd_tree library to be able to store the data tree and
        query at each forward pass?
        """
        # Find local neighborhood to compute weights
        with torch.autograd.enable_grad():
            points_padded = point_clouds.points_padded()

        lengths = point_clouds.num_points_per_cloud()
        knn_result = ops.knn_points(
            points_padded, points_padded, lengths, lengths, K=self.knn_k, return_nn=True)
        self.knn_mask = torch.full(
            knn_result.idx.shape, False, dtype=torch.bool, device=points_padded.device)
        # valid knn result
        for b in range(self.knn_mask.shape[0]):
            self.knn_mask[b, :lengths[b], :min(
                self.knn_k, lengths[b].item())] = True
            assert(torch.all(knn_result.dists[b][~self.knn_mask[b]] == 0))
        self.knn_tree = KNN(
            knn=knn_result.knn[:, :, 1:, :], dists=knn_result.dists[:, :, 1:], idx=knn_result.idx[:, :, 1:])
        self.knn_mask = self.knn_mask[:, :, 1:]
        assert(self.knn_mask.shape == self.knn_tree.dists.shape)

    def _denoise_normals(self, point_clouds, weights, point_clouds_filter=None, inplace=False):
        """
        robust normal mollification (Sec 4.4), i.e. replace normals with a weighted average
        from neighboring normals
        do this only for invisible points (?)
        Args:
            weights (tensors): (N,max_P,K)
        """
        lengths = point_clouds.num_points_per_cloud()
        P_total = lengths.sum().item()
        normals = point_clouds.normals_padded()

        knn_normals = ops.knn_gather(normals, self.knn_tree.idx, lengths)
        normals_denoised = torch.sum(knn_normals * weights[:, :, :, None], dim=-2) / \
            eps_denom(torch.sum(weights, dim=-1, keepdim=True))

        # get point visibility so that we update only the non-visible or out-of-mask normals
        if point_clouds_filter is not None:
            try:
                reliable_normals_mask = point_clouds_filter.visibility & point_clouds_filter.inmask
                if len(point_clouds) != reliable_normals_mask.shape[0]:
                    if len(point_clouds) == 1 and reliable_normals_mask.shape[0] > 1:
                        reliable_normals_mask = reliable_normals_mask.any(
                            dim=0, keepdim=True)
                    else:
                        ValueError("Incompatible point clouds {} and mask {}".format(
                            len(point_clouds), reliable_normals_mask.shape))

                # reset visible points normals to its original ones
                normals_denoised[reliable_normals_mask ==
                                 1] = normals[reliable_normals_mask == 1]
            except KeyError as e:
                pass

        normals_denoised_packed = ops.padded_to_packed(
            normals_denoised, point_clouds.cloud_to_packed_first_idx(), P_total)
        point_clouds = point_clouds.clone()
        point_clouds.update_normals_(normals_denoised_packed)
        return point_clouds

    def get_normal_w(self, point_clouds: PointClouds3D, normals: Optional[torch.Tensor] = None, **kwargs):
        """
        Weights exp(-\|n-ni\|^2/sharpness_sigma^2), for i in a local neighborhood
        Args:
            point_clouds: whose normals will be used for ni
            normals (tensor): (N, maxP, 3) padded normals as n, if not provided, use
                the normals from point_clouds
        Returns:
            weight per point per neighbor (N,maxP,K)
        """
        self.sharpness_sigma = kwargs.get(
            'sharpness_sigma', self.sharpness_sigma)
        inv_sigma_normal = 1 / (self.sharpness_sigma * self.sharpness_sigma)
        lengths = point_clouds.num_points_per_cloud()

        if normals is None:
            normals = point_clouds.normals_padded()
        knn_normals = ops.knn_gather(normals, self.knn_tree.idx, lengths)
        normals = torch.nn.functional.normalize(normals, dim=-1)
        knn_normals = torch.nn.functional.normalize(knn_normals, dim=-1)
        normal_diff = knn_normals - normals[:, :, None, :]

        weight = torch.exp(-torch.sum(normal_diff * normal_diff, dim=-1) * inv_sigma_normal)
        return weight

    def get_spatial_w(self, point_clouds: PointClouds3D, points: Optional[torch.Tensor] = None, **kwargs):
        """
        Weights exp(\|p-pi\|^2/sigma^2)
        """
        bbox = point_clouds.get_bounding_boxes()
        diag2 = torch.sum((bbox[..., 1] - bbox[..., 0])**2, dim=-1)
        inv_sigma_spatial = point_clouds.num_points_per_cloud().float() / diag2
        self.filter_scale = kwargs.get('filter_scale', self.filter_scale)
        if points is None:
            points = point_clouds.points_padded()
        deltap = self.knn_tree.knn - points[:, :, None, :]
        w = torch.exp(-torch.sum(deltap * deltap, dim=-1) * inv_sigma_spatial * self.filter_scale)
        return w

    def get_phi(self, point_clouds, **kwargs):
        """
        spatial weight
        (1 - \|x-xi\|^2/hi^2)^4, hi is up to 4 times of the local spacing
        Return:
            weight per point per neighbor (N, maxP, K) [1] Eq.(12)
        """
        self.knn_tree = kwargs.get('knn_tree', self.knn_tree)
        self.filter_scale = kwargs.get('filter_scale', self.filter_scale)
        local_point_spacing_sq = self.knn_tree.dists.mean(dim=-1, keepdim=True)
        h = local_point_spacing_sq * 4
        w = 1 - self.knn_tree.dists / h
        w[w < 0] = 0
        w = w * w
        w = w * w
        return w


# NOTE(yifan): Essentially an operation that updates point positions from normals.
# Can we formulate this as a pointflow (neural ODE)?
# i.e. we predict normals and integrate with neural ODE? TODO(yifan): think more!
class ProjectionLoss(SurfaceLoss):
    """
    Feature Preserving Point Set Surfaces based on Non-Linear Kernel Regression
    Cengiz Oztireli, Gaël Guennebaud, Markus Gross

    Attributes:
        filter_scale: variance of the low pass filter (default: 2)
        sharpness_sigma: [0.5 (sharp), 2 (smooth)]
    """

    def get_spatial_w(self, point_clouds, **kwargs):
        """
        meshlab implementation skip this step, we do so as well, especially
        since we don't really know what is the SDF function from points
        """
        w = torch.ones_like(self.knn_tree.dists)
        return w

    def compute(self, point_clouds: PointClouds3D, points_filter=None, rebuild_knn=False, **kwargs):
        """
        Args:
            point_clouds
            (optional) knn_tree: output from ops.knn_points excluding the query point itself
            (optional) knn_mask: mask valid knn results
        Returns:
            (P, N)
        """
        self.sharpness_sigma = kwargs.get(
            'sharpness_sigma', self.sharpness_sigma)
        self.filter_scale = kwargs.get('filter_scale', self.filter_scale)
        self.knn_tree = kwargs.get('knn_tree', self.knn_tree)
        self.knn_mask = kwargs.get('knn_mask', self.knn_mask)

        lengths = point_clouds.num_points_per_cloud()
        P_total = lengths.sum().item()
        points = point_clouds.points_padded()
        # - determine phi spatial with using local point spacing (i.e. 2*dist_to_nn)
        # - denoise normals
        # - determine w_normal
        # - mask out values outside ballneighbor i.e. d > filterSpatialScale * localPointSpacing
        # - projected distance dot(ni, x-xi)
        # - multiply and normalize the weights
        with torch.autograd.no_grad():
            if rebuild_knn or self.knn_tree is None or self.knn_tree.idx.shape[:2] != points.shape[:2]:
                self._build_knn(point_clouds)

            phi = self.get_phi(point_clouds, **kwargs)

            # robust normal mollification (Sec 4.4), i.e. replace normals with a weighted average
            # from neighboring normals Eq.(11)
            point_clouds = self._denoise_normals(
                point_clouds, phi, points_filter, inplace=False)

            # compute wn and wr
            normal_w = self.get_normal_w(point_clouds, **kwargs)
            # visibility weight
            visibility_nb = ops.knn_gather(points_filter.visibility.unsqueeze(-1), self.knn_tree.idx, lengths)
            visibility_w = visibility_nb.float()
            visibility_w[~visibility_nb] = 0.1
            # compose weights
            weights = phi * normal_w * visibility_w.squeeze(-1)

            # (B, P, k), dot product distance to surface
            knn_normals = ops.knn_gather(
                point_clouds.normals_padded(), self.knn_tree.idx, lengths)

        if get_debugging_mode():
            # points.requires_grad_(True)

            def save_grad():
                lengths = point_clouds.num_points_per_cloud()

                def _save_grad(grad):
                    dbg_tensor = get_debugging_tensor()
                    if dbg_tensor is None:
                        logger_py.error("dbg_tensor is None")
                    if grad is None:
                        logger_py.error('grad is None')
                    # a dict of list of tensors
                    dbg_tensor.pts_world_grad['proj'] = [
                        grad[b, :lengths[b]].detach().cpu() for b in range(grad.shape[0])]
                return _save_grad

            if points.requires_grad:
                dbg_tensor = get_debugging_tensor()
                dbg_tensor.pts_world['proj'] = [
                    points[b, :lengths[b]].detach().cpu() for b in range(points.shape[0])]
                handle = points.register_hook(save_grad())
                self.hooks.append(handle)

        sdf = torch.sum(
            (self.knn_tree.knn.detach() - points.unsqueeze(-2)) * knn_normals, dim=-1)

        # convert everything to packed
        weights = ops.padded_to_packed(
            weights, point_clouds.cloud_to_packed_first_idx(), P_total)
        sdf = ops.padded_to_packed(
            sdf, point_clouds.cloud_to_packed_first_idx(), P_total)

        # if get_debugging_mode():
        #     # save to dbg folder as normal
        #     from ..utils.io import save_ply
        #     save_ply('./dbg_repel_diff.ply', point_clouds.points_packed().cpu().detach(), normals=repel_vec.cpu().detach())

        distance_to_face = sdf*sdf
        # compute weighted signed distance to surface
        loss = torch.sum(
            weights * distance_to_face, dim=-1) / eps_denom(torch.sum(weights, dim=-1))

        return loss


class RepulsionLoss(SurfaceLoss):
    """
    Intend to compute the repulsion term in DSS Eq(12)~Eq(15)
    without SVD
    """

    def get_density_w(self, point_clouds: PointClouds3D, points: Optional[torch.Tensor], **kwargs):
        """
        1 + sum_i (exp(-\|x-xi\|^2/(sigma*h)^2))
        """
        inv_sigma_spatial = point_clouds.num_points_per_cloud() / 2.0
        if points is None:
            with torch.autograd.enable_grad():
                points = point_clouds.points_padded()
        deltap = self.knn_tree.knn - points[:, :, None, :]
        w = 1 + torch.sum(torch.exp(-torch.sum(deltap * deltap, dim=-1)
                          * inv_sigma_spatial), dim=-1)
        return w

    def compute(self, point_clouds: PointClouds3D, points_filter=None, rebuild_knn=True, **kwargs):

        self.knn_tree = kwargs.get('knn_tree', self.knn_tree)
        self.knn_mask = kwargs.get('knn_mask', self.knn_mask)

        lengths = point_clouds.num_points_per_cloud()
        P_total = lengths.sum().item()
        points_padded = point_clouds.points_padded()

        if not points_padded.requires_grad:
            logger_py.warn('Computing repulsion loss, but points_padded is not differentiable.')

        # Compute necessary weights to project points to local plane
        # TODO(yifan): This part is same as ProjectionLoss
        # how can we at best save repetitive computation
        with torch.autograd.no_grad():
            if rebuild_knn or self.knn_tree is None or points_padded.shape[:2] != self.knn_tree.shape[:2]:
                self._build_knn(point_clouds)
            phi = self.get_phi(point_clouds, **kwargs)
            point_clouds = self._denoise_normals(
                    point_clouds, phi, points_filter, inplace=False)

        # project the point to a local surface
        knn_diff = points_padded.unsqueeze(-2) - self.knn_tree.knn.detach()

        knn_normals = ops.knn_gather(
            point_clouds.normals_padded(), self.knn_tree.idx, lengths)
        pts_diff_proj = knn_diff - \
                (knn_diff * knn_normals).sum(dim=-1, keepdim=True) * knn_normals

        if get_debugging_mode():
            # points_padded.requires_grad_(True)

            def save_grad():
                lengths = point_clouds.num_points_per_cloud()

                def _save_grad(grad):
                    dbg_tensor = get_debugging_tensor()
                    if dbg_tensor is None:
                        logger_py.error("dbg_tensor is None")
                    if grad is None:
                        logger_py.error('grad is None')
                    # a dict of list of tensors
                    dbg_tensor.pts_world_grad['repel'] = [
                        grad[b, :lengths[b]].detach().cpu() for b in range(grad.shape[0])]
                return _save_grad

            if points_padded.requires_grad:
                dbg_tensor = get_debugging_tensor()
                dbg_tensor.pts_world['repel'] = [
                    points_padded[b, :lengths[b]].detach().cpu() for b in range(points_padded.shape[0])]
                handle = points_padded.register_hook(save_grad())
                self.hooks.append(handle)

        with torch.autograd.no_grad():
            spatial_w = self.get_spatial_w(point_clouds, **kwargs)
            # set far neighbors' spatial_w to 0
            normal_w = self.get_normal_w(point_clouds, **kwargs)
            density_w = torch.sum(spatial_w, dim=-1, keepdim=True) + 1.0
            weights = spatial_w * normal_w

        # convert everything to packed
        weights = ops.padded_to_packed(
            weights, point_clouds.cloud_to_packed_first_idx(), P_total)
        pts_diff_proj = ops.padded_to_packed(
            pts_diff_proj.contiguous().view(pts_diff_proj.shape[0], pts_diff_proj.shape[1], -1), point_clouds.cloud_to_packed_first_idx(), P_total).view(P_total, -1, 3)
        density_w = ops.padded_to_packed(
            density_w, point_clouds.cloud_to_packed_first_idx(), P_total)

        # we want to maximize this, so negative sign
        repel_vec = torch.sum(
            pts_diff_proj * weights.unsqueeze(-1), dim=1) / eps_denom(torch.sum(weights, dim=1).unsqueeze(-1))
        repel_vec = repel_vec * density_w

        loss = torch.exp(-repel_vec.abs())

        # if get_debugging_mode():
        #     # save to dbg folder as normal
        #     from ..utils.io import save_ply
        #     save_ply('./dbg_repel_diff.ply', point_clouds.points_packed().cpu().detach(), normals=repel_vec.cpu().detach())


        return loss


class IouLoss(BaseLoss):
    """
    Negative intersection/union,
    reduction applied only the the batch dimension
    """

    def compute(self, predict, target, **kwargs):
        """
        compute negative intersection/union, predict and target are same
        shape tensors [N,...]
        """
        dims = tuple(range(predict.ndimension())[1:])
        intersect = (predict * target).sum(dims)
        union = (predict + target - predict * target).sum(dims)
        result = 1.0 - intersect / eps_denom(union)
        return result

