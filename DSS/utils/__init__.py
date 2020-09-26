from typing import NamedTuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_sampling
import numpy as np
from skimage import measure
import cv2
from pytorch3d.renderer.utils import TensorProperties, convert_to_tensors_and_broadcast
from pytorch3d.structures import Pointclouds, Meshes
from .. import logger_py
from .mathHelper import eps_denom


def valid_value_mask(tensor: torch.Tensor):
    return torch.isfinite(tensor) & ~torch.isnan(tensor)


def check_weights(state_dict):
    ''' Checks weights for illegal values.

    Args:
        params (tensor): parameter tensor
    '''
    for k, v in state_dict.items():
        if torch.isnan(v).any():
            import pdb
            pdb.set_trace()
            logger_py.warn('NaN Values detected in model weight %s.' % k)
        if not torch.isfinite(v).all():
            import pdb
            pdb.set_trace()
            logger_py.warn('Infinite Values detected in model weight %s.' % k)


def get_class_from_string(cls_str):
    import importlib
    i = cls_str.rfind('.')
    mod = importlib.import_module(cls_str[:i])
    clss = getattr(mod, cls_str[i + 1:])
    return clss


def convert_tensor_property_to_value_dict(tensor_property: TensorProperties) -> dict:
    """
    Convert a TensorProperties object to a dictionary,
    saving all its tensor and number type attributes
    """
    from numbers import Number
    out_dict = {}
    for k in dir(tensor_property):
        if k[0] == "_":
            continue
        attr = getattr(tensor_property, k)
        if torch.is_tensor(attr):
            out_dict[k] = attr.cpu().tolist()
        elif isinstance(attr, Number):
            out_dict[k] = attr
    return out_dict


def mask_padded_to_list(values: torch.Tensor, mask: torch.Tensor) -> List[torch.Tensor]:
    """
    padded_to_list with mask
    Args:
        values (tensor(number)): (N, ..., C)
        mask   (tensor(bool)): (N, ...) bool values
    Returns:
        value_list (List(tensors)): (N,) list of filtered values (Pi, C) in each batch element,
            where Pi is the number of true values in mask[i]
    """
    from pytorch3d.structures import packed_to_list
    batch_size = values.shape[0]
    value_packed = values[mask]
    num_true_in_batch = mask.view(batch_size, -1).sum(dim=1)
    value_list = packed_to_list(value_packed, num_true_in_batch.tolist())
    return value_list

def mask_packed_to_list(value: torch.Tensor, num_points_per_cloud: torch.Tensor, mask: torch.Tensor) -> List[torch.Tensor]:
    """
    packed to list but consider mask
    """
    assert(mask.shape[0] == value.shape[0])
    assert(mask.ndim == (value.ndim - 1))

    value_list = torch.split(value, num_points_per_cloud.tolist())
    mask_list = torch.split(mask, num_points_per_cloud.tolist())
    return [x[m] for x, m in zip(value_list, mask_list)]


def reduce_mask_padded(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Remove the invalid values in a padded tensor form as much as possible
    Args:
        values (tensor(number)): (N, P, ..., C)
        mask   (tensor(bool)): (N, P, ...) bool values
    Returns:
        reduced (tensor(number)): (N, Pmax, ..., C) Pmax is the maximum number of
            True values in the batches of mask.
    """
    from pytorch3d.ops import packed_to_padded
    batch_size = values.shape[0]
    value_packed = values[mask]
    first_idx = torch.zeros(
        (values.shape[0],), device=values.device, dtype=torch.long)
    num_true_in_batch = mask.view(batch_size, -1).sum(dim=1)
    first_idx[1:] = num_true_in_batch.cumsum(dim=0)[:-1]
    value_padded = packed_to_padded(
        value_packed, first_idx, num_true_in_batch.max().item())
    return value_padded


def gather_with_neg_idx(input: torch.Tensor, dim: int, index: torch.Tensor,
                        out: Optional[torch.Tensor] = None, sparse_grad: bool = False) -> torch.Tensor:
    """
    gather function where idx can include negative values - in this case gathered value is 0
    The requirements for input dimensions are same as torch.gather
    Args:
        data (tensor)
        idx (tensor)
    """
    mask = index >= 0
    index[~mask] = 0  # a random index
    out = torch.gather(input, dim, index, out=out, sparse_grad=sparse_grad)
    out[~mask] = 0
    return out


def scatter_with_neg_idx(input: torch.Tensor, dim: int,
                         index: torch.Tensor, src: torch.Tensor, inplace: bool = False):
    mask = index >= 0
    index[~mask] = 0
    tmp = input[~mask].clone()
    if inplace:
        input.scatter_(dim, index, src)
        input[~mask] = tmp
        return input
    else:
        output = input.scatter(dim, index, src)
        output[~mask] = tmp
        return output


def gather_batch_to_packed(data, batch_idx):
    """
    Reformat batched tensors to match packed tensor.
    useful when batched tensors e.g. shape (N, 3) need to be
    multiplied with another tensor which has a different first dimension
    e.g. packed vertices of shape (V, 3).
    batch_idx can come from Meshes.verts_packed_to_mesh_idx() or Pointclouds.packed_to_cloud_idx()
    """
    if data.shape[0] > 1:
        # There are different values for each batch element
        # so gather these using the batch_idx.
        # First clone the input batch_idx tensor before
        # modifying it.
        _batch_idx = batch_idx.clone()
        idx_dims = _batch_idx.shape
        tensor_dims = data.shape
        if len(idx_dims) > len(tensor_dims):
            msg = "batch_idx cannot have more dimensions than data. "
            msg += "got shape %r and data has shape %r"
            raise ValueError(msg % (idx_dims, tensor_dims))
        if idx_dims != tensor_dims:
            # To use torch.gather the index tensor (_batch_idx) has
            # to have the same shape as the input tensor.
            new_dims = len(tensor_dims) - len(idx_dims)
            new_shape = idx_dims + (1,) * new_dims
            expand_dims = (-1,) + tensor_dims[1:]
            _batch_idx = _batch_idx.view(*new_shape)
            _batch_idx = _batch_idx.expand(*expand_dims)

        data = data.gather(0, _batch_idx)
    return data


def get_tensor_values(tensor: torch.Tensor, p: torch.Tensor, grid_sample=True, mode='nearest',
                      with_mask=False, squeeze_channel_dim=False):
    '''
    Returns values from tensor at given location p.

    Args:
        tensor (tensor): tensor of size B x C x H x W
        p (tensor): position values scaled between [-1, 1] and
            of size B x N x 2
        grid_sample (boolean): whether to use grid sampling
        mode (string): what mode to perform grid sampling in
        with_mask (bool): whether to return the mask for invalid values
        squeeze_channel_dim (bool): whether to squeeze the channel dimension
            (only applicable to 1D data)
    '''
    batch_size, _, h, w = tensor.shape

    if grid_sample:
        # (B,1,N,2)
        p = p.unsqueeze(1)
        # (B,c,1,N)
        # # NOTE pytorch 1.5 returns 0.0 for -1/1 grid if padding_mode is zero,
        # # so we need to make sure that grid p is indeed between -1 and 1
        # if not (p.min() >= -1.0 and p.max() <= 1.0).item():
        #     raise ValueError("grid value out of range [-1, 1].")

        values = torch.nn.functional.grid_sample(
            tensor, p, mode=mode, padding_mode='reflection', align_corners=True)
        # (B,c,N)
        values = values.squeeze(2)
        # (B,N,c)
        values = values.permute(0, 2, 1)
    else:
        p[:, :, 0] = (p[:, :, 0] + 1) * (w) / 2
        p[:, :, 1] = (p[:, :, 1] + 1) * (h) / 2
        p = p.long()
        values = tensor[torch.arange(batch_size).unsqueeze(-1), :, p[:, :, 1],
                        p[:, :, 0]]

    if with_mask:
        mask = valid_value_mask(values)
        if squeeze_channel_dim:
            mask = mask.squeeze(-1)

    if squeeze_channel_dim:
        values = values.squeeze(-1)

    if with_mask:
        return values, mask
    return values




def get_per_point_visibility_mask(pointclouds: Pointclouds,
                                  fragments: NamedTuple) -> Pointclouds:
    """
    compute per-point visibility (0/1), append value to pointclouds features
    Returns:
        boolean mask for packed tensors (P_total,)
    """
    P_total = pointclouds.num_points_per_cloud().sum().item()
    P_max = pointclouds.num_points_per_cloud().max().item()
    try:
        mask = fragments.occupancy.bool()  # float
    except:
        mask = fragments.idx[..., 0] >= 0  # bool

    pts_visibility = torch.full(
        (P_total,), False, dtype=torch.bool, device=pointclouds.device)

    # all rendered points (indices in packed points)
    visible_idx = fragments.idx[mask].unique().long().view(-1)
    visible_idx = visible_idx[visible_idx >= 0]
    pts_visibility[visible_idx] = True

    return pts_visibility


def intersection_with_unit_sphere(cam_pos, cam_rays, radius=1.0, depth_range=(1.0, 10)):
    '''
    get intersection with a centered sphere
    https://github.com/B1ueber2y/DIST-Renderer
    If doesn't intersect with the sphere, return two points along the ray with raylength equals
    the minimum and maximum of the depth_range respectively.
    If the camera is inside the unitsphere, set the first intersection to be the camera position
    Args:
        cam_pos (torch.FloatTensor):	 (1, *, 3) or (N, *, 3)
        cam_rays (torch.FloatTensor):	normalized viewing ray (N, *, 3)
        radius (float): sphere radius
    Returns:
        intersection0, intersection1: (N,*,3) intersections,
    '''
    if cam_pos.ndim != cam_rays.ndim:
        cam_rays = cam_rays.view(cam_rays.shape[0], -1, 3)
        cam_pos = cam_pos.view(cam_pos.shape[0], 1, 3)

    assert(cam_pos.ndim == cam_rays.ndim)

    # distance from cam_rays to center
    p, q = cam_pos, cam_rays  # (N,*,3), (N,*,3)
    ptq = (p * q).sum(dim=-1)  # (N,*)
    # middle point between the two intersections
    mid = p - ptq[..., None] * q  # (N,*,3)
    dist = torch.norm(mid, p=2, dim=-1)  # (N, *)
    cam_dist = torch.norm(p, dim=-1)
    valid_mask = (dist <= radius)
    value = radius ** 2 - dist ** 2

    # length between the intersections, valid_mask indicates whether there's an intersection
    # set ~valid_mask to be intersection with the plane tangent to the unit sphere and orthogonal
    # to the camera axis
    maxbound_marching_zdepth = torch.full_like(dist, 10.0)
    maxbound_marching_zdepth[valid_mask] = 2 * torch.sqrt(value[valid_mask])

    cam_pos_dist = torch.norm(cam_pos, dim=-1)  # (N,*)

    # If the cameras are inside the sphere, then set the initial zdepth to 0
    if torch.sum(cam_pos_dist > radius) == 0:
        init_zdepth = torch.zeros_like(dist)
    else:
        # First intersection depth
        init_zdepth = torch.zeros_like(dist)
        init_zdepth_valid = torch.sqrt(
            cam_pos_dist.expand_as(dist)[valid_mask] ** 2 - dist[valid_mask] ** 2) - maxbound_marching_zdepth[valid_mask] / 2.0  # (N)
        init_zdepth[valid_mask] = init_zdepth_valid
        # If no intersection, set first intersection to be the intersection
        # with the plane tangent on the sphere orthogonal to the viewing axis
        init_zdepth[~valid_mask] = ((cam_dist - radius) /
                                    eps_denom(-ptq / cam_dist))[~valid_mask]

    intersection0 = init_zdepth.unsqueeze(-1) * cam_rays + cam_pos
    intersection1 = maxbound_marching_zdepth[...,
                                             None] * cam_rays + intersection0
    invalid_far_zdepth = ((radius + cam_dist) /
                          eps_denom(-ptq / cam_dist))[~valid_mask]
    cam_pos = cam_pos.expand_as(cam_rays)
    intersection1[~valid_mask] = invalid_far_zdepth.unsqueeze(
        -1) * cam_rays[~valid_mask] + cam_pos[~valid_mask]

    return intersection0, intersection1, valid_mask


class CannyFilter(nn.Module):
    """
    Edge filter for image (N,C,H,W) inputs. Blur with Guassian (mu, sigma),
    horizontal and vertical sobel filter, additionally apply non-maximum-suppresion
    and thresholding
    """

    def __init__(self,
                 k_gaussian=3,
                 mu=0,
                 sigma=1,
                 k_sobel=3,
                 nms=False,
                 thresholding=False,
                 low_threshold=0.1,
                 high_threshold=0.6,
                 device='cpu'):
        super().__init__()
        # device
        self.device = device

        # gaussian
        if k_gaussian > 1:
            gaussian_2D = CannyFilter.get_gaussian_kernel(
                k_gaussian, mu, sigma)
            self.gaussian_filter_weight = torch.from_numpy(
                gaussian_2D).to(dtype=torch.float, device=device).view(1, 1, k_gaussian, k_gaussian)
            self.gaussian_filter_weight = torch.nn.Parameter(self.gaussian_filter_weight,
                                                             requires_grad=False)
        else:
            self.gaussian_filter_weight = None

        # sobel
        sobel_2D = CannyFilter.get_sobel_kernel(k_sobel)
        self.sobel_filter_x_weight = torch.from_numpy(
            sobel_2D).to(dtype=torch.float, device=device).view(1, 1, k_sobel, k_sobel)
        self.sobel_filter_x_weight = torch.nn.Parameter(self.sobel_filter_x_weight,
                                                        requires_grad=False)

        self.sobel_filter_y_weight = torch.from_numpy(
            sobel_2D.T).to(dtype=torch.float, device=device).view(1, 1, k_sobel, k_sobel)
        self.sobel_filter_y_weight = torch.nn.Parameter(self.sobel_filter_y_weight,
                                                        requires_grad=False)
        # thin
        self.nms = nms
        if nms:
            thin_kernels = CannyFilter.get_thin_kernels()
            directional_kernels = np.stack(thin_kernels)

            self.directional_filter_weight = torch.from_numpy(
                directional_kernels).to(dtype=torch.float, device=device).view(
                    8, 1, thin_kernels[0].shape[0], thin_kernels[0].shape[1])
            self.directional_filter_weight = torch.nn.Parameter(self.directional_filter_weight,
                                                                requires_grad=False)
        # hysteresis
        self.thresholding = thresholding
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    @classmethod
    def get_thin_kernels(cls, start=0, end=360, step=45):
        """
        Returns 8 (k, k) numpy arrays
        """
        k_thin = 3  # actual size of the directional kernel
        # increase for a while to avoid interpolation when rotating
        k_increased = k_thin + 2

        # get 0° angle directional kernel
        thin_kernel_0 = np.zeros((k_increased, k_increased))
        thin_kernel_0[k_increased // 2, k_increased // 2] = 1
        thin_kernel_0[k_increased // 2, k_increased // 2 + 1:] = -1

        # rotate the 0° angle directional kernel to get the other ones
        thin_kernels = []
        for angle in range(start, end, step):
            (h, w) = thin_kernel_0.shape
            # get the center to not rotate around the (0, 0) coord point
            center = (w // 2, h // 2)
            # apply rotation
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
            kernel_angle_increased = cv2.warpAffine(
                thin_kernel_0, rotation_matrix, (w, h), cv2.INTER_NEAREST)

            # get the k=3 kerne
            kernel_angle = kernel_angle_increased[1:-1, 1:-1]
            # because of the interpolation
            is_diag = (abs(kernel_angle) == 1)
            kernel_angle = kernel_angle * is_diag   # because of the interpolation
            thin_kernels.append(kernel_angle)
        return thin_kernels

    @classmethod
    def get_gaussian_kernel(cls, k=3, mu=0, sigma=1, normalize=True) -> np.array:
        """
        Returns (k, k) numpy array
        """
        # compute 1 dimension gaussian
        gaussian_1D = np.linspace(-1, 1, k)
        # compute a grid distance from center
        x, y = np.meshgrid(gaussian_1D, gaussian_1D)
        distance = (x ** 2 + y ** 2) ** 0.5

        # compute the 2 dimension gaussian
        gaussian_2D = np.exp(-(distance - mu) ** 2 / (2 * sigma ** 2))
        gaussian_2D = gaussian_2D / (2 * np.pi * sigma ** 2)

        # normalize part (mathematically)
        if normalize:
            gaussian_2D = gaussian_2D / np.sum(gaussian_2D)
        return gaussian_2D

    @classmethod
    def get_sobel_kernel(cls, k=3):
        """
        Returns (k, k) numpy array
        """
        # get range
        range = np.linspace(-(k // 2), k // 2, k)
        # compute a grid the numerator and the axis-distances
        x, y = np.meshgrid(range, range)
        sobel_2D_numerator = x
        sobel_2D_denominator = (x ** 2 + y ** 2)
        sobel_2D_denominator[:, k // 2] = 1  # avoid division by zero
        sobel_2D = sobel_2D_numerator / sobel_2D_denominator
        return sobel_2D

    def forward(self, img, hysteresis=False):
        # set the setps tensors
        B, C, H, W = img.shape
        # gaussian
        if self.gaussian_filter_weight:
            padding = tuple(
                x // 2 for x in self.gaussian_filter_weight.shape[2:] for _ in range(2))
            img = F.pad(img, padding, mode='replicate')
            blurred = F.conv2d(img.view(-1, 1, img.shape[2], img.shape[3]), self.gaussian_filter_weight,
                               ).view(B, C, H, W)
        else:
            blurred = img

        padding = tuple(
            x // 2 for x in self.sobel_filter_x_weight.shape[2:] for _ in range(2))
        blurred = F.pad(blurred, padding, mode='replicate')
        grad_x = F.conv2d(blurred.view(-1, 1, blurred.shape[2], blurred.shape[3]), self.sobel_filter_x_weight,
                          ).view(B, C, H, W).mean(dim=1, keepdim=True)
        grad_y = F.conv2d(blurred.view(-1, 1, blurred.shape[2], blurred.shape[3]), self.sobel_filter_y_weight,
                          ).view(B, C, H, W).mean(dim=1, keepdim=True)
        grad_magnitude = (grad_x ** 2 + grad_y ** 2)
        output = grad_magnitude

        # thick edges
        if self.nms:
            grad_orientation = torch.atan(grad_y / grad_x)
            grad_orientation = grad_orientation * \
                (360 / np.pi) + 180  # convert to degree
            grad_orientation = torch.round(
                grad_orientation / 45) * 45  # keep a split by 45

            # thin edges
            padding = tuple(
                x // 2 for x in self.directional_filter_weight.shape[2:] for _ in range(2))
            grad_magnitude_padded = F.pad(
                grad_magnitude, padding, mode='replicate')
            directional = F.conv2d(grad_magnitude_padded, self.directional_filter_weight,
                                   )
            # get indices of positive and negative directions
            positive_idx = (grad_orientation / 45) % 8
            negative_idx = ((grad_orientation / 45) + 4) % 8
            thin_edges = grad_magnitude.clone()
            # non maximum suppression direction by direction
            for pos_i in range(4):
                neg_i = pos_i + 4
                # get the oriented grad for the angle
                is_oriented_i = (positive_idx == pos_i) * 1
                is_oriented_i = is_oriented_i + (positive_idx == neg_i) * 1
                pos_directional = directional[:, pos_i]
                neg_directional = directional[:, neg_i]
                selected_direction = torch.stack(
                    [pos_directional, neg_directional])

            # get the local maximum pixels for the angle
            is_max = selected_direction.min(dim=0)[0] > 0.0
            is_max = torch.unsqueeze(is_max, dim=1)

            # apply non maximum suppression
            to_remove = (is_max == 0) * 1 * (is_oriented_i) > 0
            thin_edges[to_remove] = 0.0
            output = thin_edges

        # thresholds
        if self.thresholding:
            if self.low_threshold is not None:
                low = output > self.low_threshold

                if self.high_threshold is not None:
                    high = output > self.high_threshold
                    # get black/gray/white only
                    thin_edges = low * 0.5 + high * 0.5

                    # # get weaks and check if they are high or not
                    # weak = (thin_edges == 0.5) * 1
                    # thin_edges = F.conv2d(thin_edges, self.hysteresis_weight,
                    #                       padding=1, bias=None)
                    # weak_is_high = (thin_edges > 1) * weak
                    # thin_edges = high * 1.0 + weak_is_high * 1.0
                else:
                    thin_edges = low * 1
            output = thin_edges

        return output


class ImageSaliencySampler(nn.Module):
    """
    Returns the normalized image pixels (x,y) (N,P,2) based on edge saliency
    """

    def __init__(self, k_gaussian=0,
                 mu=0,
                 sigma=1,
                 k_sobel=3,
                 nms=False,
                 thresholding=False,
                 low_threshold=0.1,
                 high_threshold=0.5,
                 patch_size=1,
                 jitter=False,
                 device='cpu'):
        """
        Args:
            k_gaussian, mu, simga: parameters for the gaussian blur kernel
            k_sobel: kernel size for the sobel kernel
            nms: apply non-maximum-suppression
            thresholding: apply final global tone mapping
            patch_size: sample patches of pixels at the center of the selected salient pixels
        """
        super().__init__()
        self.edge_extractor = CannyFilter(k_gaussian=k_gaussian,
                                          mu=mu,
                                          sigma=sigma,
                                          k_sobel=k_sobel,
                                          nms=nms,
                                          thresholding=thresholding,
                                          low_threshold=low_threshold,
                                          high_threshold=high_threshold,
                                          device=device)
        self.patch_size = patch_size
        self.jitter = jitter

    def forward(self, n_pixels: int, image: torch.Tensor, patch_size: Optional[int] = None,
                jitter: Optional[bool] = None):
        """
        sample n_pixels based on the magnitude of image gradient
        Args:
            image: (N, C, H, W)
            n_pixels: number of points to sample
        Returns:
            (N, n_pixels*, 2) where n_patches * (patch_size**2) = n_pixels*
        """
        if jitter is None:
            jitter = self.jitter
        if patch_size is None:
            patch_size = self.patch_size

        n_patches = n_pixels // (patch_size * patch_size)
        # image gradient
        # 0. downsample image to 128x128
        batch_size = image.shape[0]
        w_step = 2. / image.shape[3]
        h_step = 2. / image.shape[2]
        H = min(image.shape[2], 128)
        W = min(image.shape[3], 128)
        image_small = F.interpolate(
            image, (H, W), mode='bilinear', align_corners=True)
        # 1. image gradient magnitude as probablity
        # 2. sample points with probability
        thin_edge = self.edge_extractor(image_small)
        thin_edge /= thin_edge.sum(dim=[2, 3], keepdim=True)

        xy_small = torch.stack(
            [torch_sampling.choice(torch.arange(H * W, device=thin_edge.device),
                                   n_patches, (n_patches > (thin_edge[b] > 0).sum()), thin_edge[b].view(H * W))
             for b in range(batch_size)], dim=0)  # (B, H*W)
        # to [0~H, 0~W] (2,B,H*W)
        xy_small = torch.stack((xy_small % W, xy_small // W), dim=0).float()
        # to [-1~1, -1~1]
        xy_small[0] = (xy_small[0] * 2.0 / W - 1.0)
        xy_small[1] = (xy_small[1] * 2.0 / H - 1.0)
        xy_small = xy_small.permute([1, 2, 0]).contiguous()

        # add some small noise to the sample position, variance 3 pixel-size
        if jitter:
            xy_small += torch.randn_like(xy_small) * min(6.0 / H, 6.0 / W)

        # sample patch
        patch_arange = torch.arange(
            patch_size, device=xy_small.device) - patch_size // 2
        x_offset, y_offset = torch.meshgrid(patch_arange, patch_arange)
        patch_offsets = torch.stack(
            [x_offset.reshape(-1), y_offset.reshape(-1)],
            dim=1).view(1, 1, -1, 2).repeat(batch_size, n_patches, 1, 1).float()

        patch_offsets[:, :, :, 0] *= w_step
        patch_offsets[:, :, :, 1] *= h_step

        p = xy_small.view(batch_size, -1, 1, 2) + patch_offsets
        return p.view(batch_size, -1, 2)


def tolerating_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    batch = [x for x in filter(lambda x: x is not None, batch)]
    return torch.utils.data.dataloader.default_collate(batch)
