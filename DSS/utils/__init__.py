from typing import NamedTuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import trimesh
from matplotlib import cm
import matplotlib.colors as mpc
from skimage import measure
from pytorch3d.renderer.utils import TensorProperties
from pytorch3d.structures import Pointclouds
from .. import logger_py
from .mathHelper import eps_denom


def valid_value_mask(tensor: torch.Tensor):
    return torch.isfinite(tensor) & ~torch.isnan(tensor)


def to_tensor(array, device='cpu'):
    if not isinstance(array, torch.Tensor):
        array = torch.tensor(array)
    array = array.to(device=device)
    return array


def num_points_2_cloud_to_packed_first_idx(num_points):
    cloud_to_packed_first_idx = F.pad(num_points, (1, 0), 'constant', 0)
    cloud_to_packed_first_idx = cloud_to_packed_first_idx.cumsum(0)
    return cloud_to_packed_first_idx[:-1]


def num_points_2_packed_to_cloud_idx(num_points):
    batch_size = len(num_points)
    packed_to_cloud = torch.repeat_interleave(
        torch.arange(batch_size, device=num_points.device),
        num_points,
        dim=0)
    return packed_to_cloud


def mask_from_padding(num_points):
    batch_size = num_points.shape[0]
    mask_padded = torch.full((batch_size, num_points.max().item()), True, dtype=torch.bool,
                             device=num_points.device)
    for b in range(batch_size):
        mask_padded[b, num_points[b]:] = False
    return mask_padded


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


def slice_dict(d: dict, idx: List):
    """ slice all values in a dict to take the idx
    """
    for k, v in d.items():
        if isinstance(v, dict):
            slice_dict(v, idx)
        else:
            d[k] = v[idx]
    return d


def scaler_to_color(scaler, cmap='jet'):
    """ scaler (np.array) """
    if scaler.shape[-1] == 1:
        shp = scaler.shape[:-1]
    else:
        shp = scaler.shape

    cmap = cm.get_cmap(cmap)
    normalizer = mpc.Normalize(vmin=scaler.min(), vmax=scaler.max())
    colors = normalizer(scaler.reshape(-1))
    colors = cmap(colors)[:, :3]
    colors = colors.reshape(shp + (3,))
    return colors


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
            out_dict[k] = attr.cpu().numpy()
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


def scatter_add_with_neg_idx(input: torch.Tensor, dim: int,
                             index: torch.Tensor, src: torch.Tensor, inplace: bool = False):
    mask = index >= 0
    index[~mask] = 0
    tmp = input[~mask].clone()
    if inplace:
        input.scatter_add_(dim, index, src)
        input[~mask] = tmp
        return input
    else:
        output = input.scatter_add(dim, index, src)
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


def make_image_grid(img_res, ndc=True):
    """
    img resolution (H, W), returns (H, W, 2)
    """
    # screen_x = (image_width - 1.0) / 2.0 * (1.0 - ndc_points[..., 0])
    # screen_y = (image_height - 1.0) / 2.0 * (1.0 - ndc_points[..., 1])
    H, W = img_res
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    im_grid = np.stack([xx, yy], axis=-1).astype('float32')
    if ndc:
        im_grid[..., 0] = 1 - im_grid[..., 0] / (W - 1.0) * 2.0
        im_grid[..., 1] = 1 - im_grid[..., 1] / (H - 1.0) * 2.0
    return torch.tensor(im_grid)


def get_tensor_values(tensor: torch.Tensor, p: torch.Tensor, grid_sample=True, mode='bilinear',
                      with_mask=False, squeeze_channel_dim=False):
    '''
    From DVR
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
            tensor, p, mode=mode, padding_mode='reflection')
        # (B,c,N)
        values = values.squeeze(2)
        # (B,N,c)
        values = values.permute(0, 2, 1)
    else:
        assert(h == w)
        p[:, :, 0] = (p[:, :, 0] + 1) * (w - 1) / 2
        p[:, :, 1] = (p[:, :, 1] + 1) * (h - 1) / 2
        p = p.long()
        values = tensor[torch.arange(batch_size).unsqueeze(-1), :, p[..., 1],
                        p[..., 0]]

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


def intersection_with_unit_cube(ray0, ray_direction, side_length=1.0, padding=0.1,
                                image_plane_z=1.0, eps=1e-6):
    ''' Checks if rays ray0 + d * ray_direction intersect with unit cube with
    padding padding.

    Args:
        ray0 (tensor): Start positions of the rays
        ray_direction (tensor): Directions of rays
        padding (float): Padding which is applied to the unit cube
        eps (float): The epsilon value for numerical stability
    Returns:
        intersection0: (P, 3) two intersecting points sorted in order of intersection
        intersection1: (P, 3) two intersecting points sorted in order of intersection
        mask_inside_cube: (P, ) whether the ray intersects with the unit cube
    '''
    device = ray_direction.device
    ray0 = ray0.expand_as(ray_direction)

    # calculate intersections with unit cube (< . , . >  is the dot product)
    # n is the normal of planes of the cube [1,0,0],[0,1,0],[0,0,1]
    # <n, x - p> = <n, ray0 + d * ray_direction - p_e> = 0
    # d = - <n, ray0 - p_e> / <n, ray_direction>

    # Get points on plane p_e (need two points for 6 planes)
    p_distance = side_length / 2 + padding / 2
    p_e = torch.ones(ray_direction.shape[:-1] + (6,)).to(device) * p_distance
    p_e[..., 3:] *= -1.

    # Calculate the intersection points with given formula
    nominator = p_e - torch.cat([ray0, ray0], dim=-1)
    denominator = torch.cat([ray_direction, ray_direction], dim=-1)
    # d_intersect for all 6 planes
    d_intersect = nominator / denominator
    # (P, 6, 3)
    p_intersect = ray0.unsqueeze(-2) + d_intersect.unsqueeze(-1) * \
        ray_direction.unsqueeze(-2)

    # Calculate mask where points intersect unit cube (B,N,6)
    p_mask_inside_cube = (
        (p_intersect[..., 0] <= p_distance + eps) &
        (p_intersect[..., 1] <= p_distance + eps) &
        (p_intersect[..., 2] <= p_distance + eps) &
        (p_intersect[..., 0] >= -(p_distance + eps)) &
        (p_intersect[..., 1] >= -(p_distance + eps)) &
        (p_intersect[..., 2] >= -(p_distance + eps))
    )

    # Correct rays are these which intersect exactly 2 times
    mask_inside_cube = p_mask_inside_cube.sum(-1) == 2

    if mask_inside_cube.nelement() > 0 and (not torch.any(mask_inside_cube)):
        logger_py.warning(
            "No camera rays intersect with the unit cube. Something is odd.")

    # Get interval values for p's which are valid (B,M,6,3)->(B,M,2,3)
    p_intervals = p_intersect[mask_inside_cube][p_mask_inside_cube[
        mask_inside_cube]].view(-1, 2, 3)
    p_intervals_batch = torch.zeros(
        ray_direction.shape[:-1] + (2, 3)).to(device)
    p_intervals_batch[mask_inside_cube] = p_intervals

    # Calculate ray lengths for the interval points
    d_intervals_batch = torch.zeros(ray_direction.shape[:-1] + (2,)).to(device)
    norm_ray = torch.norm(ray_direction[mask_inside_cube], dim=-1)
    if not (torch.all(norm_ray > 0)):
        logger_py.error("Ray_direction contains 0-length vectors.")
    d_intervals_batch[mask_inside_cube] = torch.stack([
        torch.norm(p_intervals[..., 0, :] -
                   ray0[mask_inside_cube], dim=-1) / norm_ray,
        torch.norm(p_intervals[..., 1, :] -
                   ray0[mask_inside_cube], dim=-1) / norm_ray,
    ], dim=-1)

    # Sort the ray lengths
    d_intervals_batch, indices_sort = d_intervals_batch.sort()
    p_intervals_batch = torch.gather(
        p_intervals_batch, -2, indices_sort.unsqueeze(-1).expand_as(p_intervals_batch))

    intersection0, intersection1 = p_intervals_batch.unbind(dim=-2)
    return intersection0, intersection1, mask_inside_cube


def intersection_with_unit_sphere(cam_pos, cam_rays, radius=1.0):
    '''
    get intersection with a centered sphere
    https://github.com/B1ueber2y/DIST-Renderer
    If no intersection, set first intersection to be the intersection
    with the plane tangent on the sphere orthogonal to the viewing axis
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
    # if torch.sum(cam_pos_dist > radius) == 0:
    #     init_zdepth = torch.zeros_like(dist)
    # else:
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


def tolerating_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    batch = [x for x in filter(lambda x: x is not None, batch)]
    return torch.utils.data.dataloader.default_collate(batch)


def get_grid_uniform(resolution, box_side_length=2.0):
    x = np.linspace(-0.5, 0.5, resolution) * box_side_length
    y = x
    z = x

    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = torch.tensor(
        np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float)

    return {"grid_points": grid_points.cuda(),
            "shortest_axis_length": box_side_length,
            "xyz": [x, y, z],
            "shortest_axis_index": 0}


def get_surface_high_res_mesh(sdf, resolution=100, box_side_length=2.0, largest_component=True):
    # get low res mesh to sample point cloud
    grid = get_grid_uniform(64, box_side_length=box_side_length)
    z = []
    points = grid['grid_points']

    for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
        z.append(sdf(pnts).detach().cpu().numpy())
    z = np.concatenate(z, axis=0)

    z = z.astype(np.float32)
    if np.sign(z.min() * z.max()) >= 0:
        return trimesh.Trimesh([])
    verts, faces, normals, values = measure.marching_cubes_lewiner(
        volume=z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                         grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
        level=0,
        spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                 grid['xyz'][0][2] - grid['xyz'][0][1],
                 grid['xyz'][0][2] - grid['xyz'][0][1]))

    verts = verts + \
        np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

    mesh_low_res = trimesh.Trimesh(verts, faces, normals)

    components = mesh_low_res.split(only_watertight=False)
    areas = np.array([c.area for c in components], dtype=np.float)
    mesh_low_res = components[areas.argmax()]

    recon_pc = trimesh.sample.sample_surface(mesh_low_res, 10000)[0]
    recon_pc = torch.from_numpy(recon_pc).float().cuda()

    # Center and align the recon pc
    s_mean = recon_pc.mean(dim=0)
    s_cov = recon_pc - s_mean
    s_cov = torch.mm(s_cov.transpose(0, 1), s_cov)
    vecs = torch.eig(s_cov, True)[1].transpose(0, 1)
    if torch.det(vecs) < 0:
        vecs = torch.mm(torch.tensor(
            [[1, 0, 0], [0, 0, 1], [0, 1, 0]]).cuda().float(), vecs)
    helper = torch.bmm(vecs.unsqueeze(0).repeat(recon_pc.shape[0], 1, 1),
                       (recon_pc - s_mean).unsqueeze(-1)).squeeze()

    grid_aligned = get_grid(helper.cpu(), resolution)

    grid_points = grid_aligned['grid_points']

    g = []
    for i, pnts in enumerate(torch.split(grid_points, 100000, dim=0)):
        g.append((torch.bmm(vecs.unsqueeze(0).repeat(pnts.shape[0], 1, 1).transpose(1, 2),
                            pnts.unsqueeze(-1)).squeeze() + s_mean).cpu().detach())
    grid_points = torch.cat(g, dim=0)

    # MC to new grid
    points = grid_points
    z = []
    for i, pnts in enumerate(torch.split(points, 50000, dim=0)):
        z.append(sdf(pnts.cuda()).detach().cpu().numpy())
    z = np.concatenate(z, axis=0)

    meshexport = None
    if (not (np.min(z) > 0 or np.max(z) < 0)):

        z = z.astype(np.float32)

        verts, faces, normals, values = measure.marching_cubes_lewiner(
            volume=z.reshape(grid_aligned['xyz'][1].shape[0], grid_aligned['xyz'][0].shape[0],
                             grid_aligned['xyz'][2].shape[0]).transpose([1, 0, 2]),
            level=0,
            spacing=(grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1],
                     grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1],
                     grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1]))

        verts = torch.from_numpy(verts).cuda().float()
        verts = torch.bmm(vecs.unsqueeze(0).repeat(verts.shape[0], 1, 1).transpose(1, 2),
                          verts.unsqueeze(-1)).squeeze()
        verts = (verts + grid_points[0].cuda()).cpu().numpy()

        meshexport = trimesh.Trimesh(verts, faces, normals)

    if largest_component:
        components = meshexport.split(only_watertight=False)
        areas = np.array([c.area for c in components], dtype=np.float)
        meshexport = components[areas.argmax()]

    return meshexport


def get_grid(points, resolution):
    eps = 0.2
    input_min = torch.min(points, dim=0)[0].squeeze().numpy()
    input_max = torch.max(points, dim=0)[0].squeeze().numpy()

    bounding_box = input_max - input_min
    shortest_axis = np.argmin(bounding_box)
    if (shortest_axis == 0):
        x = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(x) - np.min(x)
        y = np.arange(input_min[1] - eps, input_max[1] + length /
                      (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
        z = np.arange(input_min[2] - eps, input_max[2] + length /
                      (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
    elif (shortest_axis == 1):
        y = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(y) - np.min(y)
        x = np.arange(input_min[0] - eps, input_max[0] + length /
                      (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
        z = np.arange(input_min[2] - eps, input_max[2] + length /
                      (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
    elif (shortest_axis == 2):
        z = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(z) - np.min(z)
        x = np.arange(input_min[0] - eps, input_max[0] + length /
                      (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))
        y = np.arange(input_min[1] - eps, input_max[1] + length /
                      (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))

    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = torch.tensor(
        np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float).cuda()
    return {"grid_points": grid_points,
            "shortest_axis_length": length,
            "xyz": [x, y, z],
            "shortest_axis_index": shortest_axis}


def get_visible_points(point_clouds, cameras, depth_merge_threshold=0.01):
    """ Returns packed visibility """
    from pytorch3d.renderer import PointsRasterizationSettings, PointsRasterizer
    img_size = 256
    raster = PointsRasterizer(raster_settings=PointsRasterizationSettings(
        image_size=img_size, points_per_pixel=20, radius=3 *2.0 / img_size))
    frag = raster(point_clouds, cameras=cameras)
    depth_occ_mask = (frag.zbuf[..., 1:] -
                      frag.zbuf[..., :1]) < depth_merge_threshold
    occ_idx = frag.idx[...,1:][~depth_occ_mask]
    frag.idx[..., 1:][~depth_occ_mask] = -1
    mask = get_per_point_visibility_mask(point_clouds, frag)
    occ_idx = occ_idx[occ_idx!=-1]
    occ_idx = occ_idx.unique()
    mask[occ_idx.long()] = False
    return mask
