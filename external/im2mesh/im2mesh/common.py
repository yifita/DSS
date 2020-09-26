import torch
# from im2mesh.utils.libkdtree import KDTree
import numpy as np
import logging
from copy import deepcopy
import plyfile
import os


logger_py = logging.getLogger(__name__)


def rgb2gray(rgb):
    ''' rgb of size B x h x w x 3
    '''
    r, g, b = rgb[:, :, :, 0], rgb[:, :, :, 1], rgb[:, :, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


def sample_patch_points(batch_size, n_points, patch_size=1,
                        image_resolution=(128, 128), continuous=True):
    ''' Returns sampled points in the range [-1, 1].

    Args:
        batch_size (int): required batch size
        n_points (int): number of points to sample
        patch_size (int): size of patch; if > 1, patches of size patch_size
            are sampled instead of individual points
        image_resolution (tuple): image resolution (required for calculating
            the pixel distances)
        continuous (bool): whether to sample continuously or only on pixel
            locations
    '''
    assert(patch_size > 0)
    # Calculate step size for [-1, 1] that is equivalent to a pixel in
    # original resolution
    h_step = 1. / image_resolution[0]
    w_step = 1. / image_resolution[1]
    # Get number of patches
    patch_size_squared = patch_size ** 2
    n_patches = int(n_points / patch_size_squared)
    if continuous:
        p = torch.rand(batch_size, n_patches, 2)  # [0, 1]
    else:
        px = torch.randint(0, image_resolution[1], size=(
            batch_size, n_patches, 1)).float() / (image_resolution[1] - 1)
        py = torch.randint(0, image_resolution[0], size=(
            batch_size, n_patches, 1)).float() / (image_resolution[0] - 1)
        p = torch.cat([px, py], dim=-1)
    # Scale p to [0, (1 - (patch_size - 1) * step) ]
    p[:, :, 0] *= 1 - (patch_size - 1) * w_step
    p[:, :, 1] *= 1 - (patch_size - 1) * h_step

    # Add points
    patch_arange = torch.arange(patch_size)
    x_offset, y_offset = torch.meshgrid(patch_arange, patch_arange)
    patch_offsets = torch.stack(
        [x_offset.reshape(-1), y_offset.reshape(-1)],
        dim=1).view(1, 1, -1, 2).repeat(batch_size, n_patches, 1, 1).float()

    patch_offsets[:, :, :, 0] *= w_step
    patch_offsets[:, :, :, 1] *= h_step

    # Add patch_offsets to points
    p = p.view(batch_size, n_patches, 1, 2) + patch_offsets

    # Scale to [-1, x]
    p = p * 2 - 1

    p = p.view(batch_size, -1, 2)

    amax, amin = p.max(), p.min()
    assert(amax <= 1. and amin >= -1.)

    return p


def get_proposal_points_in_unit_cube(ray0, ray_direction, padding=0.1,
                                     eps=1e-6, n_steps=40):
    ''' Returns n_steps equally spaced points inside the unit cube on the rays
    cast from ray0 with direction ray_direction.

    This function is used to get the ray marching points {p^ray_j} for a given
    camera position ray0 and
    a given ray direction ray_direction which goes from the camera_position to
    the pixel location.

    NOTE: The returned values d_proposal are the lengths of the ray:
        p^ray_j = ray0 + d_proposal_j * ray_direction

    Args:
        ray0 (tensor): Start positions of the rays
        ray_direction (tensor): Directions of rays
        padding (float): Padding which is applied to the unit cube
        eps (float): The epsilon value for numerical stability
        n_steps (int): number of steps
    '''
    batch_size, n_pts, _ = ray0.shape
    device = ray0.device

    p_intervals, d_intervals, mask_inside_cube = \
        check_ray_intersection_with_unit_cube(ray0, ray_direction, padding,
                                              eps)
    d_proposal = d_intervals[:, :, 0].unsqueeze(-1) + \
        torch.linspace(0, 1, steps=n_steps).to(device).view(1, 1, -1) * \
        (d_intervals[:, :, 1] - d_intervals[:, :, 0]).unsqueeze(-1)
    d_proposal = d_proposal.unsqueeze(-1)

    return d_proposal, mask_inside_cube


def check_ray_intersection_with_unit_cube(ray0, ray_direction, padding=0.1,
                                          image_plane_z=1.0, eps=1e-6):
    ''' Checks if rays ray0 + d * ray_direction intersect with unit cube with
    padding padding.

    It returns the two intersection points (B,N,2,3) as well as the sorted ray lengths
    d (B,N,2) and the mask for points inside the unit cube (B,N).

    Args:
        ray0 (tensor): Start positions of the rays
        ray_direction (tensor): Directions of rays
        padding (float): Padding which is applied to the unit cube
        eps (float): The epsilon value for numerical stability
    Returns:
        p_intervals_batch: (B, P, 2, 3) two intersecting points sorted in order of intersection
        d_intervals_batch: (B, P, 2) ray length at the intersecting points
        mask_inside_cube: (B, P) whether the ray intersects with the unit cube
    '''
    batch_size, n_pts, _ = ray0.shape
    device = ray0.device

    # calculate intersections with unit cube (< . , . >  is the dot product)
    # n is the normal of planes of the cube [1,0,0],[0,1,0],[0,0,1]
    # <n, x - p> = <n, ray0 + d * ray_direction - p_e> = 0
    # d = - <n, ray0 - p_e> / <n, ray_direction>

    # Get points on plane p_e (need two points for 6 planes)
    p_distance = 0.5 + padding / 2
    p_e = torch.ones(batch_size, n_pts, 6).to(device) * p_distance
    p_e[:, :, 3:] *= -1.

    # Calculate the intersection points with given formula
    nominator = p_e - ray0.repeat(1, 1, 2)
    denominator = ray_direction.repeat(1, 1, 2)
    # d_intersect for all 6 planes
    d_intersect = nominator / denominator
    # (B, N, 6, 3)
    p_intersect = ray0.unsqueeze(-2) + d_intersect.unsqueeze(-1) * \
        ray_direction.unsqueeze(-2)

    # Calculate mask where points intersect unit cube (B,N,6)
    p_mask_inside_cube = (
        (p_intersect[:, :, :, 0] <= p_distance + eps) &
        (p_intersect[:, :, :, 1] <= p_distance + eps) &
        (p_intersect[:, :, :, 2] <= p_distance + eps) &
        (p_intersect[:, :, :, 0] >= -(p_distance + eps)) &
        (p_intersect[:, :, :, 1] >= -(p_distance + eps)) &
        (p_intersect[:, :, :, 2] >= -(p_distance + eps))
    ).cpu()

    # Correct rays are these which intersect exactly 2 times
    mask_inside_cube = p_mask_inside_cube.sum(-1) == 2

    if ~torch.any(mask_inside_cube):
        logger_py.warning(
            "No camera rays intersect with the unit cube. Something is odd.")

    # Get interval values for p's which are valid (B,M,6,3)->(B,M,2,3)
    p_intervals = p_intersect[mask_inside_cube][p_mask_inside_cube[
        mask_inside_cube]].view(-1, 2, 3)
    p_intervals_batch = torch.zeros(batch_size, n_pts, 2, 3).to(device)
    p_intervals_batch[mask_inside_cube] = p_intervals

    # Calculate ray lengths for the interval points
    d_intervals_batch = torch.zeros(batch_size, n_pts, 2).to(device)
    norm_ray = torch.norm(ray_direction[mask_inside_cube], dim=-1)
    if not (torch.all(norm_ray > 0)):
        logger_py.error("Ray_direction contains 0-length vectors.")
    d_intervals_batch[mask_inside_cube] = torch.stack([
        torch.norm(p_intervals[:, 0] -
                   ray0[mask_inside_cube], dim=-1) / norm_ray,
        torch.norm(p_intervals[:, 1] -
                   ray0[mask_inside_cube], dim=-1) / norm_ray,
    ], dim=-1)

    # Sort the ray lengths
    d_intervals_batch, indices_sort = d_intervals_batch.sort()
    p_intervals_batch = p_intervals_batch[
        torch.arange(batch_size).view(-1, 1, 1),
        torch.arange(n_pts).view(1, -1, 1),
        indices_sort
    ]

    check_tensor(p_intervals_batch, 'p_intervals_batch')
    check_tensor(d_intervals_batch, 'd_intervals_batch')
    return p_intervals_batch, d_intervals_batch, mask_inside_cube


def intersect_camera_rays_with_unit_cube(
        pixels, cameras, padding=0.1, eps=1e-6,
        use_ray_length_as_depth=True):
    ''' Returns the intersection points of ray cast from camera origin to
    pixel points p on the image plane.

    The function returns the intersection points (B,N,2,3) as well the depth values (B,N,2) and
    a mask (B,N) specifying which ray intersects the unit cube.

    Args:
        pixels (tensor): Pixel points on image plane (range [-1, 1])
        cameras (pytorch3d cameras): Cameras object
        padding (float): Padding which is applied to the unit cube
        eps (float): The epsilon value for numerical stability
        use_ray_length_as_depth (bool): use ray length instead of instead of z-value as depth
    '''
    batch_size, n_points, _ = pixels.shape

    pixel_world = image_points_to_world(
        pixels, cameras)
    camera_world = origin_to_world(n_points, cameras)
    ray_vector = (pixel_world - camera_world)
    p_cube, d_cube, mask_cube = check_ray_intersection_with_unit_cube(
        camera_world, ray_vector, padding=padding, eps=eps)
    if not use_ray_length_as_depth:
        p_cam = transform_to_camera_space(p_cube.view(
            batch_size, -1, 3), cameras).view(
                batch_size, n_points, -1, 3)
        d_cube = p_cam[:, :, :, -1]
    # if d_cube <= 0 (or cameras's image plane?) then set mask_cube to False
    mask_cube[torch.any(d_cube <= 0, dim=-1)] = False
    check_tensor(p_cube, 'p_cube')
    check_tensor(d_cube, 'd_cube')
    return p_cube, d_cube, mask_cube


def arange_pixels(resolution=(128, 128), batch_size=1, image_range=(-1., 1.),
                  subsample_to=None):
    ''' Arranges pixels for given resolution in range image_range.

    The function returns the unscaled pixel locations as integers and the
    scaled float values.

    Args:
        resolution (tuple): image resolution
        batch_size (int): batch size
        image_range (tuple): range of output points (default [-1, 1])
        subsample_to (int): if integer and > 0, the points are randomly
            subsampled to this value
    '''
    h, w = resolution
    n_points = resolution[0] * resolution[1]

    # Arrange pixel location in scale resolution
    pixel_locations = torch.meshgrid(torch.arange(0, w), torch.arange(0, h))
    pixel_locations = torch.stack(
        [pixel_locations[0], pixel_locations[1]],
        dim=-1).long().view(1, -1, 2).repeat(batch_size, 1, 1)
    pixel_scaled = pixel_locations.clone().float()

    # Shift and scale points to match image_range
    scale = (image_range[1] - image_range[0])
    loc = scale / 2
    pixel_scaled[:, :, 0] = scale * pixel_scaled[:, :, 0] / (w - 1) - loc
    pixel_scaled[:, :, 1] = scale * pixel_scaled[:, :, 1] / (h - 1) - loc

    # Subsample points if subsample_to is not None and > 0
    if (subsample_to is not None and subsample_to > 0 and
            subsample_to < n_points):
        idx = np.random.choice(pixel_scaled.shape[1], size=(subsample_to,),
                               replace=False)
        pixel_scaled = pixel_scaled[:, idx]
        pixel_locations = pixel_locations[:, idx]

    return pixel_locations, pixel_scaled


def to_pytorch(tensor, return_type=False):
    ''' Converts input tensor to pytorch.

    Args:
        tensor (tensor): Numpy or Pytorch tensor
        return_type (bool): whether to return input type
    '''
    is_numpy = False
    if type(tensor) == np.ndarray:
        tensor = torch.from_numpy(tensor)
        is_numpy = True
    tensor = tensor.clone()
    if return_type:
        return tensor, is_numpy
    return tensor


def get_mask(tensor):
    ''' Returns mask of non-illegal values for tensor.

    Args:
        tensor (tensor): Numpy or Pytorch tensor
    '''
    tensor, is_numpy = to_pytorch(tensor, True)
    mask = ((abs(tensor) != np.inf) & (torch.isnan(tensor) == False))
    mask = mask.bool()
    if is_numpy:
        mask = mask.numpy()

    return mask


def transform_mesh(mesh, transform):
    ''' Transforms a mesh with given transformation.

    Args:
        mesh (trimesh mesh): mesh
        transform (tensor): transformation matrix of size 4 x 4
    '''
    mesh = deepcopy(mesh)
    v = np.asarray(mesh.vertices).astype(np.float32)
    v_transformed = transform_pointcloud(v, transform)
    mesh.vertices = v_transformed
    return mesh


def transform_pointcloud(pointcloud, transform):
    ''' Transforms a point cloud with given transformation.

    Args:
        pointcloud (tensor): tensor of size N x 3
        transform (tensor): transformation of size 4 x 4
    '''

    assert(transform.shape == (4, 4) and pointcloud.shape[-1] == 3)

    pcl, is_numpy = to_pytorch(pointcloud, True)
    transform = to_pytorch(transform)

    # Transform point cloud to homogen coordinate system
    pcl_hom = torch.cat([
        pcl, torch.ones(pcl.shape[0], 1)
    ], dim=-1).transpose(1, 0)

    # Apply transformation to point cloud
    pcl_hom_transformed = transform @ pcl_hom

    # Transform back to 3D coordinates
    pcl_out = pcl_hom_transformed[:3].transpose(1, 0)
    if is_numpy:
        pcl_out = pcl_out.numpy()

    return pcl_out


def transform_points_batch(p, transform):
    ''' Transform points tensor with given transform.

    Args:
        p (tensor): tensor of size B x N x 3
        transform (tensor): transformation of size B x 4 x 4
    '''
    device = p.device
    assert(transform.shape[1:] == (4, 4) and p.shape[-1]
           == 3 and p.shape[0] == transform.shape[0])

    # Transform points to homogen coordinates
    pcl_hom = torch.cat([
        p, torch.ones(p.shape[0], p.shape[1], 1).to(device)
    ], dim=-1).transpose(2, 1)

    # Apply transformation
    pcl_hom_transformed = transform @ pcl_hom

    # Transform back to 3D coordinates
    pcl_out = pcl_hom_transformed[:, :3].transpose(2, 1)
    return pcl_out


def get_tensor_values(tensor, p, grid_sample=True, mode='nearest',
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
    p = to_pytorch(p)
    tensor, is_numpy = to_pytorch(tensor, True)
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
        mask = get_mask(values)
        if squeeze_channel_dim:
            mask = mask.squeeze(-1)
        if is_numpy:
            mask = mask.numpy()

    if squeeze_channel_dim:
        values = values.squeeze(-1)

    if is_numpy:
        values = values.numpy()

    if with_mask:
        return values, mask
    return values


def transform_to_world(pixels, depth, cameras):
    ''' Transforms pixel positions p with given depth value d to world coordinates.
    NOTE: assume positions p is increasing from left to right, and top to down (left-hand system).
    The NDC system used by pytorch3d is reversed (right-hand system).
    Args:
        pixels (tensor): pixel tensor of size B x N x 2
        depth (tensor): depth tensor of size B x N x 1
        invert (bool): whether to invert matrices (default: true)
    '''
    assert(pixels.shape[-1] == 2)

    # Convert to pytorch
    pixels, is_numpy = to_pytorch(pixels, True)

    # NOTE: negate pixels because pytorch3d NDC have x and y directions inversed
    xy_depth = torch.cat([-pixels, depth], dim=-1)
    p_world = cameras.unproject_points(xy_depth, eps=1e-8)
    check_tensor(p_world, 'p_world')
    if is_numpy:
        p_world = p_world.numpy()
    return p_world


def transform_to_camera_space(p_world, cameras):
    ''' Transforms world points to camera space.
        Args:
        p_world (tensor): world points tensor of size B x N x 3
    '''
    batch_size, n_p, _ = p_world.shape
    device = p_world.device

    world_to_view_trans = cameras.get_world_to_view_transform()
    p_cam = world_to_view_trans.transform_points(p_world)
    return p_cam


def origin_to_world(n_points, cameras):
    ''' Transforms origin (camera location) to world coordinates.

    Args:
        n_points (int): how often the transformed origin is repeated in the
            form (batch_size, n_points, 3)
    '''
    batch_size = cameras.R.shape[0]
    p_world = cameras.get_camera_center()
    p_world = p_world.view(batch_size, 1, 3).expand(-1, n_points, -1)
    return p_world


def image_points_to_world(image_points, cameras):
    ''' Transforms points on image plane to world coordinates.

    In contrast to transform_to_world, no depth value is needed as points on
    the image plane have a fixed depth of 1.

    Args:
        image_points (tensor): image points tensor of size B x N x 2
        invert (bool): whether to invert matrices (default: true)
    '''
    batch_size, n_pts, dim = image_points.shape
    assert(dim == 2)
    device = image_points.device
    image_plane_z = torch.tensor((1, ), device=device)
    try:
        image_plane_z = cameras.znear
    except AttributeError as e:
        image_plane_z = cameras.focal_length
    except Exception:
        logger_py.error('Couldn\'t figure out the image plane from the cameras instance. ' +
                        'Make sure you are using pytorch3d cameras')

    image_plane_z = image_plane_z.view(-1, 1, 1).expand(batch_size, n_pts, 1)
    return transform_to_world(image_points, image_plane_z, cameras)


def check_weights(params):
    ''' Checks weights for illegal values.

    Args:
        params (tensor): parameter tensor
    '''
    for k, v in params.items():
        if torch.isnan(v).any():
            logger_py.warn('NaN Values detected in model weight %s.' % k)


def check_tensor(tensor, tensorname='', input_tensor=None):
    ''' Checks tensor for illegal values.

    Args:
        tensor (tensor): tensor
        tensorname (string): name of tensor
        input_tensor (tensor): previous input
    '''
    if torch.isnan(tensor).any():
        logger_py.warn('Tensor %s contains nan values.' % tensorname)
        if input_tensor is not None:
            logger_py.warning('Input was:', input_tensor)
        import pdb
        pdb.set_trace()
    if not torch.isfinite(tensor).all():
        logger_py.warn('Tensor %s contains infinite values.' % tensorname)
        if input_tensor is not None:
            logger_py.warn('Input was:', input_tensor)
        import pdb
        pdb.set_trace()


def get_prob_from_logits(logits):
    ''' Returns probabilities for logits

    Args:
        logits (tensor): logits
    '''
    odds = np.exp(logits)
    probs = odds / (1 + odds)
    return probs


def get_logits_from_prob(probs, eps=1e-4):
    ''' Returns logits for probabilities.

    Args:
        probs (tensor): probability tensor
        eps (float): epsilon value for numerical stability
    '''
    probs = np.clip(probs, a_min=eps, a_max=1 - eps)
    logits = np.log(probs / (1 - probs))
    return logits


def compute_iou(occ1, occ2):
    ''' Computes the Intersection over Union (IoU) value for two sets of
    occupancy values.

    Args:
        occ1 (tensor): first set of occupancy values
        occ2 (tensor): second set of occupancy values
    '''
    occ1 = np.asarray(occ1)
    occ2 = np.asarray(occ2)

    # Put all data in second dimension
    # Also works for 1-dimensional data
    if occ1.ndim >= 2:
        occ1 = occ1.reshape(occ1.shape[0], -1)
    if occ2.ndim >= 2:
        occ2 = occ2.reshape(occ2.shape[0], -1)

    # Convert to boolean values
    occ1 = (occ1 >= 0.5)
    occ2 = (occ2 >= 0.5)

    # Compute IOU
    area_union = (occ1 | occ2).astype(np.float32).sum(axis=-1)
    area_intersect = (occ1 & occ2).astype(np.float32).sum(axis=-1)

    iou = (area_intersect / area_union)

    return iou


# def chamfer_distance(points1, points2, use_kdtree=True, give_id=False):
#     ''' Returns the chamfer distance for the sets of points.
# 
#     Args:
#         points1 (numpy array): first point set
#         points2 (numpy array): second point set
#         use_kdtree (bool): whether to use a kdtree
#         give_id (bool): whether to return the IDs of nearest points
#     '''
#     if use_kdtree:
#         return chamfer_distance_kdtree(points1, points2, give_id=give_id)
#     else:
#         return chamfer_distance_naive(points1, points2)


# def chamfer_distance_naive(points1, points2):
#     ''' Naive implementation of the Chamfer distance.
# 
#     Args:
#         points1 (numpy array): first point set
#         points2 (numpy array): second point set
#     '''
#     assert(points1.size() == points2.size())
#     batch_size, T, _ = points1.size()
# 
#     points1 = points1.view(batch_size, T, 1, 3)
#     points2 = points2.view(batch_size, 1, T, 3)
# 
#     distances = (points1 - points2).pow(2).sum(-1)
# 
#     chamfer1 = distances.min(dim=1)[0].mean(dim=1)
#     chamfer2 = distances.min(dim=2)[0].mean(dim=1)
# 
#     chamfer = chamfer1 + chamfer2
#     return chamfer


# def chamfer_distance_kdtree(points1, points2, give_id=False):
#     ''' KD-tree based implementation of the Chamfer distance.
# 
#     Args:
#         points1 (numpy array): first point set
#         points2 (numpy array): second point set
#         give_id (bool): whether to return the IDs of the nearest points
#     '''
#     # Points have size batch_size x T x 3
#     batch_size = points1.size(0)
# 
#     # First convert points to numpy
#     points1_np = points1.detach().cpu().numpy()
#     points2_np = points2.detach().cpu().numpy()
# 
#     # Get list of nearest neighbors indices
#     idx_nn_12, _ = get_nearest_neighbors_indices_batch(points1_np, points2_np)
#     idx_nn_12 = torch.LongTensor(idx_nn_12).to(points1.device)
#     # Expands it as batch_size x 1 x 3
#     idx_nn_12_expand = idx_nn_12.view(batch_size, -1, 1).expand_as(points1)
# 
#     # Get list of nearest neighbors indices
#     idx_nn_21, _ = get_nearest_neighbors_indices_batch(points2_np, points1_np)
#     idx_nn_21 = torch.LongTensor(idx_nn_21).to(points1.device)
#     # Expands it as batch_size x T x 3
#     idx_nn_21_expand = idx_nn_21.view(batch_size, -1, 1).expand_as(points2)
# 
#     # Compute nearest neighbors in points2 to points in points1
#     # points_12[i, j, k] = points2[i, idx_nn_12_expand[i, j, k], k]
#     points_12 = torch.gather(points2, dim=1, index=idx_nn_12_expand)
# 
#     # Compute nearest neighbors in points1 to points in points2
#     # points_21[i, j, k] = points2[i, idx_nn_21_expand[i, j, k], k]
#     points_21 = torch.gather(points1, dim=1, index=idx_nn_21_expand)
# 
#     # Compute chamfer distance
#     chamfer1 = (points1 - points_12).pow(2).sum(2).mean(1)
#     chamfer2 = (points2 - points_21).pow(2).sum(2).mean(1)
# 
#     # Take sum
#     chamfer = chamfer1 + chamfer2
# 
#     # If required, also return nearest neighbors
#     if give_id:
#         return chamfer1, chamfer2, idx_nn_12, idx_nn_21
# 
#     return chamfer


# def get_nearest_neighbors_indices_batch(points_src, points_tgt, k=1):
#     ''' Returns the nearest neighbors for point sets batchwise.
# 
#     Args:
#         points_src (numpy array): source points
#         points_tgt (numpy array): target points
#         k (int): number of nearest neighbors to return
#     '''
#     indices = []
#     distances = []
# 
#     for (p1, p2) in zip(points_src, points_tgt):
#         kdtree = KDTree(p2)
#         dist, idx = kdtree.query(p1, k=k)
#         indices.append(idx)
#         distances.append(dist)
# 
#     return indices, distances


def normalize_imagenet(x):
    ''' Normalize input images according to ImageNet standards.

    Args:
        x (tensor): input images
    '''
    x = x.clone()
    x[:, 0] = (x[:, 0] - 0.485) / 0.229
    x[:, 1] = (x[:, 1] - 0.456) / 0.224
    x[:, 2] = (x[:, 2] - 0.406) / 0.225
    return x


def make_3d_grid(bb_min, bb_max, shape):
    ''' Makes a 3D grid.

    Args:
        bb_min (tuple): bounding box minimum
        bb_max (tuple): bounding box maximum
        shape (tuple): output shape
    '''
    size = shape[0] * shape[1] * shape[2]

    pxs = torch.linspace(bb_min[0], bb_max[0], shape[0])
    pys = torch.linspace(bb_min[1], bb_max[1], shape[1])
    pzs = torch.linspace(bb_min[2], bb_max[2], shape[2])

    pxs = pxs.view(-1, 1, 1).expand(*shape).contiguous().view(size)
    pys = pys.view(1, -1, 1).expand(*shape).contiguous().view(size)
    pzs = pzs.view(1, 1, -1).expand(*shape).contiguous().view(size)
    p = torch.stack([pxs, pys, pzs], dim=1)

    return p


def get_occupancy_loss_points(pixels, cameras,
                              depth_image=None, use_cube_intersection=True,
                              occupancy_random_normal=False,
                              depth_range=[1.0, 2.4]):
    ''' Returns 3D points for occupancy loss.
    Penalize pixels that lie inside the object mask but the predicted surface depth is inifite.
    Use randomly sampled depth value, or in case the depth_image is give, use the ground truth
    depth to backproject the pixel in space.
    NOTE: differs to freespace_loss_points in the case when depth_image is given.
    Args:
        pixels (tensor): (N, 3) sampled pixels in range [-1, 1]
        cameras (camera object)
        depth_image tensor): if not None, these depth values are used for
            initialization (e.g. depth or visual hull depth)
        use_cube_intersection (bool): whether to check unit cube intersection
        occupancy_random_normal (bool): whether to sample from a Normal
            distribution instead of a uniform one
        depth_range (float): depth range; important when no cube
            intersection is used
    '''
    device = pixels.device
    batch_size, n_points, _ = pixels.shape

    # avoid zero depth
    d_occupancy = torch.rand(batch_size, n_points).to(device) * \
        (depth_range[1] - depth_range[0]) + depth_range[0]

    if use_cube_intersection:
        _, d_cube_intersection, mask_cube = \
            intersect_camera_rays_with_unit_cube(
                pixels, cameras, padding=0.,
                use_ray_length_as_depth=False)
        # (BM,2)
        d_cube = d_cube_intersection[mask_cube]
        # use a random depth between the two intersections with the unit cube
        d_occupancy[mask_cube] = d_cube[:, 0] + \
            torch.rand(d_cube.shape[0]).to(
                device) * (d_cube[:, 1] - d_cube[:, 0])

    if occupancy_random_normal:
        d_occupancy = torch.randn(batch_size, n_points).to(device) \
            * (depth_range[1] / 8) + depth_range[1] / 2
        if use_cube_intersection:
            mean_cube = d_cube.sum(-1) / 2
            std_cube = (d_cube[:, 1] - d_cube[:, 0]) / 8
            d_occupancy[mask_cube] = mean_cube + \
                torch.randn(mean_cube.shape[0]).to(device) * std_cube

    if depth_image is not None:
        depth_gt, mask_gt_depth = get_tensor_values(
            depth_image, pixels, squeeze_channel_dim=True, with_mask=True)
        d_occupancy[mask_gt_depth] = depth_gt[mask_gt_depth]

    p_occupancy = transform_to_world(
        pixels, d_occupancy.unsqueeze(-1), cameras)
    return p_occupancy


def get_freespace_loss_points(pixels, cameras,
                              use_cube_intersection=True, depth_range=[1.0, 2.4]):
    ''' Returns 3D points for freespace loss.

    Args:
        pixels (tensor): (B,P,2) sampled pixels in range [-1, 1]
        use_cube_intersection (bool): whether to check unit cube intersection
        depth_range (float): depth range; important when no cube
            intersection is used
    '''
    device = pixels.device
    batch_size, n_points, _ = pixels.shape

    # sample between the depth range. avoid 0 depth
    d_freespace = torch.rand(batch_size, n_points).to(device) * \
        (depth_range[1] - depth_range[0]) + depth_range[0]

    if use_cube_intersection:
        # d_freespace is a random depth between the two intersections with the unit cube
        _, d_cube_intersection, mask_cube = \
            intersect_camera_rays_with_unit_cube(
                pixels, cameras,
                use_ray_length_as_depth=False)
        d_cube = d_cube_intersection[mask_cube]
        d_freespace[mask_cube] = d_cube[:, 0] + \
            torch.rand(d_cube.shape[0]).to(
                device) * (d_cube[:, 1] - d_cube[:, 0])

    p_freespace = transform_to_world(
        pixels, d_freespace.unsqueeze(-1), cameras)
    return p_freespace


def normalize_tensor(tensor, min_norm=1e-5, feat_dim=-1):
    ''' Normalizes the tensor.

    Args:
        tensor (tensor): tensor
        min_norm (float): minimum norm for numerical stability
        feat_dim (int): feature dimension in tensor (default: -1)
    '''
    norm_tensor = torch.clamp(torch.norm(tensor, dim=feat_dim, keepdim=True),
                              min=min_norm)
    normed_tensor = tensor / norm_tensor
    return normed_tensor


def get_input_pc(data_dict):
    """ For 2.5D inputs visualize the initial pointcloud from backprojection

    Returns the point cloud in object coordinate
    """
    def process_data_dict(data):
        ''' Processes the data dictionary and returns respective tensors

        Args:
            data (dictionary): data dictionary of numpy arrays
        '''
        # Get "ordinary" data
        img = to_pytorch(data.get('img')).unsqueeze(0)
        mask_img = to_pytorch(data.get('img.mask')).unsqueeze(0).unsqueeze(1)
        world_mat = to_pytorch(data.get('img.world_mat')).unsqueeze(0)
        camera_mat = to_pytorch(data.get('img.camera_mat')).unsqueeze(0)
        scale_mat = to_pytorch(data.get('img.scale_mat')).unsqueeze(0)
        depth_img = to_pytorch(data.get('img.depth', torch.empty(1, 0)
                                        )).unsqueeze(0).unsqueeze(1)
        inputs = data.get('inputs', torch.empty(1, 0)).unsqueeze(0)

        # Get sparse point data
        if 'sparse_depth.p_img' in data:
            sparse_depth = {}
            sparse_depth['p'] = to_pytorch(
                data.get('sparse_depth.p_img')).unsqueeze(0)
            sparse_depth['p_world'] = to_pytorch(data.get(
                'sparse_depth.p_world')).unsqueeze(0)
            sparse_depth['depth_gt'] = to_pytorch(
                data.get('sparse_depth.d')).unsqueeze(0)
            sparse_depth['camera_mat'] = to_pytorch(data.get(
                'sparse_depth.camera_mat')).unsqueeze(0)
            sparse_depth['world_mat'] = to_pytorch(data.get(
                'sparse_depth.world_mat')).unsqueeze(0)
            sparse_depth['scale_mat'] = to_pytorch(data.get(
                'sparse_depth.scale_mat')).unsqueeze(0)
        else:
            sparse_depth = None

        return (img, mask_img, depth_img, world_mat, camera_mat, scale_mat,
                inputs, sparse_depth)

    # load dataset 2.5D images
    (img, mask_img, depth_img, world_mat, camera_mat, scale_mat,
     inputs, sparse_depth) = process_data_dict(data_dict)

    if depth_img.shape[-1] == 0 and sparse_depth is None:
        raise ValueError("Dataset does not contain depth information.")

    batch_size, _, h, w = img.shape

    if depth_img.shape[-1] > 0:
        # pixel (B,N,2)
        pixels = arange_pixels((h, w), batch_size)[1].to(img.device)
        assert(pixels.shape[1] == h * w)
        # if annotated mask exists, use this mask to remove outlier points
        mask_gt = get_tensor_values(
            mask_img, pixels, squeeze_channel_dim=True).bool()

        # transform to world coordinate
        depth_gt, mask_gt_depth = get_tensor_values(
            depth_img, pixels, squeeze_channel_dim=True, with_mask=True)

        mask = mask_gt_depth & mask_gt
        p_world = transform_to_world(pixels, depth_gt.unsqueeze(-1), cameras)
        # get color
        rgb_gt = get_tensor_values(img, pixels, with_mask=False)
        dense_points = torch.cat([p_world[mask], rgb_gt[mask]], dim=-1)
    else:
        dense_points = None

    # if sparse_depth is available
    if sparse_depth is not None:
        sparse_points = sparse_depth['p_world'].squeeze(0)
    else:
        sparse_points = None

    return dense_points, sparse_points