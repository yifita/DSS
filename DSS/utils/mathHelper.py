from typing import Union, Tuple, Optional
import numpy as np
import torch
from torch_batch_svd import svd as batch_svd
from pytorch3d.ops.utils import convert_pointclouds_to_tensor
from pytorch3d.ops import knn_points
from pytorch3d.ops.points_normals import _disambiguate_vector_directions


def eps_denom(denom, eps=1e-17):
    """ Prepare denominator for division """
    denom_sign = denom.sign() + (denom == 0.0).type_as(denom)
    denom = denom_sign * torch.clamp(denom.abs(), eps)
    return denom

def eps_sqrt(squared, eps=1e-17):
    """
    Prepare for the input for sqrt, make sure the input positive and
    larger than eps
    """
    return torch.clamp(squared.abs(), eps)


def pinverse(inputs: torch.Tensor):
    assert(inputs.ndim >= 2)
    shp = inputs.shape
    U, S, V = batch_svd(inputs.view(-1, shp[-2], shp[-1]))
    S[S < 1e-6] = 0
    S_inv = torch.where(S < 1e-5, torch.zeros_like(S), 1/S)
    pinv = V @ torch.diag_embed(S_inv) @ U.transpose(1,2)
    return pinv.view(shp)


def estimate_pointcloud_local_coord_frames(
    pointclouds: Union[torch.Tensor, "Pointclouds"],
    neighborhood_size: int = 50,
    disambiguate_directions: bool = True,
    return_knn_result: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional['KNN']]:
    """
    Faster version of pytorch3d estimate_pointcloud_local_coord_frames

    Estimates the principal directions of curvature (which includes normals)
    of a batch of `pointclouds`.
    Returns:
        curvatures (N,P,3) ascending order
        local_frames (N,P,3,3) corresponding eigenvectors
    """
    points_padded, num_points = convert_pointclouds_to_tensor(pointclouds)

    ba, N, dim = points_padded.shape
    if dim != 3:
        raise ValueError(
            "The pointclouds argument has to be of shape (minibatch, N, 3)"
        )

    if (num_points <= neighborhood_size).any():
        raise ValueError(
            "The neighborhood_size argument has to be"
            + " >= size of each of the point clouds."
        )
    # undo global mean for stability
    # TODO: replace with tutil.wmean once landed
    pcl_mean = points_padded.sum(1) / num_points[:, None]
    points_centered = points_padded - pcl_mean[:, None, :]

    # get K nearest neighbor idx for each point in the point cloud
    knn_result = knn_points(
        points_padded,
        points_padded,
        lengths1=num_points,
        lengths2=num_points,
        K=neighborhood_size,
        return_nn=True,
    )
    k_nearest_neighbors = knn_result.knn
    # obtain the mean of the neighborhood
    pt_mean = k_nearest_neighbors.mean(2, keepdim=True)
    # compute the diff of the neighborhood and the mean of the neighborhood
    # N,P,K,3
    central_diff = k_nearest_neighbors - pt_mean
    per_pts_diff = central_diff.view(-1, neighborhood_size, 3)
    # S (NP,3) and local_coord_framds (NP,3,3)
    _, S, local_coord_frames = batch_svd(per_pts_diff)
    curvature = S * S / neighborhood_size
    local_coord_frames = local_coord_frames.view(ba, N, dim, dim)
    curvature = curvature.view(ba, N, dim)

    # flip to ascending order
    curvature = curvature.flip(-1)
    local_coord_frames = local_coord_frames.flip(-1)

    # disambiguate the directions of individual principal vectors
    if disambiguate_directions:
        # disambiguate normal
        n = _disambiguate_vector_directions(
            points_centered, k_nearest_neighbors, local_coord_frames[:, :, :, 0]
        )
        # disambiguate the main curvature
        z = _disambiguate_vector_directions(
            points_centered, k_nearest_neighbors, local_coord_frames[:, :, :, 2]
        )
        # the secondary curvature is just a cross between n and z
        y = torch.cross(n, z, dim=2)
        # cat to form the set of principal directions
        local_coord_frames = torch.stack((n, y, z), dim=3)

    if return_knn_result:
        return curvature, local_coord_frames, knn_result
    return curvature, local_coord_frames


def estimate_pointcloud_normals(
    pointclouds: Union[torch.Tensor, "Pointclouds"],
    neighborhood_size: int = 50,
    disambiguate_directions: bool = True,
) -> torch.Tensor:
    """
    Estimates the normals of a batch of `pointclouds` using fast `estimate_pointcloud_local_coord_frames

    Args:
      **pointclouds**: Batch of 3-dimensional points of shape
        `(minibatch, num_point, 3)` or a `Pointclouds` object.
      **neighborhood_size**: The size of the neighborhood used to estimate the
        geometry around each point.
      **disambiguate_directions**: If `True`, uses the algorithm from [1] to
        ensure sign consistency of the normals of neigboring points.

    Returns:
      **normals**: A tensor of normals for each input point
        of shape `(minibatch, num_point, 3)`.
        If `pointclouds` are of `Pointclouds` class, returns a padded tensor.

    References:
      [1] Tombari, Salti, Di Stefano: Unique Signatures of Histograms for
      Local Surface Description, ECCV 2010.
    """
    curvatures, local_coord_frames = estimate_pointcloud_local_coord_frames(
        pointclouds,
        neighborhood_size=neighborhood_size,
        disambiguate_directions=disambiguate_directions,
    )

    # the normals correspond to the first vector of each local coord frame
    normals = local_coord_frames[:, :, :, 0]

    return normals


def ndc_to_pix(p, resolution):
    """
    Reverse of pytorch3d pix_to_ndc function
    Args:
        p (float tensor): (..., 3)
        resolution (scalar): image resolution (for now, supports only aspectratio = 1)
    Returns:
        pix (long tensor): (..., 2)
    """
    pix = resolution - ((p[..., :2] + 1.0) * resolution - 1.0) / 2
    return pix


def decompose_to_R_and_t(transform_mat, row_major=True):
    """ decompose a 4x4 transform matrix to R (3,3) and t (1,3)"""
    assert(transform_mat.shape[-2:] == (4, 4)), \
        "Expecting batches of 4x4 matrice"
    # ... 3x3
    if not row_major:
        transform_mat = transform_mat.transpose(-2, -1)

    R = transform_mat[..., :3, :3]
    t = transform_mat[..., -1, :3]

    return R, t


def to_homogen(x, dim=-1):
    """ append one to the specified dimension """
    if dim < 0:
        dim = x.ndim + dim
    shp = x.shape
    new_shp = shp[:dim] + (1, ) + shp[dim + 1:]
    x_homogen = x.new_ones(new_shp)
    x_homogen = torch.cat([x, x_homogen], dim=dim)
    return x_homogen
