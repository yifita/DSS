""" Utility functions for processing point clouds.
"""
import os
import numpy as np

# Point cloud IO
from matplotlib import cm
import plyfile


def normalize_point_cloud(input):
    """
    input: pc [N, P, 3]
    output: pc, centroid, furthest_distance
    """
    if len(input.shape) == 2:
        axis = 0
    elif len(input.shape) == 3:
        axis = 1
    centroid = np.mean(input, axis=axis, keepdims=True)
    input = input - centroid
    furthest_distance = np.amax(
        np.sqrt(np.sum(input ** 2, axis=-1, keepdims=True)), axis=axis, keepdims=True)
    input = input / furthest_distance
    return input, centroid, furthest_distance


def jitter_perturbation_point_cloud(batch_data, sigma=0.005, clip=0.02, is_2D=False):
    """
    Randomly jitter points. jittering is per point.
    Input:
        batch_data: BxNx3 array, original batch of point clouds
    Return:
        BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    chn = 2 if is_2D else 3
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -clip, clip).astype(dtype=batch_data.dtype)
    jittered_data[:, :, chn:] = 0
    jittered_data += batch_data
    return jittered_data


def rotate_point_cloud_and_gt(batch_data, batch_gt=None):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    for k in range(batch_data.shape[0]):
        angles = np.random.uniform(size=(3)) * 2 * np.pi
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]], dtype=batch_data.dtype)
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]], dtype=batch_data.dtype)
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]], dtype=batch_data.dtype)
        rotation_matrix = np.dot(Rz, np.dot(Ry, Rx))

        batch_data[k, ..., 0:3] = np.dot(
            batch_data[k, ..., 0:3].reshape((-1, 3)), rotation_matrix)
        if batch_data.shape[-1] > 3:
            batch_data[k, ..., 3:] = np.dot(
                batch_data[k, ..., 3:].reshape((-1, 3)), rotation_matrix)

        if batch_gt is not None:
            batch_gt[k, ..., 0:3] = np.dot(
                batch_gt[k, ..., 0:3].reshape((-1, 3)), rotation_matrix)
            if batch_gt.shape[-1] > 3:
                batch_gt[k, ..., 3:] = np.dot(
                    batch_gt[k, ..., 3:].reshape((-1, 3)), rotation_matrix)

    return batch_data, batch_gt


def random_scale_point_cloud_and_gt(batch_data, batch_gt=None, scale_low=0.5, scale_high=2):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, (B, 1, 1)).astype(batch_data.dtype)

    batch_data = np.concatenate([batch_data[:, :, :3] * scales, batch_data[:, :, 3:]], axis=-1)

    if batch_gt is not None:
        batch_gt = np.concatenate([batch_gt[:, :, :3] * scales, batch_gt[:, :, 3:]], axis=-1)

    return batch_data, batch_gt, np.squeeze(scales)


def downsample_points(pts, K):
    # if num_pts > 8K use farthest sampling
    # else use random sampling
    if pts.shape[0] >= 2 * K:
        sampler = FarthestSampler()
        return sampler(pts, K)
    else:
        return pts[np.random.choice(pts.shape[0], K,
                                    replace=(K < pts.shape[0])), :]


class FarthestSampler:
    def __init__(self):
        pass

    def _calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=1)

    def __call__(self, pts, k):
        farthest_pts = np.zeros((k, pts.shape[1]), dtype=np.float32)
        farthest_pts[0] = pts[np.random.randint(len(pts))]
        distances = self._calc_distances(farthest_pts[0, :3], pts[:, :3])
        for i in range(1, k):
            farthest_pts[i] = pts[np.argmax(distances)]
            distances = np.minimum(
                distances, self._calc_distances(farthest_pts[i, :3], pts[:, :3]))
        return farthest_pts


def read_ply_with_color(file, count=None):
    loaded = plyfile.PlyData.read(file)
    points = np.vstack([loaded['vertex'].data['x'],
                        loaded['vertex'].data['y'], loaded['vertex'].data['z']])
    if 'nx' in loaded['vertex'].data.dtype.names:
        normals = np.vstack([loaded['vertex'].data['nx'],
                             loaded['vertex'].data['ny'], loaded['vertex'].data['nz']])
        points = np.concatenate([points, normals], axis=0)
    colors = None
    if 'red' in loaded['vertex'].data.dtype.names:
        colors = np.vstack([loaded['vertex'].data['red'],
                            loaded['vertex'].data['green'], loaded['vertex'].data['blue']])
        if 'alpha' in loaded['vertex'].data.dtype.names:
            colors = np.concatenate([colors, np.expand_dims(
                loaded['vertex'].data['alpha'], axis=0)], axis=0)
        colors = colors.transpose(1, 0)
        colors = colors.astype(np.float32) / 255.0

    points = points.transpose(1, 0)
    if count is not None:
        if count > points.shape[0]:
            # fill the point clouds with the random point
            tmp = np.zeros((count, points.shape[1]), dtype=points.dtype)
            tmp[:points.shape[0], ...] = points
            tmp[points.shape[0]:, ...] = points[np.random.choice(
                points.shape[0], count - points.shape[0]), :]
            points = tmp
        elif count < points.shape[0]:
            # different to pointnet2, take random x point instead of the first
            # idx = np.random.permutation(count)
            # points = points[idx, :]
            points = downsample_points(points, count)
    return points, colors


def read_ply(file, count=None):
    loaded = plyfile.PlyData.read(file)
    points = np.vstack([loaded['vertex'].data['x'],
                        loaded['vertex'].data['y'], loaded['vertex'].data['z']])
    if 'nx' in loaded['vertex'].data.dtype.names:
        normals = np.vstack([loaded['vertex'].data['nx'],
                             loaded['vertex'].data['ny'], loaded['vertex'].data['nz']])
        points = np.concatenate([points, normals], axis=0)

    points = points.transpose(1, 0)
    if count is not None:
        if count > points.shape[0]:
            # fill the point clouds with the random point
            tmp = np.zeros((count, points.shape[1]), dtype=points.dtype)
            tmp[:points.shape[0], ...] = points
            tmp[points.shape[0]:, ...] = points[np.random.choice(
                points.shape[0], count - points.shape[0]), :]
            points = tmp
        elif count < points.shape[0]:
            # different to pointnet2, take random x point instead of the first
            # idx = np.random.permutation(count)
            # points = points[idx, :]
            points = downsample_points(points, count)
    return points


def save_ply_with_face_property(points, faces, property, property_max, filename, cmap_name="Set1"):
    face_num = faces.shape[0]
    colors = np.full(faces.shape, 0.5)
    cmap = cm.get_cmap(cmap_name)
    for point_idx in range(face_num):
        colors[point_idx] = cmap(property[point_idx] / property_max)[:3]
    save_ply_with_face(points, faces, filename, colors)


def save_ply_with_face(points, faces, filename, colors=None):
    vertex = np.array([tuple(p) for p in points], dtype=[
                      ('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    faces = np.array([(tuple(p),) for p in faces], dtype=[
                     ('vertex_indices', 'i4', (3, ))])
    descr = faces.dtype.descr
    if colors is not None:
        assert len(colors) == len(faces)
        face_colors = np.array([tuple(c * 255) for c in colors],
                               dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        descr = faces.dtype.descr + face_colors.dtype.descr

    faces_all = np.empty(len(faces), dtype=descr)
    for prop in faces.dtype.names:
        faces_all[prop] = faces[prop]
    if colors is not None:
        for prop in face_colors.dtype.names:
            faces_all[prop] = face_colors[prop]

    ply = plyfile.PlyData([plyfile.PlyElement.describe(
        vertex, 'vertex'), plyfile.PlyElement.describe(faces_all, 'face')], text=False)
    ply.write(filename)


def load(filename, count=None):
    if filename[-4:] == ".ply":
        points = read_ply(filename, count).astype(np.float32)
    else:
        points = np.loadtxt(filename).astype(np.float32)
        if count is not None:
            if count > points.shape[0]:
                # fill the point clouds with the random point
                tmp = np.zeros((count, points.shape[1]), dtype=points.dtype)
                tmp[:points.shape[0], ...] = points
                tmp[points.shape[0]:, ...] = points[np.random.choice(
                    points.shape[0], count - points.shape[0]), :]
                points = tmp
            elif count < points.shape[0]:
                # different to pointnet2, take random x point instead of the first
                # idx = np.random.permutation(count)
                # points = points[idx, :]
                points = downsample_points(points, count)
    return points


def save_ply(points, filename, colors=None, normals=None):
    vertex = np.core.records.fromarrays(points.transpose(
        1, 0), names='x, y, z', formats='f4, f4, f4')
    num_vertex = len(vertex)
    desc = vertex.dtype.descr

    if normals is not None:
        vertex_normal = np.core.records.fromarrays(
            normals.transpose(1, 0), names='nx, ny, nz', formats='f4, f4, f4')
        assert len(vertex_normal) == num_vertex
        desc = desc + vertex_normal.dtype.descr

    if colors is not None:
        assert len(colors) == num_vertex
        if colors.max() <= 1:
            colors = colors * 255
        if colors.shape[1] == 4:
            vertex_color = np.core.records.fromarrays(colors.transpose(
                1, 0), names='red, green, blue, alpha', formats='u1, u1, u1, u1')
        else:
            vertex_color = np.core.records.fromarrays(colors.transpose(
                1, 0), names='red, green, blue', formats='u1, u1, u1')
        desc = desc + vertex_color.dtype.descr

    vertex_all = np.empty(num_vertex, dtype=desc)

    for prop in vertex.dtype.names:
        vertex_all[prop] = vertex[prop]

    if normals is not None:
        for prop in vertex_normal.dtype.names:
            vertex_all[prop] = vertex_normal[prop]

    if colors is not None:
        for prop in vertex_color.dtype.names:
            vertex_all[prop] = vertex_color[prop]

    ply = plyfile.PlyData(
        [plyfile.PlyElement.describe(vertex_all, 'vertex')], text=False)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    ply.write(filename)


def save_ply_property(points, property, filename, property_max=None, normals=None, cmap_name='Set1'):
    point_num = points.shape[0]
    colors = np.full([point_num, 3], 0.5)
    cmap = cm.get_cmap(cmap_name)
    if property_max is None:
        property_max = np.amax(property, axis=0)
    for point_idx in range(point_num):
        colors[point_idx] = cmap(property[point_idx] / property_max)[:3]
    save_ply(points, filename, colors, normals)
