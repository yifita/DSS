import torch
import os
import plyfile
import numpy as np
from matplotlib import cm
import matplotlib.colors as mpc


def saveDebugPNG(projPoint, imgTensor, savePath):
    """
    save imgTensor to PNG, highlight the projPoint with grid lines
    params:
        projPoint (1, 2)
        imgTensor (H,W,3or1) torch.Tensor or numpy.array
    """
    import matplotlib.pyplot as plt
    # normalize imgTensor
    plt.clf()
    cmin = imgTensor.min()
    cmax = imgTensor.max()
    imgTensor = (imgTensor - cmin) / (cmax - cmin)
    imgTensor[np.isnan(imgTensor) != False] = 0.0
    if imgTensor.ndim == 2 or (imgTensor.ndim == 3 and imgTensor.shape[-1] == 1):
        plt.imshow(imgTensor, cmap='gray')
    else:
        plt.imshow(imgTensor)
    i, j = projPoint.flatten()[:]
    plt.scatter(i, j, facecolors='none', edgecolors="cyan")
    plt.axvline(x=i, color='red')
    plt.axhline(y=j, color='red')
    plt.savefig(savePath)


def encodeFlow(flowTensor: torch.Tensor, logScale=False):
    """
    encode the vector field to a colored image
    :params
        flowTensor: (H,W,2)
    :return
        rgb: (H,W,3) numpy array floating type
    """
    h, w = flowTensor.shape[:2]
    rho, phi = cart2pol(flowTensor[:, :, 0], flowTensor[:, :, 1])
    rmin, rmax = rho.min(), rho.max()
    rho = (rho - rmin) / (rmax - rmin) * 255
    if logScale:
        rho = torch.log(1 + rho)
    rho[np.isnan(rho) != False] = 0.0
    hsv = np.full((h, w, 3), 255, dtype=np.uint8)
    hsv[..., 0] = phi * 255 / 2 / np.pi
    hsv[..., 2] = rho
    from skimage.color import hsv2rgb
    rgb = hsv2rgb(hsv)
    return rgb


def cart2pol(x, y):
    """
    cartesian coordinates to polar coordinates
    return:
        rho: length
        phi: (, 2pi)
    """
    rho = (x**2 + y**2).sqrt()
    phi = np.arctan2(y, x) + np.pi
    return (rho, phi)


def pol2cart(rho, phi):
    """ polar to cartesian """
    x = rho * phi.cos()
    y = rho * phi.sin()
    return (x, y)


def read_ply(file):
    loaded = plyfile.PlyData.read(file)
    points = np.vstack([loaded['vertex'].data['x'],
                        loaded['vertex'].data['y'], loaded['vertex'].data['z']])
    if 'nx' in loaded['vertex'].data.dtype.names:
        normals = np.vstack([loaded['vertex'].data['nx'],
                             loaded['vertex'].data['ny'], loaded['vertex'].data['nz']])
        points = np.concatenate([points, normals], axis=0)

    points = points.transpose(1, 0)
    return points


def save_ply(filename, points, colors=None, normals=None, binary=True):
    """
    save 3D/2D points to ply file
    Args:
        points (numpy array): (N,2or3)
        colors (numpy uint8 array): (N, 3or4)
    """
    assert(points.ndim == 2)
    if points.shape[-1] == 2:
        points = np.concatenate(
            [points, np.zeros_like(points)[:, :1]], axis=-1)

    vertex = np.core.records.fromarrays(points.transpose(
        1, 0), names='x, y, z', formats='f4, f4, f4')
    num_vertex = len(vertex)
    desc = vertex.dtype.descr

    if normals is not None:
        assert(normals.ndim == 2)
        if normals.shape[-1] == 2:
            normals = np.concatenate(
                [normals, np.zeros_like(normals)[:, :1]], axis=-1)
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
        [plyfile.PlyElement.describe(vertex_all, 'vertex')], text=(not binary))
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    ply.write(filename)


def save_ply_property(filename, points, property,
                      property_max=None, property_min=None,
                      normals=None, cmap_name='Set1', binary=True):
    point_num = points.shape[0]
    colors = np.full([point_num, 3], 0.5)
    cmap = cm.get_cmap(cmap_name)
    if property_max is None:
        property_max = np.amax(property, axis=0)
    if property_min is None:
        property_min = np.amin(property, axis=0)
    p_range = property_max - property_min
    if property_max == property_min:
        property_max = property_min + 1
    normalizer = mpc.Normalize(vmin=property_min, vmax=property_max)
    p = normalizer(property)
    colors = cmap(p)[:, :3]
    save_ply(filename, points, colors, normals, binary)
