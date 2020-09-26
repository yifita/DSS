import torch
import os
from DSS.misc.visualize import animate_points, animate_mesh, figures_to_html
from DSS import logger_py


def create_animation(pts_dir, show_max=-1):
    figs = []
    # points
    pts_files = [f for f in os.listdir(pts_dir) if f[-4:].lower() == '.ply']
    if len(pts_files) == 0:
        logger_py.info("Couldn't find ply files in {}".format(pts_dir))
    else:
        pts_files.sort()
        if show_max > 0:
            pts_files = pts_files[::max(len(pts_files) // show_max, 1)]
        pts_names = list(map(lambda x: os.path.basename(x)
                             [:-4].split('_')[0], pts_files))
        pts_paths = [os.path.join(pts_dir, fname) for fname in pts_files]
        fig = animate_points(pts_paths, pts_names)
        figs.append(fig)
    # mesh
    mesh_files = [f for f in os.listdir(pts_dir) if f[-4:].lower() == '.obj']
    mesh_files = list(filter(lambda x: x.split('_')
                             [1] == '000.obj', mesh_files))
    if len(mesh_files) == 0:
        logger_py.info("Couldn't find *_000.obj files in {}".format(pts_dir))
    else:
        mesh_files.sort()
        if show_max > 0:
            mesh_files = mesh_files[::max(len(mesh_files) // show_max, 1)]
        mesh_names = list(map(lambda x: os.path.basename(x)
                              [:-4].split('_')[0], mesh_files))
        mesh_paths = [os.path.join(pts_dir, fname) for fname in mesh_files]
        fig = animate_mesh(mesh_paths, mesh_names)
        figs.append(fig)

    save_html = os.path.join(pts_dir, 'animation.html')
    os.makedirs(os.path.dirname(save_html), exist_ok=True)
    figures_to_html(figs, save_html)


def get_tri_color_lights(has_specular=False, point_lights=False):
    """
    Create RGB directional light direction ()
    The direction is given in the same coordinates as the pointcloud

    Returns:
        DirectionalLights with three RGB light sources (B: right, G: left, R: bottom)
    """
    import math
    from DSS.core.lighting import (DirectionalLights, PointLights)

    elev = torch.FloatTensor(((30, 30, -90),))
    azim = torch.FloatTensor(((90, -90, 0),))
    elev = math.pi / 180.0 * elev
    azim = math.pi / 180.0 * azim

    x = torch.cos(elev) * torch.sin(azim)
    y = torch.sin(elev)
    z = torch.cos(elev) * torch.cos(azim)
    light_directions = torch.stack([x, y, z], dim=-1)
    ambient_color = torch.FloatTensor((((0.2, 0.2, 0.2), ), ))
    diffuse_color = torch.FloatTensor(
        (((0.0, 0.0, 0.8), (0.0, 0.8, 0.0), (0.8, 0.0, 0.0), ), ))
    if has_specular:
        specular_color = 0.15 * diffuse_color
        diffuse_color *= 0.85
    else:
        specular_color = (((0, 0, 0), (0, 0, 0), (0, 0, 0), ), )
    if not point_lights:
        lights = DirectionalLights(ambient_color=ambient_color, diffuse_color=diffuse_color,
                                   specular_color=specular_color, direction=light_directions)
    else:
        location = light_directions*10
        lights = PointLights(ambient_color=ambient_color, diffuse_color=diffuse_color,
                             specular_color=specular_color, location=location)
    return lights


def get_colored_lights(has_specular=False):
    """
    Create RGB directional light direction ()
    The direction is given in the same coordinates as the pointcloud

    Returns:
        DirectionalLights with four RGB light sources
    """
    import math
    from DSS.core.lighting import (DirectionalLights, PointLights)

    elev = torch.FloatTensor(((90, -30, -30, -30),))
    azim = torch.FloatTensor(((0, -60, 60, 180),))
    elev = math.pi / 180.0 * elev
    azim = math.pi / 180.0 * azim

    x = torch.cos(elev) * torch.sin(azim)
    y = torch.sin(elev)
    z = torch.cos(elev) * torch.cos(azim)
    light_directions = torch.stack([x, y, z], dim=-1)

    ambient_color = torch.FloatTensor((((0.0, 0.0, 0.0), ), ))
    diffuse_color = torch.FloatTensor(
        (((0.0, 0.0, 0.8), (0.0, 0.8, 0.0), (0.6, 0.0, 0.0), (0.4, 0.2, 0.2)), ))
    if has_specular:
        specular_color = 0.15 * diffuse_color
        diffuse_color *= 0.85
    else:
        specular_color = (((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0),), )
    lights = DirectionalLights(ambient_color=ambient_color, diffuse_color=diffuse_color,
                               specular_color=specular_color, direction=light_directions)
    return lights
