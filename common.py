import torch
import torch.nn.functional as F
import os
import math
import random
from DSS.misc.visualize import animate_points, animate_mesh, figures_to_html
from DSS.core.lighting import PointLights, DirectionalLights
from DSS import logger_py


def create_animation(pts_dir, show_max=-1):
    figs = []
    # points
    pts_files = [f for f in os.listdir(pts_dir) if 'pts' in f and f[-4:].lower() in ('.ply', 'obj')]
    if len(pts_files) == 0:
        logger_py.info("Couldn't find '*pts*' files in {}".format(pts_dir))
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
    mesh_files = [f for f in os.listdir(pts_dir) if 'mesh' in f and f[-4:].lower() in ('.ply', '.obj')]
    # mesh_files = list(filter(lambda x: x.split('_')
    #                          [1] == '000.obj', mesh_files))
    if len(mesh_files) == 0:
        logger_py.info("Couldn't find '*mesh*' files in {}".format(pts_dir))
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


def get_tri_color_lights_for_view(cams, has_specular=False, point_lights=True):
    """
    Create RGB lights direction in the half dome
    The direction is given in the same coordinates as the pointcloud
    Args:
        cams
    Returns:
        Lights with three RGB light sources (B: right, G: left, R: bottom)
    """
    import math
    from DSS.core.lighting import (DirectionalLights, PointLights)
    from pytorch3d.renderer.cameras import look_at_rotation
    from pytorch3d.transforms import Rotate

    elev = torch.tensor(((30, 30, 30),),device=cams.device)
    azim = torch.tensor(((-60, 60, 180),),device=cams.device)
    elev = math.pi / 180.0 * elev
    azim = math.pi / 180.0 * azim

    x = torch.cos(elev) * torch.sin(azim)
    y = torch.sin(elev)
    z = torch.cos(elev) * torch.cos(azim)
    light_directions = torch.stack([x, y, z], dim=-1)
    cam_pos = cams.get_camera_center()
    R = look_at_rotation(torch.zeros_like(cam_pos), at=F.normalize(torch.cross(cam_pos, torch.rand_like(cam_pos)), dim=-1), up=cam_pos)
    light_directions = Rotate(R=R.transpose(1,2), device=cams.device).transform_points(light_directions)
    # trimesh.Trimesh(vertices=torch.cat([cam_pos, light_directions[0]], dim=0).cpu().numpy(), process=False).export('tests/outputs/light_dir.ply')
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
        location = light_directions*5
        lights = PointLights(ambient_color=ambient_color, diffuse_color=diffuse_color,
                             specular_color=specular_color, location=location)
    return lights

def get_light_for_view(cams, point_lights, has_specular):
    # create tri-color lights and a specular+diffuse shader
    ambient_color = torch.FloatTensor((((0.6, 0.6, 0.6),),))
    diffuse_color = torch.FloatTensor(
        (((0.2, 0.2, 0.2),),))

    if has_specular:
        specular_color = 0.15 * diffuse_color
        diffuse_color *= 0.85
    else:
        specular_color = (((0, 0, 0),),)

    elev = torch.tensor(((random.randint(10, 90),),), dtype=torch.float, device=cams.device)
    azim = torch.tensor(((random.randint(0, 360)),), dtype=torch.float, device=cams.device)
    elev = math.pi / 180.0 * elev
    azim = math.pi / 180.0 * azim

    x = torch.cos(elev) * torch.sin(azim)
    y = torch.sin(elev)
    z = torch.cos(elev) * torch.cos(azim)
    light_directions = torch.stack([x, y, z], dim=-1)
    # transform from camera to world
    light_directions = cams.get_world_to_view_transform().inverse().transform_points(light_directions)
    if not point_lights:
        lights = DirectionalLights(ambient_color=ambient_color, diffuse_color=diffuse_color,
                                   specular_color=specular_color, direction=light_directions)
    else:
        location = light_directions*5
        lights = PointLights(ambient_color=ambient_color, diffuse_color=diffuse_color,
                             specular_color=specular_color, location=location)
    return lights