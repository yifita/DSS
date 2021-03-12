"""
PointTexture class

Inputs should be fragments (including point location,
normals and other features)
Output is the color per point (doesn't have the blending step)

diffuse shader
specular shader
neural shader
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.renderer.cameras import OrthographicCameras
from .lighting import DirectionalLights
from .cloud import PointClouds3D
from .. import logger_py
from ..utils import gather_batch_to_packed


__all__ = ["LightingTexture", "NeuralTexture"]


def apply_lighting(points, normals, lights, cameras,
                   specular=True, shininess=64):
    """
    Args:
        points: torch tensor of shape (N, P, 3) or (P, 3).
        normals: torch tensor of shape (N, P, 3) or (P, 3)
        lights: instance of the Lights class.
        cameras: instance of the Cameras class.
        shininess: scalar for the specular coefficient.
        specular: (bool) whether to add the specular effect

    Returns:
        ambient_color: same shape as materials.ambient_color
        diffuse_color: same shape as the input points
        specular_color: same shape as the input points
    """
    light_diffuse = lights.diffuse(normals=normals, points=points)
    light_specular = lights.specular(
        normals=normals,
        points=points,
        camera_position=cameras.get_camera_center(),
        shininess=shininess,
    )
    ambient_color = lights.ambient_color
    if ambient_color.ndim==3:
        if ambient_color.shape[1] > 1:
            logger_py.warn('Found multiple ambient colors')
        ambient_color = torch.sum(ambient_color, dim=1)
    diffuse_color = light_diffuse
    specular_color = light_specular
    if normals.dim() == 2 and points.dim() == 2:
        # If given packed inputs remove batch dim in output.
        return (
            ambient_color.squeeze(0),
            diffuse_color.squeeze(0),
            specular_color.squeeze(0),
        )
    return ambient_color, diffuse_color, specular_color


class LightingTexture(nn.Module):
    def __init__(self, device="cpu",
                 cameras=None, lights=None, materials=None):
        super().__init__()
        self.lights = lights
        self.cameras = cameras
        if materials is not None:
            logger_py.warning("Material is not supported, ignored.")

    def forward(self, pointclouds, shininess=64, **kwargs) -> PointClouds3D:
        """
        Args:
            pointclouds (Pointclouds3D)
            points_rgb (P, 3): same shape as the packed features
        Returns:
            pointclouds (Pointclouds3D) with features set to RGB colors
        """
        if pointclouds.isempty():
            return pointclouds

        lights = kwargs.get("lights", self.lights).to(pointclouds.device)
        cameras = kwargs.get("cameras", self.cameras).to(pointclouds.device)
        if len(cameras) != len(pointclouds) and len(pointclouds) == 1:
            pointclouds = pointclouds.extend(len(cameras))
        points = pointclouds.points_packed()
        point_normals = pointclouds.normals_packed()
        points_rgb = kwargs.get("points_rgb", None)
        if points_rgb is None:
            try:
                points_rgb = pointclouds.features_packed()[:, :3]
            except:
                points_rgb = torch.ones_like(points)

        if point_normals is None:
            logger_py.warning("Point normals are required, "
                              "but not available in pointclouds. "
                              "Using estimated normals instead.")

        vert_to_cloud_idx = pointclouds.packed_to_cloud_idx()
        if points_rgb.shape[-1] != 3:
            raise ValueError("Expected points_rgb to be 3-channel,"
                             "got {}".format(points_rgb.shape))

        # Format properties of lights and materials so they are compatible
        # with the packed representation of the vertices. This transforms
        # all tensor properties in the class from shape (N, ...) -> (V, ...) where
        # V is the number of packed vertices. If the number of meshes in the
        # batch is one then this is not necessary.
        if len(pointclouds) > 1:
            lights = lights.clone().gather_props(vert_to_cloud_idx)
            cameras = cameras.clone().gather_props(vert_to_cloud_idx)

        # Calculate the illumination at each point
        ambient, diffuse, specular = apply_lighting(
            points, point_normals, lights, cameras,
            shininess=shininess,
        )
        points_colors_shaded = points_rgb * (ambient + diffuse) + specular

        pointclouds_colored = pointclouds.clone()
        pointclouds_colored.update_features_(points_colors_shaded)

        return pointclouds_colored


class NeuralTexture(nn.Module):
    def __init__(self, decoder, view_dependent=True):
        super().__init__()
        self.view_dependent = view_dependent
        self.decoder = decoder

    def forward(self, pointclouds: PointClouds3D, c=None, **kwargs) -> PointClouds3D:
        if self.decoder.dim == 3 and not self.view_dependent:
            x = pointclouds.points_packed()
        else:
            x = pointclouds.normals_packed()
            assert(x is not None)
            # x = F.normalize(x, dim=-1, eps=1e-15)
            p = pointclouds.points_packed()
            x = torch.cat([x,p], dim=-1)
            if self.view_dependent:
                cameras = kwargs.get('cameras', None)
                if cameras is not None:
                    cameras = cameras.to(pointclouds.device)
                    cam_pos = cameras.get_camera_center()
                    cam_pos = gather_batch_to_packed(
                        cam_pos, pointclouds.packed_to_cloud_idx())
                    view_direction = p[...,:3].detach() - cam_pos
                    view_direction = F.normalize(view_direction, dim=-1)
                    if hasattr(self.decoder, 'embed_fn') and self.decoder.embed_fn is not None:
                        view_direction = self.decoder.embed_fn(view_direction)
                    x = torch.cat([x, view_direction], dim=-1)


        output = self.decoder(x, c=c, **kwargs)
        pointclouds_colored = pointclouds.clone()
        pointclouds_colored.update_features_(output.rgb)
        return pointclouds_colored
