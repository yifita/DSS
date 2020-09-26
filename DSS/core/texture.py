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
    ambient_color = lights.ambient_color
    diffuse_color = light_diffuse
    if ~specular:
        if normals.dim() == 2 and points.dim() == 2:
            # If given packed inputs remove batch dim in output.
            return (
                ambient_color.squeeze(),
                diffuse_color.squeeze(),
                None
            )
        return ambient_color, diffuse_color, None

    light_specular = lights.specular(
        normals=normals,
        points=points,
        camera_position=cameras.get_camera_center(),
        shininess=shininess,
    )
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
    def __init__(self, device="cpu", specular=True,
                 cameras=None, lights=None, materials=None):
        super().__init__()
        self.specular = specular
        self.lights = lights if lights is not None else DirectionalLights(
            device=device)
        self.cameras = (
            cameras if cameras is not None else OrthographicCameras(
                device=device)
        )
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

        lights = kwargs.get("lights", self.lights)
        cameras = kwargs.get("cameras", self.cameras)
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
            specular=self.specular, shininess=shininess,
        )
        points_colors_shaded = points_rgb * (ambient + diffuse)

        if specular is not None and self.specular:
            points_colors_shaded = points_colors_shaded + specular

        pointclouds_colored = pointclouds.clone()
        pointclouds_colored.update_features_(points_colors_shaded)

        return pointclouds_colored


class NeuralTexture(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, pointclouds: PointClouds3D, c=None, **kwargs) -> PointClouds3D:
        x = pointclouds.points_packed()
        n = pointclouds.normals_packed()
        cameras = kwargs.get('cameras', None)
        view_direction = None
        if cameras is not None:
            cam_pos = cameras.get_camera_center()
            cam_pos = gather_batch_to_packed(
                cam_pos, pointclouds.packed_to_cloud_idx())
            view_direction = x.detach() - cam_pos

        point_colors = self.decoder(x, normals=n, view_direction=view_direction, c=c, **kwargs)
        pointclouds_colored = pointclouds.clone()
        pointclouds_colored.update_features_(point_colors)
        return pointclouds_colored
