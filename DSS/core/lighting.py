"""
Handle multiple light sources per batch for pytorch3d.renderer.lighting
"""
import torch
import torch.nn.functional as F

from pytorch3d.renderer import lighting, convert_to_tensors_and_broadcast


def diffuse(normals, color, direction) -> torch.Tensor:
    """
    Calculate the diffuse component of light reflection using Lambert's
    cosine law.

    Args:
        normals: (N, ..., 3) xyz normal vectors. Normals and points are
            expected to have the same shape.
        color: (1, L, 3) or (N, L, 3) RGB color of the diffuse component of the light.
        direction: (x,y,z) direction of the light

    Returns:
        colors: (N, ..., 3), same shape as the input points.

    The normals and light direction should be in the same coordinate frame
    i.e. if the points have been transformed from world -> view space then
    the normals and direction should also be in view space.

    NOTE: to use with the packed vertices (i.e. no batch dimension) reformat the
    inputs in the following way.

    .. code-block:: python

        Args:
            normals: (P, 3)
            color: (N, L, 3)[batch_idx, :] -> (P, L, 3)
            direction: (N, L, 3)[batch_idx, :] -> (P, L, 3)

        Returns:
            colors: (P, 3)

        where batch_idx is of shape (P). For meshes, batch_idx can be:
        meshes.verts_packed_to_mesh_idx() or meshes.faces_packed_to_mesh_idx()
        depending on whether points refers to the vertex coordinates or
        average/interpolated face coordinates.
    """
    # TODO: handle attentuation.
    # Ensure color and location have same batch dimension as normals
    # (N,3) (1,L,3)
    normals, color, direction = convert_to_tensors_and_broadcast(
        normals, color, direction, device=normals.device
    )

    # Ensure the same number of light color and light direction
    num_lights_per_batch = color.shape[1]
    assert(direction.shape[1] == num_lights_per_batch), \
        "color and direction must have the length on dimension (1), {} != {}".format(
            color.shape, direction.shape
    )

    normals = normals[:, None, ...]
    # Reshape direction and color so they have all the arbitrary intermediate
    # dimensions as normals. Assume first dim = batch dim, seconde dim = light dim
    # and last dim = 3.
    points_dims = normals.shape[2:-1]
    expand_dims = (-1, num_lights_per_batch) + (1,) * len(points_dims) + (3,)
    if direction.shape[2: -1] != normals.shape[2: -1]:
        direction = direction.view(expand_dims)
    if color.shape[2: -1] != normals.shape[2: -1]:
        color = color.view(expand_dims)

    # Renormalize the normals in case they have been interpolated.
    normals = F.normalize(normals, p=2, dim=-1, eps=1e-6)
    direction = F.normalize(direction, p=2, dim=-1, eps=1e-6)
    angle = F.relu(torch.sum(normals * direction, dim=-1))

    # Sum light sources
    acc_color = torch.sum(color * angle[..., None], dim=1)
    return acc_color


def specular(
    points, normals, direction, color, camera_position, shininess
) -> torch.Tensor:
    """
    Calculate the specular component of light reflection.

    Args:
        points: (N, ..., 3) xyz coordinates of the points.
        normals: (N, ..., 3) xyz normal vectors for each point.
        color: (N, L, 3) RGB color of the specular component of the light.
        direction: (N, L, 3) vector direction of the light.
        camera_position: (N, 3) The xyz position of the camera.
        shininess: (N)  The specular exponent of the material.

    Returns:
        colors: (N, ..., 3), same shape as the input points.

    The points, normals, camera_position, and direction should be in the same
    coordinate frame i.e. if the points have been transformed from
    world -> view space then the normals, camera_position, and light direction
    should also be in view space.

    To use with a batch of packed points reindex in the following way.
    .. code-block:: python::

        Args:
            points: (P, 3)
            normals: (P, 3)
            color: (N, L, 3)[batch_idx] -> (P, 3)
            direction: (N, L, 3)[batch_idx] -> (P, 3)
            camera_position: (N, 3)[batch_idx] -> (P, 3)
            shininess: (N)[batch_idx] -> (P)
        Returns:
            colors: (P, 3)

        where batch_idx is of shape (P). For meshes batch_idx can be:
        meshes.verts_packed_to_mesh_idx() or meshes.faces_packed_to_mesh_idx().
    """
    # TODO: attentuate based on inverse squared distance to the light source
    if points.shape != normals.shape:
        msg = "Expected points and normals to have the same shape: got %r, %r"
        raise ValueError(msg % (points.shape, normals.shape))

    # Ensure all inputs have same batch dimension as points
    matched_tensors = convert_to_tensors_and_broadcast(
        points, color, direction, camera_position, shininess, device=points.device
    )
    _, color, direction, camera_position, shininess = matched_tensors

    # Ensure the same number of light color and light direction
    num_lights_per_batch = color.shape[1]
    assert(direction.shape[1] == num_lights_per_batch), \
        "color and direction must have the length on dimension (1), {} != {}".format(
            color.shape, direction.shape
    )
    batch_size = color.shape[0]
    points = points[:, None, ...]
    normals = normals[:, None, ...]
    camera_position = camera_position[:, None, ...]
    # Reshape direction and color so they have all the arbitrary intermediate
    # dimensions as points. Assume first dim = batch dim, seconde dim = light dim
    # and last dim = 3.
    points_dims = normals.shape[2:-1]
    expand_dims = (-1,) + (1,) * len(points_dims)
    if direction.shape[2: -1] != normals.shape[2: -1]:
        direction = direction.view((batch_size,) + expand_dims + (3,))
    if color.shape[2: -1] != normals.shape[2: -1]:
        color = color.view((batch_size,) + expand_dims + (3,))
    if camera_position.shape != normals.shape:
        camera_position = camera_position.view(
            (batch_size,) + expand_dims + (3,))
    if shininess.shape != normals.shape:
        shininess = shininess.view((batch_size,) + expand_dims)

    # Renormalize the normals in case they have been interpolated.
    normals = F.normalize(normals, p=2, dim=-1, eps=1e-6)
    direction = F.normalize(direction, p=2, dim=-1, eps=1e-6)
    cos_angle = torch.sum(normals * direction, dim=-1)
    # No specular highlights if angle is less than 0.
    mask = (cos_angle > 0).to(torch.float32)

    # Calculate the specular reflection.
    view_direction = camera_position - points
    view_direction = F.normalize(view_direction, p=2, dim=-1, eps=1e-6)
    reflect_direction = -direction + 2 * (cos_angle[..., None] * normals)

    # Cosine of the angle between the reflected light ray and the viewer
    alpha = F.relu(torch.sum(view_direction * reflect_direction,
                             dim=-1)) * mask

    acc_color = torch.sum(color * torch.pow(alpha, shininess)[..., None],
                          dim=1)
    return acc_color


class DirectionalLights(lighting.DirectionalLights):

    def __init__(
        self,
        ambient_color=(((0.5, 0.5, 0.5), ), ),
        diffuse_color=(((0.3, 0.3, 0.3), ), ),
        specular_color=(((0.2, 0.2, 0.2), ), ),
        direction=(((0, 1, 0), ), ),
        device: str = "cpu", **kwargs
    ):
        """
        Args:
            ambient_color: RGB color of the ambient component.
            diffuse_color: RGB color of the diffuse component.
            specular_color: RGB color of the specular component.
            direction: (x, y, z) direction vector of the light.
            device: torch.device on which the tensors should be located

        The inputs can each be
            - 3 element tuple/list or list of lists
            - torch tensor of shape (1, 1, 3)
            - torch tensor of shape (1, L, 3)
            - torch tensor of shape (N, L, 3)
        The inputs are broadcast against each other so they all have batch
        dimension N.
        """
        super().__init__(
            device=device,
            ambient_color=ambient_color,
            diffuse_color=diffuse_color,
            specular_color=specular_color,
            direction=direction,
        )
        # check diffuse_color, specular_color and direction
        for prop in ('diffuse_color', 'specular_color', 'direction'):
            if getattr(self, prop).dim() != 3:
                raise ValueError("{} must be (N,L,3) tensor, got {}".format(
                    prop, repr(getattr(self, prop).shape)))

    def diffuse(self, normals, points=None) -> torch.Tensor:
        # NOTE: Points is not used but is kept in the args so that the API is
        # the same for directional and point lights. The call sites should not
        # need to know the light type.
        return diffuse(
            normals=normals, color=self.diffuse_color, direction=self.direction
        )

    def specular(self, normals, points, camera_position, shininess) -> torch.Tensor:
        return specular(
            points=points,
            normals=normals,
            color=self.specular_color,
            direction=self.direction,
            camera_position=camera_position,
            shininess=shininess,
        )


class PointLights(lighting.PointLights):
    def __init__(
        self,
        ambient_color=(((0.5, 0.5, 0.5), ), ),
        diffuse_color=(((0.3, 0.3, 0.3), ), ),
        specular_color=(((0.2, 0.2, 0.2), ), ),
        location=(((0, 1, 0), ), ),
        device: str = "cpu", **kwargs
    ):
        """
        Args:
            ambient_color: RGB color of the ambient component
            diffuse_color: RGB color of the diffuse component
            specular_color: RGB color of the specular component
            location: xyz position of the light.
            device: torch.device on which the tensors should be located

        The inputs can each be
            - 3 element tuple/list or list of lists
            - torch tensor of shape (1, L, 3)
            - torch tensor of shape (N, L, 3)
        The inputs are broadcast against each other so they all have batch
        dimension N.
        """
        super().__init__(
            device=device,
            ambient_color=ambient_color,
            diffuse_color=diffuse_color,
            specular_color=specular_color,
            location=location,
        )

    def diffuse(self, normals, points) -> torch.Tensor:
        location, points = convert_to_tensors_and_broadcast(
            self.location, points, device=self.device)
        batch, L = location.shape[:2]
        location = location

        location = location.view(
            (batch, L) + (1,) * (points.ndim - 2) + (3,))

        direction = location - points.unsqueeze(1)
        return diffuse(normals=normals, color=self.diffuse_color, direction=direction)

    def specular(self, normals, points, camera_position, shininess) -> torch.Tensor:
        """
        Args:
            points (N,*,3)
            normals (N,*,3)
            camera_position (N,3) or (1,3)
            shininess
        """
        location, points = convert_to_tensors_and_broadcast(
            self.location, points, device=self.device)
        batch, L = location.shape[:2]
        location = location

        location = location.view(
            (batch, L) + (1,) * (points.ndim - 2) + (3,))

        direction = location - points.unsqueeze(1)
        return specular(
            points=points,
            normals=normals,
            color=self.specular_color,
            direction=direction,
            camera_position=camera_position,
            shininess=shininess,
        )
