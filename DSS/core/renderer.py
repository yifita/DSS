import torch
from pytorch3d.renderer import PointsRenderer, NormWeightedCompositor
from pytorch3d.renderer.compositing import weighted_sum
from .. import logger_py


__all__ = ['SurfaceSplattingRenderer']

"""
Returns a 4-Channel image for RGBA
"""


class SurfaceSplattingRenderer(PointsRenderer):

    def __init__(self, rasterizer, compositor, antialiasing_sigma: float = 1.0,
                 density: float = 1e-4, frnn_radius=-1):
        super().__init__(rasterizer, compositor)

        self.cameras = self.rasterizer.cameras
        self._Vrk_h = None
        # screen space low pass filter
        self.antialiasing_sigma = antialiasing_sigma
        # average of squared distance to the nearest neighbors
        self.density = density

        if self.compositor is None:
            logger_py.info('Composite with weighted sum.')
        elif not isinstance(self.compositor, NormWeightedCompositor):
            logger_py.warning('Expect a NormWeightedCompositor, but initialized with {}'.format(
                self.compositor.__class__.__name__))

        self.frnn_radius = frnn_radius
        # logger_py.error("frnn_radius: {}".format(frnn_radius))

    def forward(self, point_clouds, **kwargs) -> torch.Tensor:
        """
        point_clouds_filter: used to get activation mask and update visibility mask
        cutoff_threshold
        """
        if point_clouds.isempty():
            return None

        # rasterize
        fragments = kwargs.get('fragments', None)
        if fragments is None:
            if kwargs.get('verbose', False):
                fragments, point_clouds, per_point_info = self.rasterizer(point_clouds, **kwargs)
            else:
                fragments, point_clouds = self.rasterizer(point_clouds, **kwargs)

        # compute weight: scalar*exp(-0.5Q)
        weights = torch.exp(-0.5 * fragments.qvalue) * fragments.scaler
        weights = weights.permute(0, 3, 1, 2)

        # from fragments to rgba
        pts_rgb = point_clouds.features_packed()[:, :3]

        if self.compositor is None:
            # NOTE: weight _splat_points_weights_backward, weighted sum will return
            # zero gradient for the weights.
            images = weighted_sum(fragments.idx.long().permute(0, 3, 1, 2),
                                  weights,
                                  pts_rgb.permute(1, 0),
                                  **kwargs)
        else:
            images = self.compositor(
                fragments.idx.long().permute(0, 3, 1, 2),
                weights,
                pts_rgb.permute(1, 0),
                **kwargs
            )

        # permute so image comes at the end
        images = images.permute(0, 2, 3, 1)
        mask = fragments.occupancy

        images = torch.cat([images, mask.unsqueeze(-1)], dim=-1)

        if kwargs.get('verbose', False):
            return images, fragments
        return images
