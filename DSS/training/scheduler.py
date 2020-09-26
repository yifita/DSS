"""
change trainer settings according to iterations
"""
from typing import List
import bisect
from .. import logger_py


class TrainerScheduler(object):
    """ Increase n_points_per_cloud and Reduce n_training_points """

    def __init__(self, init_n_points_dss: int, init_n_rays: int, init_dss_backward_radii: float,
                 steps_n_points_dss: int, steps_n_rays: int, steps_dss_backward_radii: int,
                 gamma_n_points_dss: float = 2.0, gamma_n_rays: float = 0.5,
                 limit_n_points_dss: int = 1e5, limit_n_rays: int = 0,
                 gamma_dss_backward_radii: float = 0.99, limit_dss_backward_radii: float = 1.5,):
        """ steps_n_points_dss: list  """

        self.init_n_points_dss = init_n_points_dss
        self.init_n_rays = init_n_rays
        self.init_dss_backward_radii = init_dss_backward_radii

        self.steps_n_points_dss = steps_n_points_dss
        self.steps_n_rays = steps_n_rays
        self.steps_dss_backward_radii = steps_dss_backward_radii

        self.gamma_n_points_dss = gamma_n_points_dss
        self.gamma_n_rays = gamma_n_rays
        self.gamma_dss_backward_radii = gamma_dss_backward_radii

        self.limit_n_points_dss = limit_n_points_dss
        self.limit_n_rays = limit_n_rays
        self.limit_dss_backward_radii = limit_dss_backward_radii

    def step(self, trainer, it):
        if self.steps_n_points_dss > 0:
            i = it // self.steps_n_points_dss
            gamma = self.gamma_n_points_dss ** i
            old_n_points_per_cloud = trainer.model.n_points_per_cloud
            trainer.model.n_points_per_cloud = min(
                int(self.init_n_points_dss * gamma), self.limit_n_points_dss)
            if old_n_points_per_cloud != trainer.model.n_points_per_cloud:
                logger_py.info('Updated n_points_per_cloud: {} -> {}'.format(
                    old_n_points_per_cloud, trainer.model.n_points_per_cloud))

                # also reduce cutoff threshold
                if not trainer.model.renderer.rasterizer.raster_settings.Vrk_isotropic:
                    old_density = trainer.model.renderer.density
                    trainer.model.renderer.density = max(old_density / (self.gamma_n_points_dss), 1e-4)
                    if old_density != trainer.model.renderer.density:
                        logger_py.info('Updated density: {} -> {}'.format(
                            old_density, trainer.model.renderer.density))

        if self.steps_n_rays > 0:
            # reduce n_rays gradually
            i = it // self.steps_n_rays
            gamma = self.gamma_n_rays ** i
            old_n_rays = trainer.n_training_points
            trainer.n_training_points = max(
                int(self.init_n_rays * gamma), self.limit_n_rays)
            if old_n_rays != trainer.n_training_points:
                logger_py.info('Updated n_training_points: {} -> {}'.format(
                    old_n_rays, trainer.n_training_points))

        # change rasterize backward radii
        if self.steps_dss_backward_radii > 0:
            # shortcut
            raster_settings = trainer.model.renderer.rasterizer.raster_settings
            i = it // self.steps_dss_backward_radii
            gamma = self.gamma_dss_backward_radii ** i
            old_backward_scaler = raster_settings.radii_backward_scaler
            raster_settings.radii_backward_scaler = max(
                self.init_dss_backward_radii * gamma, self.limit_dss_backward_radii)
            if old_backward_scaler != raster_settings.radii_backward_scaler:
                logger_py.info('Updated radii_backward_scaler: {} -> {}'.format(
                    old_backward_scaler, raster_settings.radii_backward_scaler))