"""
change trainer settings according to iterations
"""
from typing import List
import bisect
from .. import logger_py


class TrainerScheduler(object):
    """ Increase n_points_per_cloud and Reduce n_training_points """

    def __init__(self, init_dss_backward_radii: float = 0,
                 steps_dss_backward_radii: int = -1,
                 steps_proj: int=-1,
                 warm_up_iters: int = 0,
                 gamma_dss_backward_radii: float = 0.99,
                 gamma_proj: float = 5,
                 limit_dss_backward_radii: float = 1.5,
                 limit_proj: float = 1.0,
                 ):
        """ steps_n_points_dss: list  """

        self.init_dss_backward_radii = init_dss_backward_radii

        self.steps_dss_backward_radii = steps_dss_backward_radii
        self.steps_proj = steps_proj

        self.gamma_dss_backward_radii = gamma_dss_backward_radii
        self.gamma_proj = gamma_proj

        self.limit_dss_backward_radii = limit_dss_backward_radii
        self.limit_proj = limit_proj

        self.warm_up_iters = warm_up_iters

    def step(self, trainer, it):
        # change rasterize backward radii
        if self.steps_dss_backward_radii > 0 and hasattr(trainer.model, 'renderer'):
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

        if self.steps_proj > 0:
            i = it // self.steps_proj
            gamma = self.gamma_proj ** i
            trainer.lambda_dr_proj = min(trainer.lambda_dr_proj * gamma, self.limit_proj)