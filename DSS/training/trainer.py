from typing import Optional
import os
from collections import OrderedDict, defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import trimesh
import point_cloud_utils as pcu
import imageio
import plotly.graph_objs as go
import sys
from im2mesh.common import (
    check_weights, get_tensor_values, transform_to_world,
    transform_to_camera_space, sample_patch_points, arange_pixels,
    make_3d_grid, compute_iou
)
from im2mesh import losses
from im2mesh.eval import MeshEvaluator
from pytorch3d.renderer import PointsRenderer
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.ops import padded_to_packed, knn_points
from PIL import Image
from ..core.cloud import PointClouds3D, PointCloudsFilters
from ..utils.mathHelper import decompose_to_R_and_t, eps_sqrt
from ..training.losses import (
    IouLoss, ProjectionLoss, RepulsionLoss, NormalLengthLoss,
    L1Loss, SmapeLoss, ImageGradientLoss, NormalLoss, RegularizationLoss)
from .scheduler import TrainerScheduler
from ..models import PointModel
from .. import get_debugging_mode, set_debugging_mode_, get_debugging_tensor, logger_py
from ..misc import Thread
from ..misc.visualize import figures_to_html, plot_2D_quiver, plot_3D_quiver
from ..utils import ImageSaliencySampler, intersection_with_unit_sphere, valid_value_mask


class BaseTrainer(object):
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer,
                 scheduler: optim.lr_scheduler._LRScheduler, *args, **kwargs):
        self.model = model
        self.optimizer = optimizer

    def forward(self, *args, mode="train", **kwargs):
        """
        One forward pass, returns all necessary outputs to getting the losses or evaluation
        """
        raise NotImplementedError

    def train_step(self, *args, **kwargs):
        ''' Performs a training step.
        '''
        raise NotImplementedError

    def eval_step(self, *args, **kwargs):
        """ Performs a evaluation step """
        raise NotImplementedError

    def compute_loss(self, *args, **kwargs):
        """ Returns the training loss (a scalar)  """
        raise NotImplementedError

    def evaluate(self, val_dataloader, **kwargs):
        """Make models eval mode during test time"""
        eval_list = defaultdict(list)

        for data in tqdm(val_dataloader):
            eval_step_dict = self.eval_step(data, **kwargs)

            for k, v in eval_step_dict.items():
                eval_list[k].append(v)

        eval_dict = {k: np.mean(v) for k, v in eval_list.items()}
        return eval_dict

    def update_learning_rate(self):
        raise NotImplementedError


class Trainer(BaseTrainer):
    def __init__(self, model, optimizer, scheduler, device='cpu',
                 cameras=None, log_dir=None, vis_dir=None, debug_dir=None, val_dir=None,
                 threshold=0.5, n_training_points=2048, n_eval_points=4000,
                 lambda_occupied=1., lambda_freespace=1., lambda_rgb=1.,
                 lambda_normal=0.05, lambda_depth=0., lambda_image_gradients=0,
                 lambda_dr_rgb=0.0, lambda_dr_silhouette=0.0,
                 lambda_eikonal=0.01, lambda_boundary=1.0,
                 lambda_sparse_depth=0., lambda_dr_proj=1000, lambda_dr_repel=100,
                 lambda_sal=0.0, lambda_sdf_3d=0.0,
                 generator=None, patch_size=1,
                 reduction_method='sum', sample_continuous=False,
                 overwrite_visualization=True,
                 multi_gpu=False, n_debug_points=1, saliency_sampling=False,
                 combined_after=-1, clip_grad=True, resample_every=-1, resample_threshold=0.9,
                 gamma_n_points_dss=2.0, gamma_n_rays=0.6, gamma_dss_backward_radii=0.99,
                 steps_n_points_dss=500, steps_n_rays=500, steps_dss_backward_radii=100,
                 loss_frnn_radius=-1,
                 **kwargs):
        """Initialize the BaseModel class.
        Args:
            model (nn.Module)
            optimizer: optimizer
            scheduler: scheduler
            device: device
        """
        self.device = device
        self.model = model
        self.cameras = cameras

        self.tb_logger = SummaryWriter(log_dir)

        # implicit function model
        self.vis_dir = vis_dir
        self.val_dir = val_dir
        self.threshold = threshold

        self.lambda_eikonal = lambda_eikonal
        self.lambda_boundary = lambda_boundary
        self.lambda_dr_rgb = lambda_dr_rgb
        self.lambda_dr_silhouette = lambda_dr_silhouette
        self.lambda_dr_proj = lambda_dr_proj
        self.lambda_dr_repel = lambda_dr_repel
        self.lambda_occupied = lambda_occupied
        self.lambda_freespace = lambda_freespace
        self.lambda_rgb = lambda_rgb
        self.lambda_normal = lambda_normal
        self.lambda_sal = lambda_sal
        self.lambda_sdf_3d = lambda_sdf_3d
        self.lambda_depth = lambda_depth
        self.lambda_image_gradients = lambda_image_gradients
        self.lambda_sparse_depth = lambda_sparse_depth

        self.generator = generator
        self.n_eval_points = n_eval_points
        self.patch_size = patch_size
        self.reduction_method = reduction_method
        self.sample_continuous = sample_continuous
        self.overwrite_visualization = overwrite_visualization
        self.saliency_sampling = saliency_sampling
        self.combined_after = combined_after
        self.resample_every = resample_every
        self.resample_threshold = resample_threshold
        self.clip_grad = clip_grad

        self.sample_salient_image_patches = ImageSaliencySampler(
            k_gaussian=0, mu=0, sigma=1, k_sobel=3, nms=False, thresholding=True,
            patch_size=patch_size, device=device
        )

        if isinstance(self.model, PointModel):
            self.training_scheduler = TrainerScheduler(init_n_points_dss=self.model.n_points_per_cloud,
                                                       init_n_rays=n_training_points,
                                                       init_dss_backward_radii=self.model.renderer.rasterizer.raster_settings.radii_backward_scaler,
                                                       steps_n_points_dss=steps_n_points_dss,
                                                       steps_n_rays=steps_n_rays,
                                                       steps_dss_backward_radii=steps_dss_backward_radii,
                                                       gamma_n_points_dss=gamma_n_points_dss,
                                                       gamma_n_rays=gamma_n_rays,
                                                       gamma_dss_backward_radii=gamma_dss_backward_radii)

        self.debug_dir = debug_dir
        self.hooks = []

        self.multi_gpu = multi_gpu

        self.n_training_points = n_training_points
        self.n_debug_points = n_debug_points

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.projection_loss = ProjectionLoss(
            reduction=self.reduction_method, filter_scale=2.0)
        self.repulsion_loss = RepulsionLoss(
            reduction=self.reduction_method, filter_scale=2.0)
        self.regularization_loss = RegularizationLoss(reduction=self.reduction_method, filter_scale=2.0, loss_frnn_radius=loss_frnn_radius) 
        self.iou_loss = IouLoss(
            reduction=self.reduction_method, channel_dim=None)
        self.eikonal_loss = NormalLengthLoss(
            reduction=self.reduction_method)
        self.l1_loss = L1Loss(reduction=self.reduction_method)
        self.smape_loss = SmapeLoss(reduction=self.reduction_method)
        self.image_gradient_loss = ImageGradientLoss(
            reduction=self.reduction_method, channel_dim=None)
        self.normal_loss = NormalLoss(reduction=self.reduction_method)

        self.evaluator = MeshEvaluator(n_points=100000)

    # def evaluate(self, val_dataloader, it, **kwargs):
    #     """Make models eval mode during test time"""
    #     if hasattr(self, '_eval_process'):
    #         eval_dict = self._eval_process.get()

    #     from multiprocessing.pool import ThreadPool
    #     pool = ThreadPool(processes=1)
    #     self._eval_process = pool.apply_async(
    #         self._evaluate, (val_dataloader, it), kwargs)  # tuple of args for foo
    #     # self._evaluate(val_dataloader, it, **kwargs)

    def evaluate(self, val_dataloader, it, **kwargs):
        if not os.path.exists(self.val_dir):
            os.makedirs(self.val_dir)

        eval_list = defaultdict(list)

        mesh_gt = val_dataloader.dataset.get_meshes()
        assert(mesh_gt is not None)
        mesh_gt = mesh_gt.to(device=self.device)

        pointcloud_tgt = val_dataloader.dataset.get_pointclouds(
            num_points=self.n_eval_points)

        logger_py.info("Evaluating")
        # create mesh using generator
        _resolution0 = self.generator.resolution0
        _upsampling_steps = self.generator.upsampling_steps
        self.generator.resolution0 = 32
        self.generator.upsampling_steps = 3

        output = self.generator.generate_meshes(
            {}, with_colors=False, with_normals=False)
        mesh = output.pop()
        self.generator.resolution0 = _resolution0
        self.generator.upsampling_steps = _upsampling_steps

        # evaluate in another thread
        eval_dict_mesh = self.evaluator.eval_mesh(
            mesh, pointcloud_tgt.points_packed().numpy(), pointcloud_tgt.normals_packed().numpy())

        # save to "val" dict
        mesh.export(os.path.join(self.val_dir, "%010d.obj" % it))
        return eval_dict_mesh

    def train_step(self, data, cameras, **kwargs):
        """
        Args:
            data (dict): contains img, img.mask and img.depth and camera_mat
            cameras (Cameras): Cameras object from pytorch3d
        Returns:
            loss
        """
        self.model.train()
        self.optimizer.zero_grad()
        it = kwargs.get("it", None)
        data, cameras = self.process_data_dict(data, cameras)
        # autograd.set_detect_anomaly(True)
        loss = self.compute_loss(data, cameras, it=it)
        loss.backward()
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.)
        self.optimizer.step()
        check_weights(self.model.state_dict())
        if hasattr(self, 'training_scheduler'):
            self.training_scheduler.step(self, it)
        return loss.item()

    def process_data_dict(self, data, cameras):
        ''' Processes the data dictionary and returns respective tensors

        Args:
            data (dictionary): data dictionary
        '''
        device = self.device

        # Get "ordinary" data
        img = data.get('img.rgb').to(device)
        assert(img.min() >= 0 and img.max() <=
               1), "Image must be a floating number between 0 and 1."
        mask_img = data.get('img.mask').to(device)

        camera_mat = data.get('camera_mat', None)

        # inputs for SVR
        inputs = data.get('inputs', torch.empty(0, 0)).to(device)

        mesh = data.get('shape.mesh', None)
        if mesh is not None:
            mesh = mesh.to(device=device)

        # set camera matrix to cameras
        if camera_mat is None:
            logger_py.warning(
                "Camera matrix is not provided! Using the identity matrix")

        cameras.R, cameras.T = decompose_to_R_and_t(camera_mat)
        cameras._N = cameras.R.shape[0]
        cameras.to(device)

        return (img, mask_img, inputs, mesh), cameras

    def sample_pixels(self, n_rays: int, img: Optional[torch.Tensor] = None,
                      batch_size=None, h=None, w=None, it: int = 0):

        if img is None:
            assert(batch_size is not None)
            assert(h is not None)
            assert(w is not None)
        else:
            batch_size, _, h, w = img.shape

        if n_rays >= h * w:
            p = arange_pixels((h, w), batch_size)[1].to(self.device)
        else:
            if self.saliency_sampling:
                # half random
                _it = it or 0
                n_random_points = max(
                    n_rays // 4, int(n_rays - _it / 4000.0 * n_rays))
                p_random = sample_patch_points(batch_size, n_random_points,
                                               patch_size=self.patch_size,
                                               image_resolution=(h, w),
                                               continuous=self.sample_continuous,
                                               ).to(self.device)
                p_salient = self.sample_salient_image_patches(
                    n_rays - n_random_points, img)
                p = torch.cat([p_random, p_salient], dim=1)
            else:
                p = sample_patch_points(batch_size, n_rays,
                                        patch_size=self.patch_size,
                                        image_resolution=(h, w),
                                        continuous=self.sample_continuous,
                                        ).to(self.device)
        return p

    def sample_from_mesh(self):
        """
        Construct mesh from implicit model and sample from the mesh to get iso-surface points,
        which is used for projection in the combined model
        """
        n_points = self.model.n_points_per_cloud
        try:
            _resolution0 = self.generator.resolution0
            _upsampling_steps = self.generator.upsampling_steps
            self.generator.resolution0 = 32
            self.generator.upsampling_steps = 4
            output = self.generator.generate_meshes(
                {}, with_colors=False, with_normals=False)
            mesh = output.pop()
            self.generator.resolution0 = _resolution0
            self.generator.upsampling_steps = _upsampling_steps
            if not os.path.exists(self.vis_dir):
                os.makedirs(self.vis_dir)
            points, _ = pcu.sample_mesh_poisson_disk(
                mesh.vertices, mesh.faces, mesh.vertex_normals.ravel().reshape(-1, 3), n_points, use_geodesic_distance=True)
            p_idx = np.random.permutation(points.shape[0])[:n_points]
            points = points[p_idx, ...]
            self.model.points = torch.tensor(
                points, dtype=torch.float, device=self.model.device).view(1, -1, 3)

            # skip model uniform sampling
        except Exception as e:
            logger_py.error("Couldn't sample points from mesh: " + repr(e))

    def compute_loss(self, data, cameras, n_points=None, eval_mode=False, it=None):
        ''' Compute the loss.
        Args:
            data (dict): data dictionary
            eval_mode (bool): whether to use eval mode
            it (int): training iteration
        '''
        # Initialize loss dictionary and other values
        loss = {}

        # overwrite n_points
        if n_points is None:
            n_points = self.n_eval_points if eval_mode else self.n_training_points

        # Process data dictionary
        img, mask_img, inputs, mesh_gt = data

        # Shortcuts
        device = self.device
        patch_size = self.patch_size
        reduction_method = self.reduction_method
        batch_size, _, h, w = img.shape

        # Assertions
        assert(((h, w) == mask_img.shape[2:4]) and
               (patch_size > 0))

        # Apply losses
        # 1.) Initialize loss
        loss['loss'] = 0
        pcl_filters = None
        if isinstance(self.model, PointModel):
            point_clouds, img_pred, mask_img_pred = self.model(
                mask_img, cameras=cameras)

        # else:
        #     # 1.) Sample points on image plane ("pixels")
        #     p = self.sample_pixels(n_points, img, it=it)

        #     # 2.) Get Object Mask values and define masks for losses
        #     mask_gt = get_tensor_values(
        #         mask_img.float(), p, squeeze_channel_dim=True).bool()

        #     point_clouds, mask_pred, \
        #         p_freespace, p_occupancy, \
        #         sdf_freespace, sdf_occupancy = self.model(
        #             p, mask_gt=mask_gt, inputs=inputs, cameras=cameras, it=it)
        #     rgb_pred = point_clouds.features_packed()
        #     # rgb groundtruth
        #     mask_rgb = mask_pred & mask_gt
        #     rgb_gt = get_tensor_values(img, p)[mask_rgb]

        # 4.) Calculate Loss
        if isinstance(self.model, PointModel):
            # DifferentiableRenderer loss
            if img_pred is not None and mask_img_pred is not None:

                self.calc_dr_loss(img.permute(0, 2, 3, 1),
                                  img_pred,
                                  mask_img.squeeze(1),
                                  mask_img_pred.squeeze(-1),
                                  'mean', loss=loss)
            # Projection and Repulsion loss
            self.calc_pcl_reg_loss(
                point_clouds, reduction_method, loss, it=it
            )


        for k, v in loss.items():
            mode = 'val' if eval_mode else 'train'
            if isinstance(v, torch.Tensor):
                self.tb_logger.add_scalar('%s/%s' % (mode, k), v.item(), it)
            else:
                self.tb_logger.add_scalar('%s/%s' % (mode, k), v, it)

        return loss if eval_mode else loss['loss']

    def prune(self, data_loader, n_views=1, **kwargs):
        """ Prune points that receive no gradients in DSS """
        # TODO(yifan) need to improve
        cameras = kwargs.get('cameras', self.cameras)
        loss_func = losses.l1_loss
        active_points = None
        if isinstance(self.model, PointModel):
            view_counter = 0
            for batch in data_loader:
                if view_counter >= n_views:
                    break
                data, cameras = self.process_data_dict(batch, cameras)
                mask_gt = data[1]
                _active_points = self.model.prune_points(
                    mask_gt.squeeze(1), loss_func, cameras=cameras)
                if active_points is None:
                    active_points = _active_points
                else:
                    active_points = _active_points | active_points

                view_counter += cameras.R.shape[0]
            # update self.model.activation
            self.model.points_activation.copy_(active_points.to(
                dtype=self.model.points_activation.dtype))
            logger_py.info('Pruned {} points'.format((~active_points).sum()))

    def calc_normals_loss(self, point_clouds, loss={}):
        """ Iso points normal computed from PCA == computed from gradient """
        if self.lambda_normal > 0:
            # reduce point_clouds if the normals are the same
            normal_loss = self.normal_loss(
                point_clouds, neighborhood_size=10) * self.lambda_normal
            loss['normal'] = normal_loss
            loss['loss'] += normal_loss

    def calc_boundary_loss(self, point_clouds, reduction_method, loss={}):
        """ penalize iso points that are outside the unit sphere """
        points = point_clouds.points_packed()
        bdry_loss = torch.nn.functional.relu(points.norm(dim=-1) - 1)
        if reduction_method == 'sum':
            bdry_loss = bdry_loss.sum()
        else:
            bdry_loss = bdry_loss.mean()
        bdry_loss = self.lambda_boundary * bdry_loss
        loss['loss_boundary'] = bdry_loss
        loss['loss'] = bdry_loss + loss['loss']

    def calc_eikonal_loss(self, normals, reduction_method, loss={}):
        """ Implicit function gradient norm == 1 """
        eikonal_loss = self.eikonal_loss(normals) * self.lambda_eikonal
        loss['loss_eikonal'] = eikonal_loss
        loss['loss'] = eikonal_loss + loss['loss']

    def calc_pcl_reg_loss(self, point_clouds, reduction_method, loss={}, **kwargs):
        """
        Args:
            point_clouds (PointClouds3D): point clouds in source space (object coordinate)
        """
        it = kwargs.get('it', 0)
        if it is None:
            it = 0

        loss_dr_repel = 0
        loss_dr_proj = 0
        # if self.lambda_dr_proj > 0:
        #     loss_dr_proj = self.projection_loss(
        #         # point_clouds, rebuild_knn=(it % 10 == 0), points_filter=self.model.points_filter) * self.lambda_dr_proj
        #         point_clouds, rebuild_knn=True, points_filter=self.model.points_filter) * self.lambda_dr_proj
        # if self.lambda_dr_repel > 0:
        #     loss_dr_repel = self.repulsion_loss(
        #         # point_clouds, rebuild_knn=(it % 10 == 0), points_filter=self.model.points_filter) * self.lambda_dr_repel
        #         point_clouds, rebuild_knn=True, points_filter=self.model.points_filter) * self.lambda_dr_repel
        if self.lambda_dr_proj > 0 or self.lambda_dr_repel > 0:
            proj_loss, repel_loss = self.regularization_loss(point_clouds, rebuild_nn=(it % 10), points_filter=self.model.points_filter)
            proj_loss *= self.lambda_dr_proj
            repel_loss *= self.lambda_dr_repel

        # with torch.no_grad():
        #     proj_loss, repel_loss = self.regularization_loss(point_clouds, rebuild_nn=(it % 10), points_filter=self.model.points_filter)
        #     proj_loss *= self.lambda_dr_proj
        #     repel_loss *= self.lambda_dr_repel
        #     logger_py.info("original proj loss: {:.6f}; repel loss: {:.6f}; new proj loss: {:.6f}; repel loss: {:.6f};".format(
        #         loss_dr_proj, loss_dr_repel, proj_loss, repel_loss
        #     ))



        # loss['loss'] = loss_dr_proj + loss_dr_repel + loss['loss']
        loss['loss'] = proj_loss + repel_loss + loss['loss']
        # loss['loss_dr_proj'] = loss_dr_proj
        # loss['loss_dr_repel'] = loss_dr_repel
        # loss['loss_dr_proj'] = proj_loss
        # loss['loss_dr_repel'] = repel_loss

    def calc_dr_loss(self, img, img_pred, mask_img, mask_img_pred,
                     reduction_method, loss={}, **kwargs):
        """
        Calculates image loss
        Args:
            img (tensor): (N,H,W,C) range [0, 1]
            img_pred (tensor): (N,H,W,C) range [0, 1]
            mask_img (tensor): (N,H,W) range [0, 1]
            mask_img_pred (tensor): (N,H,W) range [0, 1]
        """
        lambda_dr_silhouette = self.lambda_dr_silhouette
        lambda_dr_rgb = self.lambda_dr_rgb
        lambda_image_gradients = self.lambda_image_gradients

        loss_dr_silhouette = 0.
        loss_dr_rgb = 0.
        loss_image_grad = 0.

        assert(img.shape == img_pred.shape), \
            "Ground truth mage shape and predicted image shape is unequal"
        # TODO(yifan): replace mask_img with mask_img & mask_img_pred
        if lambda_dr_rgb > 0:
            mask_pred = mask_img & mask_img_pred.bool()
            if mask_pred.sum() > 0:
                loss_dr_rgb = self.l1_loss(
                    img, img_pred, mask_pred) * lambda_dr_rgb
                # loss_dr_rgb = self.smape_loss(
                #     img, img_pred, mask_pred) * lambda_dr_rgb

            if lambda_image_gradients > 0:
                loss_image_grad = self.image_gradient_loss(img.permute(0, 3, 1, 2),
                                                           img_pred.permute(
                                                               0, 3, 1, 2),
                                                           mask_pred) * lambda_image_gradients
        else:
            loss_dr_rgb = 0.0

        if lambda_dr_silhouette > 0:
            # gt_edge = self.image_gradient_loss.edge_extractor.to(
            #     device=mask_img.device)(mask_img.float().unsqueeze(1)).squeeze(1)
            loss_mask = (mask_img.float() - mask_img_pred).abs() * (1)
            loss_mask = loss_mask.mean()
            if isinstance(self.model, PointModel):
                loss_iou = self.iou_loss(mask_img.float(), mask_img_pred)
                loss_dr_silhouette = (
                    10 * loss_iou + loss_mask) * lambda_dr_silhouette
            else:
                loss_dr_silhouette = (loss_mask) * lambda_dr_silhouette

        loss['loss'] = loss_dr_rgb + loss_dr_silhouette + \
            loss_image_grad + loss['loss']
        loss['loss_dr_silhouette'] = loss_dr_silhouette
        loss['loss_dr_rgb'] = loss_dr_rgb
        loss['loss_image_gradients'] = loss_image_grad

    def calc_sdf_loss(self, sdf_freespace, sdf_occupancy, alpha, reduction_method, loss={}):
        """
        [1] Multiview Neural Surface Reconstruction with Implicit Lighting and Material (eq. 7)
        Penalize occupancy, different to [1], penalize only if sdf sign is wrong,
        TODO: Use point clouds to find the closest point for sdf?
        Args:
            sdf_freespace (tensor): (N1,)
            sdf_occupancy (tensor): (N2,)
            alpha (float):
        """
        if self.lambda_freespace > 0 and sdf_freespace is not None and sdf_freespace.nelement() > 0:
            assert(sdf_freespace.ndim == 1)
            # sdf_freespace = sdf_freespace[sdf_freespace < 0]
            if sdf_freespace.nelement() > 0:
                loss_freespace = losses.freespace_loss(-alpha * sdf_freespace,
                                                       reduction_method=reduction_method)
                loss_freespace = loss_freespace * self.lambda_freespace / alpha
            else:
                loss_freespace = 0.0
            loss['loss'] = loss_freespace + loss['loss']
            loss['loss_freespace'] = loss_freespace

        if self.lambda_occupied > 0 and sdf_occupancy is not None and sdf_occupancy.nelement() > 0:
            assert(sdf_occupancy.ndim == 1)
            # sdf_occupancy = sdf_occupancy[sdf_occupancy > 0]
            if sdf_occupancy.nelement() > 0:
                loss_occupancy = losses.occupancy_loss(-alpha * sdf_occupancy,
                                                       reduction_method=reduction_method)
                loss_occupancy = loss_occupancy * self.lambda_occupied / alpha
            else:
                loss_occupancy = 0.0
            loss['loss'] = loss_occupancy + loss['loss']
            loss['loss_occupancy'] = loss_occupancy

    def calc_photoconsistency_loss(self, rgb_pred, rgb_gt,
                                   reduction_method, loss, patch_size,
                                   eval_mode=False):
        ''' Calculates the photo-consistency loss.

        Args:
            rgb_pred (tensor): predicted rgb color values (N,3)
            rgb_gt (tensor): ground truth color (N,3)
            reduction_method (string): how to reduce the loss tensor
            loss (dict): loss dictionary
            patch_size (int): size of sampled patch
            eval_mode (bool): whether to use eval mode
        '''
        if self.lambda_rgb != 0:
            n_pts, _ = rgb_pred.shape
            loss_rgb_eval = torch.tensor(3)

            # 3.1) Calculate RGB Loss
            loss_rgb = losses.l1_loss(
                rgb_pred, rgb_gt,
                reduction_method) * self.lambda_rgb
            loss['loss'] = loss_rgb + loss['loss']
            loss['loss_rgb'] = loss_rgb

    def calc_mask_intersection(self, mask_gt, mask_pred, loss={}):
        ''' Calculates th intersection and IoU of provided mask tensors.

        Args:
            mask_gt (tensor): GT mask
            mask_pred (tensor): predicted mask
            loss (dict): loss dictionary
        '''
        mask_intersection = (mask_gt == mask_pred).float().mean()
        mask_iou = compute_iou(
            mask_gt.cpu().float(), mask_pred.cpu().float()).mean()
        loss['mask_intersection'] = mask_intersection
        loss['mask_iou'] = mask_iou

    def visualize(self, data, cameras, it=0, vis_type='mesh', **kwargs):
        ''' Visualized the data.

        Args:
            data (dict): data dictionary
            it (int): training iteration
            vis_type (string): visualization type
        '''
        if self.multi_gpu:
            print(
                "Sorry, visualizations currently not implemented when using \
                multi GPU training.")
            return 0

        if not os.path.exists(self.vis_dir):
            os.makedirs(self.vis_dir)

        # naming
        if self.overwrite_visualization:
            prefix = ''
        else:
            prefix = '%010d_' % it

        # use only one mini-batch
        for k in data:
            data[k] = data[k][:1]

        with torch.autograd.no_grad():
            device = self.device
            _, cameras = self.process_data_dict(data, cameras)
            try:
                # visualize the rendered image and pointcloud
                if vis_type == 'image':
                    img_list = self.generator.generate_images(
                        data, cameras=cameras, **kwargs)
                    for i, img in enumerate(img_list):
                        out_file = os.path.join(
                            self.vis_dir, '%s%03d' % (prefix, i))
                        if isinstance(img, go.Figure):
                            img.write_html(out_file + '.html')
                        else:
                            imageio.imwrite(out_file + '.png', img)

                    # visualize ground truth image and mask
                    img_gt = data.get('img.rgb').permute(0, 2, 3, 1)
                    mask_gt = data.get('img.mask').float().permute(0, 2, 3, 1)
                    rgba_gt = torch.cat([img_gt, mask_gt], dim=-1)
                    for i in range(rgba_gt.shape[0]):
                        img = rgba_gt[i].cpu().numpy() * 255.0
                        out_file = os.path.join(
                            self.vis_dir, '%s%03d_Gt.png' % (prefix, i))
                        imageio.imwrite(out_file, img.astype(np.uint8))

                elif vis_type == 'pointcloud':
                    pcl_list = self.generator.generate_pointclouds(
                        data, cameras=cameras, **kwargs)
                    for i, pcl in enumerate(pcl_list):
                        if isinstance(pcl, trimesh.Trimesh):
                            pcl_out_file = os.path.join(
                                self.vis_dir, '%s%03d.ply' % (prefix, i))
                            pcl.export(pcl_out_file, vertex_normal=True)

                elif vis_type == 'mesh':
                    mesh_list = self.generator.generate_meshes(
                        data, cameras=cameras, **kwargs)

                    for i, mesh in enumerate(mesh_list):
                        if isinstance(mesh, trimesh.Trimesh):
                            mesh_out_file = os.path.join(
                                self.vis_dir, '%s%03d.obj' % (prefix, i))
                            mesh.export(mesh_out_file, include_color=True)

            except Exception as e:
                logger_py.error(
                    "Exception occurred during visualization: {} ".format(e))

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def update_learning_rate(self):
        """Update learning rates for all modifiers"""
        self.scheduler.step()

    def debug(self, data_dict, cameras, it=0, mesh_gt=None, **kwargs):
        """
        output interactive plots for debugging
        # TODO(yifan): reused code from visualize
        """
        self._threads = getattr(self, '_threads', [])
        for t in self._threads:
            t.join()

        if not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir)

        # use only one mini-batch
        for k in data_dict:
            data_dict[k] = data_dict[k][:1]

        data, cameras = self.process_data_dict(data_dict, cameras)
        with torch.autograd.no_grad():
            generated_mesh_list = self.generator.generate_meshes(
                data_dict, cameras=cameras, with_colors=False, with_normals=False)

        # incoming data is channel fist
        mask_img_gt = data[1].detach().cpu().squeeze()
        H, W = mask_img_gt.shape

        set_debugging_mode_(True)
        self.model.train()
        self.model.debug(True)
        self.optimizer.zero_grad()
        loss = self.compute_loss(data, cameras, it=it)
        loss.backward()

        # plot
        with torch.autograd.no_grad():
            dbg_tensor = get_debugging_tensor()

            # save figure
            if self.overwrite_visualization:
                ending = ''
            else:
                ending = '%010d_' % it

            # plot ground truth mesh if provided
            if mesh_gt is not None:
                assert(len(mesh_gt) == 1), \
                    "mesh_gt and gt_mask_img must have the same or broadcastable batchsize"

            try:
                # prepare data to create 2D and 3D figure
                n_pts = OrderedDict((k, dbg_tensor.pts_world_grad[k][0].shape[0])
                                    for k in dbg_tensor.pts_world_grad)

                for i, k in enumerate(dbg_tensor.pts_world_grad):
                    if dbg_tensor.pts_world[k][0].shape[0] != n_pts[k]:
                        logger_py.error('Found unequal pts[{0}] ({2}) and pts_grad[{0}] ({1}).'.format(
                            k, n_pts[k], dbg_tensor.pts_world[k][0].shape[0]))

                pts_list = [dbg_tensor.pts_world[k][0] for k in n_pts]
                grad_list = [dbg_tensor.pts_world_grad[k][0]
                             for k in n_pts]

                pts_world = torch.cat(pts_list, dim=0)
                pts_world_grad = torch.cat(grad_list, dim=0)

                try:
                    img_mask_grad = dbg_tensor.img_mask_grad[0].clone()
                except Exception:
                    img_mask_grad = None

                # convert world to ndc
                if len(cameras) > 1:
                    _cams = cameras.clone().to(device=pts_world.device)
                    _cams.R = _cams[0:0 + 1].R
                    _cams.T = _cams[0:0 + 1].T
                    _cams._N = 1
                else:
                    _cams = cameras.clone().to(device=pts_world.device)

                pts_ndc = _cams.transform_points(pts_world, eps=1e-17)
                pts_grad_ndc = _cams.transform_points(
                    pts_world + pts_world_grad, eps=1e-8)

                # create 2D plot
                pts_ndc_dict = {k: t for t, k in zip(torch.split(
                    pts_ndc, list(n_pts.values())), n_pts.keys())}
                grad_ndc_dict = {k: t for t, k in zip(torch.split(
                    pts_grad_ndc, list(n_pts.values())), n_pts.keys())}

                for k, v in pts_ndc_dict.items():
                    mask_incamera = (v[..., 2] >= _cams.znear) & (v[..., 2] <= _cams.zfar) & (
                        v[..., :2].abs() <= 1.0).all(dim=-1)
                    pts_ndc_dict[k] = v[mask_incamera]
                    grad_ndc_dict[k] = grad_ndc_dict[k][mask_incamera]

                plotter_2d = Thread(target=plot_2D_quiver, name='%sproj.html' % ending,
                                    args=(pts_ndc_dict, grad_ndc_dict,
                                          mask_img_gt.clone()),
                                    kwargs=dict(img_mask_grad=img_mask_grad,
                                                save_html=os.path.join(
                                                    self.debug_dir, '%sproj.html' % ending)),
                                    )
                plotter_2d.start()
                self._threads.append(plotter_2d)

                # create 3D plot
                pts_world_dict = {k: t for t, k in zip(torch.split(
                    pts_world, list(n_pts.values())), n_pts.keys())}
                grad_world_dict = {k: t for t, k in zip(torch.split(
                    pts_world_grad, list(n_pts.values())), n_pts.keys())}
                plotter_3d = Thread(target=plot_3D_quiver, name='%sworld.html' % ending,
                                    args=(pts_world_dict, grad_world_dict),
                                    kwargs=dict(mesh_gt=mesh_gt[0], mesh=generated_mesh_list.pop(),
                                                camera=_cams, n_debug_points=-1,
                                                save_html=os.path.join(self.debug_dir, '%sworld.html' % ending)),
                                    )
                plotter_3d.start()
                self._threads.append(plotter_2d)

            except Exception as e:
                logger_py.error(
                    'Could not plot gradient: {}'.format(repr(e)))

        # set debugging to false and remove hooks
        set_debugging_mode_(False)
        self.model.debug(False)
        self.iou_loss.debug(False)
        self.repulsion_loss.debug(False)
        self.projection_loss.debug(False)
        logger_py.info('Disabled debugging mode.')

        for h in self.hooks:
            h.remove()
        self.hooks.clear()
