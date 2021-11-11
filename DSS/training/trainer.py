from collections import OrderedDict, defaultdict
import datetime
import os
import numpy as np
import time
import trimesh
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.utils.tensorboard import SummaryWriter
from pytorch3d.renderer.cameras import FoVPerspectiveCameras
from pytorch3d.loss import chamfer_distance
from .. import set_debugging_mode_, get_debugging_tensor, logger_py
from ..utils import slice_dict, check_weights
from ..utils.mathHelper import decompose_to_R_and_t
from ..training.losses import (
    IouLoss, ProjectionLoss, RepulsionLoss,
    L2Loss, L1Loss, SmapeLoss)
from .scheduler import TrainerScheduler
from ..models import PointModel
from ..misc import Thread
from ..misc.visualize import plot_2D_quiver, plot_3D_quiver


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

    def evaluate(self, val_dataloader, reduce=True, **kwargs):
        """Make models eval mode during test time"""
        eval_list = defaultdict(list)

        for data in tqdm(val_dataloader):
            eval_step_dict = self.eval_step(data, **kwargs)

            for k, v in eval_step_dict.items():
                eval_list[k].append(v)

        eval_dict = {k: torch.stack(v) for k, v in eval_list.items()}
        if reduce:
            eval_dict = {k: torch.mean(v) for k, v in eval_dict.items()}
        return eval_dict

    def update_learning_rate(self):
        raise NotImplementedError


class Trainer(BaseTrainer):
    def __init__(self, model, optimizer, scheduler, generator, train_loader, val_loader, device='cpu',
                 cameras=None, lights=None, log_dir=None, vis_dir=None, debug_dir=None, val_dir=None,
                 n_eval_points=8000,
                 lambda_dr_rgb=1.0, lambda_dr_silhouette=1.0,
                 lambda_dr_proj=0.0, lambda_dr_repel=0.0,
                 overwrite_visualization=True,
                 n_debug_points=4000, steps_dss_backward_radii=100,
                 **kwargs):
        """Initialize the BaseModel class.
        Args:
            model (nn.Module)
            optimizer: optimizer
            scheduler: scheduler
            device: device
        """
        self.cfg = kwargs
        self.device = device
        self.model = model
        self.cameras = cameras
        self.lights = lights

        self.val_loader = val_loader
        self.train_loader = train_loader

        self.tb_logger = SummaryWriter(
            log_dir + datetime.datetime.now().strftime("-%Y%m%d-%H%M%S"))

        # implicit function model
        self.vis_dir = vis_dir
        self.val_dir = val_dir

        self.lambda_dr_rgb = lambda_dr_rgb
        self.lambda_dr_silhouette = lambda_dr_silhouette
        self.lambda_dr_proj = lambda_dr_proj
        self.lambda_dr_repel = lambda_dr_repel

        self.generator = generator
        self.n_eval_points = n_eval_points
        self.overwrite_visualization = overwrite_visualization

        #  tuple (score, mesh)
        init_dss_backward_radii = 0
        if isinstance(self.model, PointModel):
            init_dss_backward_radii = self.model.renderer.rasterizer.raster_settings.radii_backward_scaler

        self.training_scheduler = TrainerScheduler(init_dss_backward_radii=init_dss_backward_radii,
                                                   steps_dss_backward_radii=steps_dss_backward_radii,
                                                   limit_dss_backward_radii=1.0,
                                                   steps_proj=self.cfg.get(
                                                       'steps_proj', -1),
                                                   gamma_proj=self.cfg.get('gamma_proj', 5))

        self.debug_dir = debug_dir
        self.hooks = []
        self._mesh_cache = None

        self.n_debug_points = n_debug_points

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.projection_loss = ProjectionLoss(
            reduction='mean', filter_scale=2.0, knn_k=12)
        self.repulsion_loss = RepulsionLoss(
            reduction='mean', filter_scale=2.0, knn_k=12)
        self.iou_loss = IouLoss(
            reduction='mean', channel_dim=None)
        self.l1_loss = L1Loss(reduction='mean')
        self.l2_loss = L2Loss(reduction='mean')
        self.smape_loss = SmapeLoss(reduction='mean')

    def evaluate_3d(self, val_dataloader, it, **kwargs):
        logger_py.info("[3D Evaluation]")
        t0 = time.time()
        if not os.path.exists(self.val_dir):
            os.makedirs(self.val_dir)

        # create mesh using generator
        pointcloud = self.model.get_point_clouds(
            with_colors=False, with_normals=True,
            require_normals_grad=False)

        pointcloud_tgt = val_dataloader.dataset.get_pointclouds(
            num_points=self.n_eval_points).to(device=pointcloud.device)

        cd_p, cd_n = chamfer_distance(pointcloud_tgt, pointcloud,
                         x_lengths=pointcloud_tgt.num_points_per_cloud(), y_lengths=pointcloud.num_points_per_cloud(),
                         )
        # save to "val" dict
        t1 = time.time()
        logger_py.info('[3D Evaluation] time ellapsed {}s'.format(t1 - t0))
        eval_dict = {'chamfer_point': cd_p.item(), 'chamfer_normal': cd_n.item()}
        self.tb_logger.add_scalars(
            'eval', eval_dict, global_step=it)
        if not pointcloud.isempty():
            self.tb_logger.add_mesh('eval',
                                    pointcloud.points_padded(), global_step=it)
            # mesh.export(os.path.join(self.val_dir, "%010d.ply" % it))
        return eval_dict

    def eval_step(self, data, **kwargs):
        """
        evaluate with image mask iou or image rgb psnr
        """
        from skimage.transform import resize
        lights_model = kwargs.get(
            'lights', self.val_loader.dataset.get_lights())
        cameras_model = kwargs.get(
            'cameras', self.val_loader.dataset.get_cameras())
        img_size = self.generator.img_size
        eval_dict = {'iou': 0.0, 'psnr': 0.0}
        with autograd.no_grad():
            self.model.eval()
            data = self.process_data_dict(
                data, cameras_model, lights=lights_model)
            img_mask = data['mask_img']
            img = data['img']
            # render image
            rgbas = self.generator.raytrace_images(
                img_size, img_mask, cameras=data['camera'], lights=data['light'])
            assert(len(rgbas) == 1)
            rgba = rgbas[0]
            rgba = torch.tensor(
                rgba[None, ...], dtype=torch.float, device=img_mask.device).permute(0, 3, 1, 2)

            # compare iou
            mask_gt = F.interpolate(
                img_mask.float(), img_size, mode='bilinear', align_corners=False).squeeze(1)
            mask_pred = rgba[:, 3, :, :]
            eval_dict['iou'] += self.iou_loss(mask_gt.float(),
                                              mask_pred.float(), reduction='mean')

            # compare psnr
            rgb_gt = F.interpolate(
                img, img_size, mode='bilinear', align_corners=False)
            rgb_pred = rgba[:, :3, :, :]
            eval_dict['psnr'] += self.l2_loss(
                rgb_gt, rgb_pred, channel_dim=1, reduction='mean', align_corners=False).detach()

        return eval_dict

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
        lights = kwargs.get('lights', None)
        if hasattr(self, 'training_scheduler'):
            self.training_scheduler.step(self, it)

        data = self.process_data_dict(data, cameras, lights=lights)
        self.model.train()
        # autograd.set_detect_anomaly(True)
        loss = self.compute_loss(data['img'], data['mask_img'], data['input'],
                                 data['camera'], data['light'], it=it)
        loss.backward()
        self.optimizer.step()
        check_weights(self.model.state_dict())

        return loss.item()

    def process_data_dict(self, data, cameras, lights=None):
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

        # set camera matrix to cameras
        if camera_mat is None:
            logger_py.warning(
                "Camera matrix is not provided! Using the default matrix")
        else:
            cameras.R, cameras.T = decompose_to_R_and_t(camera_mat)
            cameras._N = cameras.R.shape[0]
            cameras.to(device)

        if lights is not None:
            lights_params = data.get('lights', None)
            if lights_params is not None:
                lights = type(lights)(**lights_params).to(device)

        return {'img': img, 'mask_img': mask_img, 'input': inputs, 'camera': cameras, 'light': lights}

    def compute_loss(self, img, mask_img, inputs, cameras, lights, n_points=None, eval_mode=False, it=None):
        ''' Compute the loss.
        Args:
            data (dict): data dictionary
            eval_mode (bool): whether to use eval mode
            it (int): training iteration
        '''
        # Initialize loss dictionary and other values
        loss = {}

        # Shortcuts
        _, _, h, w = img.shape

        # Apply losses
        # Initialize loss
        loss['loss'] = 0

        model_outputs = self.model(
            mask_img, cameras=cameras, lights=lights, it=it)

        point_clouds = model_outputs.get('iso_pcl')
        mask_img_pred = model_outputs.get('mask_img_pred')
        img_pred = model_outputs.get('img_pred')

        # 4.) Calculate Loss
        self.calc_dr_loss(img.permute(0, 2, 3, 1), img_pred, mask_img.reshape(
            -1, h, w), mask_img_pred.reshape(-1, h, w), reduction_method='mean', loss=loss)
        self.calc_pcl_reg_loss(
            point_clouds, reduction_method='mean', loss=loss, it=it)

        for k, v in loss.items():
            mode = 'val' if eval_mode else 'train'
            if isinstance(v, torch.Tensor):
                self.tb_logger.add_scalar('%s/%s' % (mode, k), v.item(), it)
            else:
                self.tb_logger.add_scalar('%s/%s' % (mode, k), v, it)

        return loss if eval_mode else loss['loss']

    def calc_pcl_reg_loss(self, point_clouds, reduction_method='mean', loss={}, **kwargs):
        """
        Args:
            point_clouds (PointClouds3D): point clouds in source space (object coordinate)
        """
        loss_dr_repel = 0
        loss_dr_proj = 0
        if self.lambda_dr_proj > 0:
            loss_dr_proj = self.projection_loss(
                point_clouds, rebuild_knn=True, points_filter=self.model.points_filter) * self.lambda_dr_proj
        if self.lambda_dr_repel > 0:
            loss_dr_repel = self.repulsion_loss(
                point_clouds, rebuild_knn=True, points_filter=self.model.points_filter) * self.lambda_dr_repel

        loss['loss'] = loss_dr_proj + loss_dr_repel + loss['loss']
        loss['loss_dr_proj'] = loss_dr_proj
        loss['loss_dr_repel'] = loss_dr_repel

    def calc_dr_loss(self, img, img_pred, mask_img, mask_img_pred,
                     reduction_method='mean', loss={}, **kwargs):
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

        loss_dr_silhouette = 0.
        loss_dr_rgb = 0.
        loss_image_grad = 0.

        assert(img.shape == img_pred.shape), \
            "Ground truth mage shape and predicted image shape is unequal"
        if lambda_dr_rgb > 0:
            mask_pred = mask_img.bool() & mask_img_pred.bool()
            if mask_pred.sum() > 0:
                loss_dr_rgb = self.l1_loss(
                    img, img_pred, mask=mask_pred, reduction=reduction_method) * lambda_dr_rgb
                # loss_dr_rgb = self.smape_loss(
                #     img, img_pred, mask=mask_pred) * lambda_dr_rgb

        else:
            loss_dr_rgb = 0.0

        if lambda_dr_silhouette > 0:
            # gt_edge = self.image_gradient_loss.edge_extractor.to(
            #     device=mask_img.device)(mask_img.float().unsqueeze(1)).squeeze(1)
            loss_mask = (mask_img.float() - mask_img_pred).abs()
            loss_mask = loss_mask.mean()

            loss_iou = self.iou_loss(mask_img.float(), mask_img_pred)
            loss_dr_silhouette = (
                0.01 * loss_iou + loss_mask) * lambda_dr_silhouette

        loss['loss'] = loss_dr_rgb + loss_dr_silhouette + \
            loss_image_grad + loss['loss']
        loss['loss_dr_silhouette'] = loss_dr_silhouette
        loss['loss_dr_rgb'] = loss_dr_rgb
        loss['loss_image_gradients'] = loss_image_grad

    def visualize(self, data, cameras, lights=None, it=0, vis_type='mesh', **kwargs):
        ''' Visualized the data.

        Args:
            data (dict): data dictionary
            it (int): training iteration
            vis_type (string): visualization type
        '''
        if not os.path.exists(self.vis_dir):
            os.makedirs(self.vis_dir)

        # use only one instance in the mini-batch
        data = slice_dict(data, [0, ])

        with torch.autograd.no_grad():
            data = self.process_data_dict(data, cameras, lights)
            cameras = data['camera']
            lights = data['light']
            # visualize the rendered image and pointcloud
            try:
                if vis_type == 'image':
                    img_list = self.generator.generate_images(
                        data, cameras=cameras, lights=lights, **kwargs)
                    for i, img in enumerate(img_list):
                        self.tb_logger.add_image(
                            'train/vis/render%02d' % i, img[..., :3], global_step=it, dataformats='HWC')

                    # visualize ground truth image and mask
                    img_gt = data.get('img')
                    self.tb_logger.add_image(
                        'train/vis/gt', img_gt[0, :3], global_step=it, dataformats='CHW')

                elif vis_type == 'pointcloud':
                    pcl_list = self.generator.generate_pointclouds(
                        data, cameras=cameras, lights=lights, **kwargs)
                    camera_threejs = {}
                    if isinstance(cameras, FoVPerspectiveCameras):
                        camera_threejs = {'cls': 'PerspectiveCamera', 'fov': cameras.fov.item(),
                                          'far': cameras.zfar.item(), 'near': cameras.znear.item(),
                                          'aspect': cameras.aspect_ratio.item()}
                    for i, pcl in enumerate(pcl_list):
                        if isinstance(pcl, trimesh.Trimesh):
                            self.tb_logger.add_mesh('train/vis/points', np.array(pcl.vertices)[None, ...],
                                                    config_dict=camera_threejs,
                                                    global_step=it)

                elif vis_type == 'mesh':
                    mesh = self.generator.generate_mesh(
                        data, with_colors=False, with_normals=False)
                    camera_threejs = {}
                    if isinstance(cameras, FoVPerspectiveCameras):
                        camera_threejs = {'cls': 'PerspectiveCamera', 'fov': cameras.fov.item(),
                                          'far': cameras.far.item(), 'near': cameras.near.item(),
                                          'aspect': cameras.aspect_ratio.item()}
                    if isinstance(mesh, trimesh.Trimesh):
                        self.tb_logger.add_mesh('train/vis/mesh', np.array(mesh.vertices)[None, ...],
                                                faces=np.array(mesh.faces)[
                            None, ...],
                            config_dict=camera_threejs, global_step=it)

            except Exception as e:
                logger_py.error(
                    "Exception occurred during visualization: {} ".format(e))

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def update_learning_rate(self, it):
        """Update learning rates for all modifiers"""
        self.scheduler.step()
        for param_group in self.optimizer.param_groups:
            v = param_group['lr']
            self.tb_logger.add_scalar('train/lr', v, it)

    def debug(self, data_dict, cameras, lights=None, it=0, mesh_gt=None, **kwargs):
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
        data_dict = slice_dict(data_dict, [0, ])

        data = self.process_data_dict(data_dict, cameras, lights)

        # incoming data is channel fist
        mask_img_gt = data['mask_img'].detach().cpu().squeeze()
        H, W = mask_img_gt.shape

        set_debugging_mode_(True)
        self.model.train()
        self.model.debug(True)
        self.optimizer.zero_grad()
        loss = self.compute_loss(data['img'], data['mask_img'], data['input'],
                                 data['camera'], data['light'], it=it)
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
                mesh_gt = mesh_gt[0]
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

                pts_ndc = _cams.transform_points_screen(pts_world.view(
                    1, -1, 3), ((W, H),), eps=1e-17).view(-1, 3)[..., :2]
                pts_grad_ndc = _cams.transform_points_screen(
                    (pts_world + pts_world_grad).view(1, -1, 3), ((W, H),), eps=1e-8).view(-1, 3)[..., :2]

                # create 2D plot
                pts_ndc_dict = {k: t for t, k in zip(torch.split(
                    pts_ndc, list(n_pts.values())), n_pts.keys())}
                grad_ndc_dict = {k: t for t, k in zip(torch.split(
                    pts_grad_ndc, list(n_pts.values())), n_pts.keys())}

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
                                    kwargs=dict(mesh_gt=mesh_gt, mesh=None,
                                                camera=_cams, n_debug_points=self.n_debug_points,
                                                save_html=os.path.join(self.debug_dir, '%sworld.html' % ending)),
                                    )
                plotter_3d.start()
                self._threads.append(plotter_3d)

            except Exception as e:
                logger_py.error('Could not plot gradient: {}'.format(repr(e)))

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

    def evaluate_2d(self, val_dataloader, reduce=True, **kwargs):
        """ evaluate the model by the rendered images """
        eval_dict = super().evaluate(val_dataloader, reduce=reduce,
                                     cameras=self.val_loader.dataset.get_cameras(), lights=self.val_loader.dataset.get_lights())

        return eval_dict
