from typing import List
import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import trimesh
from im2mesh.common import get_tensor_values
from pytorch3d.renderer import DirectionalLights
from pytorch3d.structures import padded_to_list
from pytorch3d.ops import packed_to_padded
from pypoisson import poisson_reconstruction
from .. import get_debugging_mode, get_debugging_tensor, logger_py
from . import BaseGenerator
from ..core.cloud import PointClouds3D, PointCloudsFilters
from ..core.texture import LightingTexture
from ..utils.mathHelper import vectors_to_angles

import time


class Model(nn.Module):
    def __init__(self, points, normals, colors, renderer, texture=None,
                 learn_points=True, learn_normals=True, learn_colors=True, learn_size=True,
                 point_scale_range=(0.5, 1.5),
                 device='cpu', **kwargs):
        """
        points (1,N,3)
        normals (1,N,3)
        colors (1,N,3)
        points_activation (1,N,1)
        points_visibility (1,N,1)
        renderer
        texture
        """
        super().__init__()
        self.points = nn.Parameter(points.to(device=device)).requires_grad_(
            learn_points)
        azim, elev, _ = vectors_to_angles(*torch.unbind(normals, dim=-1))
        self.normal_azim = nn.Parameter(azim.to(device=device)).requires_grad_(
            learn_normals)
        self.normal_elev = nn.Parameter(elev.to(device=device)).requires_grad_(
            learn_normals)
        self.colors = nn.Parameter(colors.to(device=device)).requires_grad_(
            learn_colors)
        self.point_size_scaler = nn.Parameter(
            torch.tensor(1.0)).requires_grad_(learn_size)
        assert(len(point_scale_range) == 2 and point_scale_range[1] >= point_scale_range[0]), \
            "point_scale_range ({}, {}) is not valid".format(
                point_scale_range[0], point_scale_range[1])
        self.n_points_per_cloud = self.points.shape[1]
        self.point_scale_range = point_scale_range
        self.renderer = renderer.to(device=device)
        self.texture = texture.to(device=device) if texture is not None else LightingTexture(
            specular=False, device=device)
        self.register_buffer('points_activation', torch.full(
            self.points.shape[:2], True, device=device, dtype=torch.bool))
        self.points_filter = PointCloudsFilters(
            activation=self.points_activation).to(device)
        self.encoder = None
        self.cameras = None  # will be set in forward pass
        self.hooks = []

    def encode_inputs(self, inputs):
        ''' Encodes the input.

        Args:
            input (tensor): the input
        '''

        if self.encoder is not None:
            c = self.encoder(inputs)
        else:
            c = None

        return c

    def decode_color(self, pointclouds, **kwargs):
        """ Color per point """
        colored_pointclouds = self.texture(pointclouds, **kwargs)
        return colored_pointclouds

    def _get_normals(self,):
        x = torch.cos(self.normal_elev) * torch.sin(self.normal_azim)
        y = torch.sin(self.normal_elev)
        z = torch.cos(self.normal_elev) * torch.cos(self.normal_azim)
        normals = torch.stack([x, y, z], dim=-1)
        return normals

    def get_point_clouds(self, with_colors=False, filter_inactive=True, **kwargs):
        """
        Create point clouds using points parameter, normals from the implicit function
        gradients, colors from the color decoder.
        Pointclouds contains additional features: activation and visibility (if available)
        """
        points = self.points
        normals = self._get_normals()
        colors = self.colors
        pointclouds = PointClouds3D(
            points=points, normals=normals, features=colors)

        self.points_filter.set_filter(activation=self.points_activation)
        self.points_filter.to(pointclouds.device)

        if filter_inactive:
            self.points_filter.to(pointclouds.device)
            pointclouds = self.points_filter.filter_with(
                pointclouds, ('activation',))

        if with_colors:
            pointclouds = self.decode_color(pointclouds, **kwargs)

        return pointclouds

    def prune_points(self, mask_gt, loss_func, **kwargs):
        """
        signal inactive points
        compute gradient from silhouette loss in given views,
        if accumulated gradient is 0, then this point is "dead"
        Args:
            mask_gt: (N,H,W)
            loss_func: function that accept two (N,H,W) tensors as inputs and
                outputs a scalar, e.g. with loss_func(mask, mask_gt)
        Returns:
            active_points_mask: (1,P,3) bool tensor
        """
        _, _, mask = self.forward(**kwargs)
        mask_loss = loss_func(mask.squeeze().float(),
                              mask_gt.squeeze().float())
        grad = autograd.grad([mask_loss], [self.points])[0]
        active_points = ~torch.all(grad == 0.0, dim=-1)
        # self.points_activation.copy_(active_points.to(
        #     dtype=self.points_activation.dtype))
        return active_points

    def forward(self, mask_img=None, **kwargs):
        """
        Returns:
            rgb (tensor): (N, H, W, 3)
            mask (tensor): (N, H, W, 1)
        """
        t1 = time.time()
        self.cameras = kwargs.get('cameras', self.cameras)
        assert(self.cameras is not None), 'cameras wasn\'t set.'
        device = self.points.device

        batch_size = self.cameras.R.shape[0]
        num_points = self.points.shape[1]
        if batch_size != self.points.shape[0]:
            assert(batch_size == 1 or self.points.shape[0] == 1), \
                'Cameras batchsize and points batchsize are incompatible.'

        # do not filter inactive here, because it will be filtered in renderer
        colored_pointclouds = self.get_point_clouds(
            with_colors=True, cameras=self.cameras, filter_inactive=False)

        self.point_size_scaler.data = torch.clamp(
            self.point_size_scaler, min=self.point_scale_range[0], max=self.point_scale_range[1])
        torch.cuda.synchronize()
        t2 = time.time()

        rgba = self.renderer(
            colored_pointclouds, self.points_filter,
            cutoff_thres_alpha=self.point_size_scaler, cameras=self.cameras)
        torch.cuda.synchronize()
        t3 = time.time()
        self.points_filter.visibility = self.points_filter.visibility.any(dim=0, keepdim=True)
        # the activation is expanded when creating visibility filter
        self.points_filter.activation = self.points_filter.activation[:1]
        self.points_filter.inmask = self.points_filter.inmask[:1]

        rgb = rgba[..., :3]
        mask = rgba[..., -1:]

        # Gets point clouds repulsion and projection losses
        point_clouds = self.get_point_clouds(with_colors=False)
        point_clouds.points_padded()

        # compute inmask filter, which is used for projection and repulsion
        with autograd.no_grad():
            self.points_filter.visibility = self.points_filter.visibility[
                self.points_filter.activation].unsqueeze(0)
            if mask_img is not None:
                # transform the pointclouds to view,
                # use sample_grid to get a mask_pred that shows which points are inside the mask
                # This causes repetitive transform, but makes the code better structured
                # for train/evaluation separation
                p_screen_hat = self.cameras.transform_points(
                    point_clouds.points_padded())
                # p_screen_hat x, y is reversed in NDC (N,max_P,2)
                p = -p_screen_hat[..., :2]
                # (N,C,max_P)
                mask_pred = get_tensor_values(mask_img.float(),
                                              p.clamp(-1.0, 1.0),
                                              squeeze_channel_dim=True).bool()
                # NOTE(yifan): our model assumes one point cloud multiple views, so the number of points
                # in each minibatch is the same
                # TODO(yifan): change point_modeling to also use divided batches? randomize input clouds
                mask_pred = mask_pred.any(dim=0, keepdim=True)
                mask_pred = mask_pred & self.points_filter.visibility
                # will be used for projection and repulsion loss
                self.points_filter.set_filter(inmask=mask_pred)
        torch.cuda.synchronize()
        t4 = time.time()

        if get_debugging_mode():
            dbg_tensor = get_debugging_tensor()
            points = colored_pointclouds.points_padded()
            dbg_tensor.pts_world['all'] = [
                points[b].cpu().detach() for b in range(batch_size)]

            def save_mask_grad():
                def _save_grad(grad):
                    dbg_tensor = get_debugging_tensor()
                    dbg_tensor.img_mask_grad = grad.detach().cpu()
                return _save_grad

            def save_grad_with_name(name):
                def _save_grad(grad):
                    dbg_tensor = get_debugging_tensor()
                    # a dict of list of tensors
                    dbg_tensor.pts_world_grad[name] = [
                        grad[b].detach().cpu() for b in range(grad.shape[0])]

                return _save_grad

            handle = mask.register_hook(save_mask_grad())
            self.hooks.append(handle)
            handle = points.register_hook(save_grad_with_name('all'))
            self.hooks.append(handle)
        
        t1 *= 1000
        t2 *= 1000
        t3 *= 1000
        t4 *= 1000
        logger_py.info("pre-rendering time: {:.3f}; rendering time: {:.3f}; post-rendering time: {:.3f}".format(t2-t1, t3-t2, t4-t3))

        return point_clouds, rgb, mask

    def render(self, *args, **kwargs) -> torch.Tensor:
        """ Render point clouds to RGBA (N, H, W, 4) images"""
        assert(self.cameras is not None), 'cameras wasn\'t set.'

        batch_size = self.cameras.R.shape[0]
        if batch_size != self.points.shape[0]:
            assert(batch_size == 1 or self.points.shape[0] == 1), \
                'Cameras batchsize and points batchsize are incompatible.'

        # set filter_inactive False because in points will be filtered in the renderer
        colored_pointclouds = self.get_point_clouds(
            **kwargs, with_colors=True, filter_inactive=False)

        rgba = self.renderer(
            colored_pointclouds, self.points_filter, **kwargs)
        self.points_filter.activation = self.points_filter.activation[:1]
        self.points_filter.visibility = self.points_filter.visibility.any(dim=0, keepdim=True)
        self.points_filter.inmask = self.points_filter.inmask[:1]

        return rgba

    def debug(self, is_debug, **kwargs):
        if is_debug:
            # nothing to do
            pass
        else:
            for hook in self.hooks:
                hook.remove()
            self.hooks.clear()


class Generator(BaseGenerator):
    def __init__(self, model, device='cpu', with_colors=False, with_normals=True, **kwargs):
        super().__init__(model, device=device)
        self.with_colors = with_colors
        self.with_normals = with_normals

    def generate_mesh(self, *args, **kwargs):
        """
        Generage mesh via poisson reconstruction
        """
        with_normals = kwargs.pop('with_normals', self.with_normals)
        with_colors = kwargs.pop('with_colors', self.with_colors)
        pcl = self.model.get_point_clouds(**kwargs, with_normals=with_normals, with_colors=with_colors)
        points = pcl.points_list()
        normals = pcl.normals_list()
        colors = pcl.features_list()
        # logger_py.info('Running poisson reconstruction')
        meshes = []
        for b in range(len(points)):
            faces, vertices = poisson_reconstruction(
                points[b][:, :3].detach().cpu().numpy(),
                normals[b][:, :3].detach().cpu().numpy(), depth=10)
            mesh = trimesh.Trimesh(
                vertices=vertices, faces=faces, process=False)
            meshes.append(mesh)

        return meshes

    def generate_pointclouds(self, *args, outputs=[], **kwargs) -> trimesh.Trimesh:
        super().generate_pointclouds(*args, outputs=[], **kwargs)
        with_normals = kwargs.pop('with_normals', self.with_normals)
        with_colors = kwargs.pop('with_colors', self.with_colors)
        self.model.eval()
        pcl = self.model.get_point_clouds(
            with_colors=with_colors, with_normals=with_normals, require_normals_grad=False, **kwargs)
        points = pcl.points_list()
        normals = pcl.normals_list()
        colors = pcl.features_list()
        points = [x[:, :3].detach().cpu().numpy() for x in points]

        if normals is None:
            normals = [None] * len(points)
        else:
            normals = [x[:, :3].detach().cpu().numpy() for x in normals]
        if colors is None:
            colors = [None] * len(points)
        else:
            colors = [x[:, :3].detach().cpu().numpy() for x in colors]

        meshes = []
        for b in range(len(points)):
            if points[b].size != 0:
                mesh = trimesh.Trimesh(vertices=points[b],
                                    vertex_normals=normals[b],
                                    vertex_colors=colors[b],
                                    process=False)
                meshes.append(mesh)

        outputs.extend(meshes)
        return meshes

    def generate_meshes(self, *args, **kwargs) -> List[trimesh.Trimesh]:
        outputs = super().generate_meshes(*args, **kwargs)
        self.model.eval()
        meshes = Generator.generate_mesh(self, *args, **kwargs)
        outputs.extend(meshes)
        return outputs

    def generate_images(self, data, **kwargs) -> List[np.array]:
        """
        Return list of rendered point clouds (H,W,3)
        """
        outputs = super().generate_images(data, **kwargs)
        self.model.eval()
        rgba = self.model.render(**kwargs)
        if rgba is not None:
            rgba = rgba.detach().cpu().numpy()
            rgba = [(np.clip(img, 0, 1) * 255.0).astype(np.uint8).squeeze(0)
                    for img in np.vsplit(rgba, rgba.shape[0])]
            outputs.extend(rgba)
        return outputs
