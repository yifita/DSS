from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
from matplotlib import cm
import matplotlib.colors as mpc
import trimesh
from . import BaseGenerator
from .. import get_debugging_mode, get_debugging_tensor
from ..utils import get_tensor_values
from ..core.cloud import PointClouds3D, PointCloudsFilters
from ..core.texture import LightingTexture


def save_grad_with_name(name):
    def _save_grad(grad):
        dbg_tensor = get_debugging_tensor()
        # a dict of list of tensors
        dbg_tensor.pts_world_grad[name] = [
            grad[b].detach().cpu() for b in range(grad.shape[0])]

    return _save_grad

def save_mask_grad():
    def _save_grad(grad):
        dbg_tensor = get_debugging_tensor()
        dbg_tensor.img_mask_grad = grad.detach().cpu()
    return _save_grad


class Model(nn.Module):
    def __init__(self, points, normals, colors, renderer, texture=None,
                 learn_points=True, learn_normals=True, learn_colors=True,
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
        self.normals = nn.Parameter(normals.to(device=device).requires_grad_(learn_normals))
        self.colors = nn.Parameter(colors.to(device=device)).requires_grad_(
            learn_colors)
        self.n_points_per_cloud = self.points.shape[1]

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
        normals = F.normalize(self.normals, dim=-1)
        return normals

    def get_point_clouds(self, points=None, with_colors=False,
                         filter_inactive=True, **kwargs):
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
        mask = self.forward(**kwargs)['mask_img_pred']
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
        self.cameras = kwargs.get('cameras', self.cameras)
        assert(self.cameras is not None), 'cameras wasn\'t set.'
        device = self.points.device

        batch_size = self.cameras.R.shape[0]
        num_points = self.points.shape[1]
        if batch_size != self.points.shape[0]:
            assert(batch_size == 1 or self.points.shape[0] == 1), \
                'Cameras batchsize and points batchsize are incompatible.'


        if get_debugging_mode():
            dbg_tensor = get_debugging_tensor()
            points = self.points
            dbg_tensor.pts_world['position'] = [
                points[b].cpu().detach() for b in range(batch_size)]
            dbg_tensor.pts_world['normal'] = dbg_tensor.pts_world['position']

            handle = points.register_hook(save_grad_with_name('position'))
            self.hooks.append(handle)
            handle = self.normals.register_hook(save_grad_with_name('normal'))
            self.hooks.append(handle)

        # do not filter inactive here, because it will be filtered in renderer
        colored_pointclouds = self.get_point_clouds(
            with_colors=True, filter_inactive=False, **kwargs)
        # from ..core.rasterizer import _check_grad
        # colored_pointclouds.points_padded().register_hook(lambda x: _check_grad(x, 'point_modeling_padded'))
        # colored_pointclouds.points_packed().register_hook(lambda x: _check_grad(x, 'point_modeling_packed'))
        rgba = self.renderer(
            colored_pointclouds, point_clouds_filter=self.points_filter, cameras=self.cameras)
        self.points_filter.visibility = self.points_filter.visibility.any(
            dim=0, keepdim=True)
        # the activation is expanded when creating visibility filter
        self.points_filter.activation = self.points_filter.activation[:1]
        self.points_filter.inmask = self.points_filter.inmask[:1]

        rgb = rgba[..., :3]
        mask = rgba[..., -1:]

        if get_debugging_mode():
            handle = mask.register_hook(save_mask_grad())
            self.hooks.append(handle)

        # Gets point clouds repulsion and projection losses
        point_clouds = self.get_point_clouds(with_colors=False)

        # compute inmask filter, which is used for projection and repulsion
        with autograd.no_grad():
            self.points_filter.visibility = self.points_filter.visibility[
                self.points_filter.activation].unsqueeze(0)
            if mask_img is not None:
                # transform the pointclouds to view,
                # use sample_grid to get a mask_pred that shows which points are inside the mask
                # This causes repetitive transform, but makes the code better structured
                # for train/evaluation separation
                with autograd.enable_grad():
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

        return {'iso_pcl': point_clouds, 'img_pred': rgb, 'mask_img_pred': mask}

    def render(self, p_world=None, cameras=None, lights=None) -> torch.Tensor:
        """ Render point clouds to RGBA (N, H, W, 4) images"""
        cameras = cameras or self.cameras
        batch_size = cameras.R.shape[0]

        pointclouds = self.get_point_clouds(p_world, with_colors=False, with_normals=True,
            cameras=cameras, lights=lights, filter_inactive=False)

        if batch_size != len(pointclouds) and len(pointclouds) == 1:
            pointclouds = pointclouds.extend(len(cameras))

        colored_pointclouds = self.decode_color(pointclouds, cameras=cameras, lights=lights)

        # render
        rgba = self.renderer(colored_pointclouds, point_clouds_filter=self.points_filter, cameras=cameras)
        self.points_filter.activation = self.points_filter.activation[:1]
        self.points_filter.visibility = self.points_filter.visibility.any(
            dim=0, keepdim=True)
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
    def __init__(self, model, device='cpu', with_colors=False, with_normals=True,
                 img_size=(512, 512), **kwargs):
        super().__init__(model, device=device)
        self.with_colors = with_colors
        self.with_normals = with_normals
        self.img_size = img_size

    def generate_mesh(self, *args, **kwargs):
        """
        Generage mesh via poisson reconstruction
        """
        with_normals = kwargs.pop('with_normals', self.with_normals)
        with_colors = kwargs.pop('with_colors', self.with_colors)
        pcl = self.model.get_point_clouds(
            **kwargs, with_normals=with_normals, with_colors=with_colors)
        points = pcl.points_list()
        normals = pcl.normals_list()
        # logger_py.info('Running poisson reconstruction')
        meshes = []
        for b in range(len(points)):
            import pymeshlab
            m = pymeshlab.Mesh(vertex_matrix=points[b][:, :3].detach().cpu().numpy(),
                               v_normals_matrix=normals[b][:, :3].detach().cpu().numpy())
            ms = pymeshlab.MeshSet()
            ms.add_mesh(m)
            ms.surface_reconstruction_screened_poisson(depth=8)
            m = ms.current_mesh()
            mesh = trimesh.Trimesh(
                vertices=m.vertex_matrix().astype(np.float32),
                faces=m.face_matrix(), process=False)
            meshes.append(mesh)
        if len(meshes) == 1:
            return meshes.pop()
        return meshes

    def generate_pointclouds(self, *args, **kwargs) -> List[trimesh.Trimesh]:
        outputs = super().generate_pointclouds(*args, **kwargs)
        with_normals = kwargs.pop('with_normals', self.with_normals)
        with_colors = kwargs.pop('with_colors', self.with_colors)

        self.model.eval()
        pcl = self.model.get_point_clouds(
            with_colors=with_colors, with_normals=with_normals,
            require_normals_grad=False, **kwargs)
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
            colors = [x.detach().cpu().numpy() for x in colors]

        for b in range(len(points)):
            if points[b].size != 0:
                if colors[b] is not None:
                    color_dim = colors[b].shape[-1]
                    assert(color_dim == 3 or color_dim == 1)
                    if color_dim == 1:
                        # uncertainty output dim 1, color code
                        cvalue = colors[b].squeeze(1)
                        cmap = cm.get_cmap('jet')
                        normalizer = mpc.Normalize(vmin=cvalue.min(), vmax=cvalue.max())
                        cvalue = normalizer(cvalue)
                        colors[b] = cmap(cvalue)

                mesh = trimesh.Trimesh(vertices=points[b],
                                       vertex_normals=normals[b],
                                       vertex_colors=colors[b],
                                       process=False, validate=False)
                outputs.append(mesh)

        return outputs

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
        with torch.autograd.no_grad():
            self.model.eval()
            rgba = self.model.render(**kwargs)
            if rgba is not None:
                rgba = rgba.detach().cpu().numpy()
                rgba = [np.clip(img, 0, 1).squeeze(0)
                        for img in np.vsplit(rgba, rgba.shape[0])]
                outputs.extend(rgba)
            return outputs
