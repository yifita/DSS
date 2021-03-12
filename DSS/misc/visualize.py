from collections import OrderedDict
from typing import Union, Dict, Optional, List
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.offline as offline
from plotly.subplots import make_subplots
import numpy as np
import torch
import os
import imageio
import numpy as np
import trimesh
from ..utils.mathHelper import decompose_to_R_and_t, ndc_to_pix, eps_denom
from .. import logger_py
from . import Thread
from ..utils import get_grid_uniform
from PIL import Image
from pytorch3d.structures import Meshes
from pytorch3d.renderer.cameras import CamerasBase
import time


def animate_points(pts_files, names=None, save_html: Optional[str] = None):
    if names is not None:
        assert(len(pts_files) == len(names))
    fig_dict = {
        "data": [],
        "layout": {},
        "frames": []
    }
    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Step:",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 300, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": []
    }
    for i, f in enumerate(pts_files):
        try:
            mesh = trimesh.load(f)
            pts = mesh.vertices
            pts_trace = go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                name=str(i) if names is None else names[i],
                mode='markers',
                marker_size=5,
            )
            data_name = str(i) if names is None else names[i]
        except Exception:
            logger_py.warn('Failed to plot Mesh3d from {}'.format(f))
        else:
            # make data and frames
            if i == 0:
                fig_dict['data'].append(pts_trace)
            frame = {"data": [pts_trace],
                     "name": data_name}
            fig_dict['frames'].append(frame)
            # slider
            step = dict(
                args=[
                    [data_name],
                    dict(frame={"duration": 300, "redraw": True},
                         mode="immediate",
                         transition={"duration": 300},)

                ],
                label=data_name,
                method="animate"
            )
            sliders_dict['steps'].append(step)

    fig_dict['layout']['sliders'] = [sliders_dict]
    fig_dict["layout"]["updatemenus"] = [
        {
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 500, "redraw": True},
                                    "fromcurrent": True, "transition": {"duration": 300,
                                                                        "easing": "quadratic-in-out"}}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                      "mode": "immediate",
                                      "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }
    ]
    fig_dict['layout']['scene'] = dict(xaxis=dict(range=[-1.5, 1.5], autorange=False),
                                       yaxis=dict(
                                           range=[-1.5, 1.5], autorange=False),
                                       zaxis=dict(
                                           range=[-1.5, 1.5], autorange=False),
                                       aspectratio=dict(x=1, y=1, z=1),
                                       camera=dict(up=dict(x=0, y=1, z=0),
                                                   center=dict(x=0, y=0, z=0),
                                                   eye=dict(x=0, y=0, z=1)))

    fig = go.Figure(fig_dict)
    if save_html is not None:
        os.makedirs(os.path.dirname(save_html), exist_ok=True)
        figures_to_html([fig], save_html)
    return fig


def animate_mesh(pts_files, names=None, save_html: Optional[str] = None):
    if names is not None:
        assert(len(pts_files) == len(names))
    fig_dict = {
        "data": [],
        "layout": {},
        "frames": []
    }
    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Step:",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 300, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": []
    }
    for i, f in enumerate(pts_files):
        try:
            mesh = trimesh.load(f)
            data_name = str(i) if names is None else names[i]
            mesh_trace = go.Mesh3d(x=mesh.vertices[:, 0],
                                   y=mesh.vertices[:, 1],
                                   z=mesh.vertices[:, 2],
                                   i=mesh.faces[:, 0],
                                   j=mesh.faces[:, 1],
                                   k=mesh.faces[:, 2],
                                   name=data_name,
                                   )
        except Exception:
            logger_py.warn('Failed to plot Mesh3d from {}'.format(f))
        else:
            # make data and frames
            if i == 0:
                fig_dict['data'].append(mesh_trace)
            frame = {"data": [mesh_trace],
                     "name": data_name}
            fig_dict['frames'].append(frame)
            # slider
            step = dict(
                args=[
                    [data_name],
                    dict(frame={"duration": 300, "redraw": True},
                         mode="immediate",
                         transition={"duration": 300},)

                ],
                label=data_name,
                method="animate"
            )
            sliders_dict['steps'].append(step)

    fig_dict['layout']['sliders'] = [sliders_dict]
    fig_dict["layout"]["updatemenus"] = [
        {
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 500, "redraw": True},
                                    "fromcurrent": True, "transition": {"duration": 300,
                                                                        "easing": "quadratic-in-out"}}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                      "mode": "immediate",
                                      "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }
    ]
    fig_dict['layout']['scene'] = dict(xaxis=dict(range=[-1.5, 1.5], autorange=False),
                                       yaxis=dict(
                                           range=[-1.5, 1.5], autorange=False),
                                       zaxis=dict(
                                           range=[-1.5, 1.5], autorange=False),
                                       aspectratio=dict(x=1, y=1, z=1),
                                       camera=dict(up=dict(x=0, y=1, z=0),
                                                   center=dict(x=0, y=0, z=0),
                                                   eye=dict(x=0, y=0, z=1)))

    fig = go.Figure(fig_dict)
    if save_html is not None:
        os.makedirs(os.path.dirname(save_html), exist_ok=True)
        figures_to_html([fig], save_html)
    return fig


def plot_3D_quiver(pts_world: Union[Dict, torch.Tensor],
                   pts_world_grad: Union[Dict, torch.Tensor],
                   mesh_gt: Optional[Meshes] = None,
                   mesh: List[trimesh.Trimesh] = None,
                   camera: CamerasBase = None,
                   n_debug_points: int = -1,
                   save_html: Optional[str] = None) -> go.Figure:
    """
    Plot 3D quiver plots (cone), for each point, plot the normalized negative gradient.
    Accept dictionaries of {str: tensor(N,3)} or tensors (N,3). In case dictionaries are proviced,
    the keys are used for hovertext, otherwise, the indices of points are used.
    Args:
        pts_world (tensor):
        pts_world_grad (tensor): the vector projected to world space, i.e. proj(pts+grad) - pts_world
        mesh_gt (Meshes): (H,W) image value range [0,1]
        save_html (str): If given, then save to file
    """
    n_pts_per_cat = None
    text_to_id = None
    if isinstance(pts_world, torch.Tensor):
        n_pts_per_cat = OrderedDict(all=pts_world.shape[0])
        if n_debug_points > 0:
            n_pts_per_cat['all'] = min(n_debug_points, n_pts_per_cat['all'])

        pts_world = pts_world.cpu()[:n_pts_per_cat['all']]
        pts_world_grad = pts_world_grad.cpu()[:n_pts_per_cat['all']]
        text = np.array([str(i) for i in range(pts_world.shape[0])])

    elif isinstance(pts_world, dict):
        n_pts_per_cat = OrderedDict(
            (k, (pts.shape[0])) for k, pts in pts_world.items())
        if n_debug_points > 0:
            for k in n_pts_per_cat:
                n_pts_per_cat[k] = min(n_debug_points, n_pts_per_cat[k])

        text_to_id = {k: i for i, k in enumerate(pts_world)}
        pts_world = torch.cat([pts_world[k][:v]
                               for k, v in n_pts_per_cat.items()], dim=0)
        pts_world_grad = torch.cat(
            [pts_world_grad[k][:v] for k, v in n_pts_per_cat.items()], dim=0)
        pts_world = pts_world.cpu()
        pts_world_grad = pts_world_grad.cpu()
        text = np.array([k for k, pts in n_pts_per_cat.items()
                         for i in range(pts)])

    # one 3D plot for the pts_world
    fig3D = go.Figure()
    # log scale pts_world_grad and clip grad with very large norms
    grad_norm = torch.norm(pts_world_grad, dim=-1, keepdim=True)
    pts_world_grad_normalized = torch.nn.functional.normalize(pts_world_grad)

    # plot Scatter3D with different colors
    if len(n_pts_per_cat) > 1 and pts_world.shape[0] > 0:
        # plot all pts and grad together
        t0 = time.time()
        fig3D.add_trace(go.Cone(
            x=pts_world.cpu().numpy()[:, 0],
            y=pts_world.cpu().numpy()[:, 1],
            z=pts_world.cpu().numpy()[:, 2],
            u=-pts_world_grad.cpu().numpy()[:, 0],
            v=-pts_world_grad.cpu().numpy()[:, 1],
            w=-pts_world_grad.cpu().numpy()[:, 2],
            customdata=grad_norm.cpu().numpy(),
            name='all',
            visible=True,
            anchor='tail',
            text=text,
            hovertemplate='<b>%{text}</b> ||Fp|| = %{customdata} <extra></extra>',
            sizemode="scaled",  # absolute|scaled
            sizeref=1.0,
            showscale=False,
            colorscale='Reds'
        ))
        t1 = time.time()
        # print('Cone All {:.3f}'.format(t1-t0))
    # plot gradients
    _start_pts = 0
    for k, n_pts in n_pts_per_cat.items():
        pts = pts_world[_start_pts:(n_pts + _start_pts)].cpu().numpy()
        grad = pts_world_grad[_start_pts:(n_pts + _start_pts)].cpu().numpy()
        norm = grad_norm[_start_pts:(n_pts + _start_pts)].cpu().numpy()
        _start_pts += n_pts
        if pts.size == 0:
            continue
        t0 = time.time()
        fig3D.add_trace(go.Cone(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            u=-grad[:, 0],
            v=-grad[:, 1],
            w=-grad[:, 2],
            customdata=norm,
            name=k + ' grad',
            visible=False,
            anchor='tail',
            hovertemplate='||Fp|| = %{customdata} <extra></extra>',
            sizemode="scaled",  # absolute|scaled
            sizeref=1,
            showscale=False,
            colorscale='Reds'
        ))
        t1 = time.time()
        # print('Cone {} {:.3f}'.format(k, t1-t0))
    # plot all sampled points
    _start_pts = 0
    for k, n_pts in n_pts_per_cat.items():
        pts = pts_world[_start_pts:(n_pts + _start_pts)].cpu().numpy()
        _start_pts += n_pts
        if pts.size == 0:
            continue
        t0 = time.time()
        fig3D.add_trace(go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            name=k,
            visible=False,
            mode='markers',
            marker_size=5,
        ))
        t1 = time.time()
        # print('Scatter3d {} {:.3f}'.format(k, t1-t0))

    # make first plot visible
    fig3D.data[0].visible = True

    # plot ground truth mesh
    # TODO(yifan): toggle visibility
    n_always_visible_plots = 0
    if mesh_gt is not None:
        t0 = time.time()
        fig3D.add_trace(
            go.Mesh3d(x=mesh_gt.verts_list()[0][:, 0], y=mesh_gt.verts_list()[0][:, 1], z=mesh_gt.verts_list()[0][:, 2],
                      i=mesh_gt.faces_list()[0][:, 0], j=mesh_gt.faces_list()[
                0][:, 1], k=mesh_gt.faces_list()[0][:, 2],
                name='mesh_gt', opacity=0.5, color='gold')
        )
        t1 = time.time()
        # print('Mesh GT {:.3f}'.format(t1-t0))
        n_always_visible_plots += 1

    # plot predicted mesh
    # TODO(yifan): toggle visibility
    if mesh is not None and not mesh.is_empty:
        t0 = time.time()
        fig3D.add_trace(
            go.Mesh3d(x=mesh.vertices[:, 0],
                      y=mesh.vertices[:, 1],
                      z=mesh.vertices[:, 2],
                      i=mesh.faces[:, 0],
                      j=mesh.faces[:, 1],
                      k=mesh.faces[:, 2],
                      name='predicted surface', opacity=0.2, color='lightgray',
                      hoverinfo='name')
        )
        t1 = time.time()
        # print('Mesh Predicted {:.3f}'.format(t1-t0))
        n_always_visible_plots += 1

    menu_buttons = []
    for i, trace in enumerate(fig3D.data):
        visibles = [False] * len(fig3D.data)
        visibles[i] = True
        for p in range(n_always_visible_plots):
            visibles[len(fig3D.data) - p - 1] = True
        menu_buttons.append({
            'label': trace.name,
            'args': [{'visible': visibles},
                     {'title': trace.name}],
            'method': 'update'
        })
    cam_up = camera.get_full_projection_transform().inverse().get_matrix()[
        0, 1, :3].tolist()
    cam_up = dict(x=cam_up[0], y=cam_up[1], z=cam_up[2])
    cam_eye = camera.get_camera_center()[0, :].tolist()
    cam_eye = dict(x=cam_eye[0], y=cam_eye[1], z=cam_eye[2])

    fig3D.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=menu_buttons,
            )
        ],
        scene=dict(xaxis=dict(range=[-1.2, 1.2], autorange=False),
                   yaxis=dict(range=[-1.2, 1.2], autorange=False),
                   zaxis=dict(range=[-1.2, 1.2], autorange=False),
                   aspectratio=dict(x=1, y=1, z=1),
                   camera=dict(up=cam_up, center=dict(x=0, y=0, z=0), eye=cam_eye))
    )
    if save_html is not None:
        os.makedirs(os.path.dirname(save_html), exist_ok=True)
        # figures_to_html([fig3D], save_html)
        fig3D.write_html(save_html)
    return fig3D


def plot_2D_quiver(pixels: Union[torch.Tensor, dict],
                   grad_pixels: Union[torch.Tensor, dict],
                   img_mask_gt: torch.Tensor,
                   img_mask_grad=None, img=None,
                   n_debug_points=400, save_html=None) -> go.Figure:
    """
    Plot 2D quiver plots, for each point, point the negative gradient projected on 2D image space.
    Accept dictionaries of {str: tensor(N,3)} or tensors (N,3). In case dictionaries are proviced,
    the keys are used for hovertext, otherwise, the indices of points are used.
    Args:
        pts_ndc (tensor): (P, 3) or dictionary {name: tensor(P,3)}
        pts_grad_ndc (tensor): the vector projected to ndc space, i.e. proj(pts+grad)
        img_mask_gt (tensor): (H,W,1) value from ground truth mask
        img_mask_grad (tensor): (H,W,1) value from occupancy map gradient
        img (tensor): (H,W,3) value of the rendered rgb, will be ignored if img_mask_grad is provided
        n_debug_points (int): plot limited number of quiver to better readability,
            if < 0, plot all.
        save_html (str): If given, then save to file
    """
    H, W = img_mask_gt.shape[:2]
    img_mask_gt = img_mask_gt.cpu()
    resolution = torch.tensor(
        (W, H), device=img_mask_gt.device, dtype=torch.float)
    if img_mask_gt.ndim == 3:
        img_mask_gt.squeeze_(-1)
    assert(img_mask_gt.ndim == 2)

    n_pts_per_cat = None
    text_to_id = None
    if isinstance(pixels, torch.Tensor):
        pixels = pixels.cpu()
        grad_pixels = grad_pixels.cpu()
        text = np.array([str(i) for i in range(pixels.shape[0])])
        n_pts_per_cat = OrderedDict(all=pixels.shape[0])
    elif isinstance(pixels, dict):
        n_pts_per_cat = OrderedDict(
            (k, pts.shape[0]) for k, pts in pixels.items())
        text_to_id = {k: i for i, k in enumerate(pixels)}
        pixels = torch.cat([pixels[k] for k in n_pts_per_cat.keys()], dim=0)
        grad_pixels = torch.cat(
            [grad_pixels[k] for k in n_pts_per_cat.keys()], dim=0)
        pixels = pixels.cpu()
        grad_pixels = grad_pixels.cpu()
        text = np.array([k for k, pts in n_pts_per_cat.items()
                         for i in range(pts)])

    assert(
        grad_pixels.shape == pixels.shape), 'Found unequal pts and pts_grad.'
    # convert ndc to pixel
    # pixels = ndc_to_pix(pixels, resolution).cpu()
    # grad_pixels = ndc_to_pix(pts_grad_ndc, resolution).cpu()
    uv = grad_pixels - pixels
    valid_mask = ((pixels < resolution) & (pixels >= 0)).all(dim=-1)
    pixels = pixels[valid_mask]
    grad_pixels = grad_pixels[valid_mask]
    uv = uv[valid_mask]
    # scale uv to less than 1/5 image size

    # based on the gradient magnitude, select top points to visualize
    if n_debug_points > 0:
        _, indices = torch.sort(torch.norm(
            uv, dim=-1), dim=0, descending=True)
        indices = indices[:n_debug_points].cpu()
        # plot quiver for gradient
        pixels_selected = pixels[indices].numpy()
        grad_pixels_selected = grad_pixels[indices].numpy()
        text_selected = text[indices.numpy()]
        uv_selected = uv[indices].numpy()
    else:
        # plot quiver for gradient
        pixels_selected = pixels.numpy()
        grad_pixels_selected = grad_pixels.numpy()
        text_selected = text
        uv_selected = uv.numpy()

    uv_selected = uv_selected / \
        np.linalg.norm(uv_selected, axis=-1).max().item() * max(H, W) * 0.1
    fig = ff.create_quiver(x=pixels_selected[:, 0], y=pixels_selected[:, 1],
                           u=-uv_selected[:, 0], v=-uv_selected[:, 1],
                           name=('proj_grad'),
                           hoverinfo='none',
                           marker_color='orange',
                           scale=.25,
                           arrow_scale=.4,)

    # draw all points with different colors
    _start_pts = 0
    for k, n_pts in n_pts_per_cat.items():
        _pixels = pixels[_start_pts:(n_pts + _start_pts)].cpu().numpy()
        t0 = time.time()
        fig.add_traces(
            go.Scatter(x=_pixels[:, 0], y=_pixels[:, 1],
                       name=k, mode='markers',
                       marker_size=5,
                       hoverinfo='text',
                       text=text[_start_pts:(n_pts + _start_pts)]
                       )
        )
        t1 = time.time()
        # print('2D Scatter {} {:.3f}'.format(k, t1-t0))
        _start_pts += n_pts

    fig.update_layout(
        title_text=('2D quiver'),
        xaxis=dict(
            range=(0, W), constrain='domain', visible=False),
        yaxis=dict(range=(H, 0), visible=False,
                   scaleanchor="x",
                   scaleratio=1)
    )
    # background is the ground truth mask overlayed with the rendered image
    if img_mask_grad is not None:
        img_mask_grad = img_mask_grad.cpu()
        # convert to 0-1
        img_mask_grad /= eps_denom(2 * img_mask_grad.abs().max())
        assert(img_mask_grad.min().item() >= -
               0.5 and img_mask_grad.max().item() <= 0.5)
        img_mask_grad = 0.5 + img_mask_grad
        img = Image.fromarray(
            np.array(img_mask_grad.squeeze() * 255).astype(dtype=np.uint8))
    elif img is not None:
        img = img.cpu()
        # overlay two images with a *soft mask
        img_mask_gt = img_mask_gt.float()
        mask = (img_mask_gt == 0)
        img_mask_gt[mask] = 0.35
        img_mask_gt[mask == 0] = 0.65
        img_mask_gt = img_mask_gt.unsqueeze(-1)
        if img.ndim == 2:
            img = img.unsqueeze(-1)
        overlayed_image = torch.where(
            img < 0.5, 2 * img * img_mask_gt, 1 - 2 * (1 - img) * (1 - img_mask_gt))
        img = Image.fromarray(
            np.array(overlayed_image.squeeze() * 255).astype(dtype=np.uint8))
    else:
        img_mask_gt = img_mask_gt.float()
        img = Image.fromarray(
            np.array(img_mask_gt * 255).astype(dtype=np.uint8))

    fig.add_layout_image(
        dict(source=img,
             xref="x",
             yref="y",
             sizex=img.size[0],
             sizey=img.size[1],
             opacity=1.0,
             x=0, y=0,
             layer='below')
    )
    fig.update_layout(height=800, width=W / H * 800,
                      template="plotly_white")
    if save_html is not None:
        os.makedirs(os.path.dirname(save_html), exist_ok=True)
        # figures_to_html([fig], save_html)
        fig.write_html(save_html)
    return fig


def figures_to_html(figs, filename):
    '''Saves a list of plotly figures in an html file.

    Parameters
    ----------
    figs : list[plotly.graph_objects.Figure]
        List of plotly figures to be saved.

    filename : str
        File name to save in.

    '''
    with open(filename, 'w') as dashboard:
        dashboard.write("<html><head></head><body>" + "\n")

        add_js = True
        for fig in figs:
            inner_html = offline.plot(
                fig, include_plotlyjs=add_js, output_type='div'
            )

            dashboard.write(inner_html)
            add_js = False

        dashboard.write("</body></html>" + "\n")


def plot_iso_surface(sdf,
                     box_size=(1.0, 1.0, 1.0), max_n_eval_pts=1e6,
                     iso_max=0.1,
                     resolution=64, thres=0.0, save_path=None) -> go.Figure:
    """ plot levelset at a certain cross section, assume inputs are centered
    Args:
        decode_points_func: A function to extract the SDF/occupancy logits of (N, 3) points
        box_size (List[float]): bounding box dimension
        max_n_eval_pts (int): max number of points to evaluate in one inference
        resolution (int): cross section resolution xy
        thres (float): levelset value
        imgs_per_cut (int): number of images for each cut (plotted in rows)
    Returns:
        a numpy array for the image
    """
    grid = get_grid_uniform(resolution, box_side_length=box_size[0])
    z = []
    points = grid['grid_points']

    for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
        values = sdf(pnts)
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        z.append(values)

    z = np.concatenate(z, axis=0)

    z = z.astype(np.float32)
    points = points.cpu().numpy()
    fig = go.Figure(data=go.Isosurface(
        x=points[..., 0].flatten(),
        y=points[..., 1].flatten(),
        z=points[..., 2].flatten(),
        value=z.flatten(),
        opacity=0.6,
        isomin=0.0,
        isomax=iso_max,
        surface_count=3,
        surface_fill=0.8,
        # surface_pattern='A',
        # surface=dict(count=3, fill=0.7),
        caps=dict(x_show=False, y_show=False, z_show=False)
    ))

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)

    return fig


def plot_cuts(decode_points_func,
              box_size=(1.0, 1.0, 1.0), max_n_eval_pts=1e6,
              resolution=256, thres=0.0, imgs_per_cut=1, save_path=None) -> go.Figure:
    """ plot levelset at a certain cross section, assume inputs are centered
    Args:
        decode_points_func: A function to extract the SDF/occupancy logits of (N, 3) points
        box_size (List[float]): bounding box dimension
        max_n_eval_pts (int): max number of points to evaluate in one inference
        resolution (int): cross section resolution xy
        thres (float): levelset value
        imgs_per_cut (int): number of images for each cut (plotted in rows)
    Returns:
        a numpy array for the image
    """
    xmax, ymax, zmax = [b / 2 for b in box_size]
    xx, yy = np.meshgrid(np.linspace(-xmax, xmax, resolution),
                         np.linspace(-ymax, ymax, resolution))
    xx = xx.ravel()
    yy = yy.ravel()

    fig = make_subplots(rows=imgs_per_cut, cols=3,
                        subplot_titles=('xz', 'xy', 'yz'),
                        shared_xaxes='all', shared_yaxes='all',
                        vertical_spacing=0.01, horizontal_spacing=0.01,
                        )

    def _plot_cut(fig, idx, pos, decode_points_func, xmax, ymax, resolution):
        """ plot one cross section pos (3, N) """
        # evaluate points in serial
        field_input = torch.tensor(pos.T, dtype=torch.float).cuda()
        values = decode_points_func(field_input).flatten()
        if isinstance(values, torch.Tensor):
            values = values.cpu().numpy()
        values = values.reshape(resolution, resolution)
        contour_dict = dict(autocontour=False,
                            colorscale='RdBu',
                            contours=dict(start=-0.2 + thres,
                                          end=0.2 + thres,
                                          size=0.05,
                                          showlabels=True,  # show labels on contours
                                          labelfont=dict(  # label font properties
                                              size=12,
                                              color='white',
                                          )
                                          ),)
        r_idx = idx // 3

        fig.add_trace(
            go.Contour(x=np.linspace(-xmax, xmax, resolution),
                        y=np.linspace(-ymax, ymax, resolution),
                        z=values,
                        **contour_dict
                        ),
            col=idx % 3 + 1, row=r_idx + 1  # 1-idx
        )

        fig.update_xaxes(
            range=[-xmax, xmax],  # sets the range of xaxis
            constrain="range",   # meanwhile compresses the xaxis by decreasing its "domain"
            col=idx % 3 + 1, row=r_idx + 1)
        fig.update_yaxes(
            range=[-ymax, ymax],
            col=idx % 3 + 1, row=r_idx + 1
        )

    steps = np.stack([np.linspace(-b/2, b/2, imgs_per_cut+2)[1:-1] for b in box_size], axis=-1)
    for index in range(imgs_per_cut):
        position_cut = [np.vstack([xx, np.full(xx.shape[0], steps[index, 1]), yy]),
                        np.vstack([xx, yy, np.full(xx.shape[0], steps[index, 2])]),
                        np.vstack([np.full(xx.shape[0], steps[index, 0]), xx, yy]), ]
        _plot_cut(
            fig, index*3, position_cut[0], decode_points_func, xmax, zmax, resolution)
        _plot_cut(
            fig, index*3+1, position_cut[1], decode_points_func, xmax, ymax, resolution)
        _plot_cut(
            fig, index*3+2, position_cut[2], decode_points_func, ymax, zmax, resolution)

    fig.update_layout(
        title='iso-surface',
        height=512 * imgs_per_cut,
        width=512 * 3,
        autosize=False,
        scene=dict(aspectratio=dict(x=1, y=1))
    )

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)

    return fig
