"""
Create Synthetic MVR data using pytorch3d point renderer
per-shape:
    data_dict.npz:
        cameras_type
        cameras_params
        lights_type
        camera_mat [V,4,4] matrix
saving per-view:
    mask (png)
    RGB (png)
    lights (dict)
    camera_mat (4,4)
    ------------------
    (used for dvr only)
    cameras.npz
        camera_mat_%d (4,4) projection scaling part (top-left 2x2 matrix from pytorch3d projection matrix)
        world_mat_%d (4,4) source-to-view matrix
        scale_mat_%d (4,4) identity matrix
    pcl.npz (sparse point clouds)
        points
        colors
        normals
"""
import numpy as np
import imageio
import argparse
import os
from tqdm import tqdm
from itertools import chain
from glob import glob
from pytorch3d.renderer import (
    RasterizationSettings,
    FoVPerspectiveCameras,
    MeshRenderer,
    MeshRasterizer,
    HardFlatShader,
    HardGouraudShader,
    HardPhongShader
)
from pytorch3d.ops import eyes, sample_points_from_meshes
from pytorch3d.io import load_obj, load_ply, save_obj
import torch
from DSS.core.camera import CameraSampler
from DSS.core.lighting import PointLights, DirectionalLights
from pytorch3d.renderer import Textures
from pytorch3d.structures import Meshes
from DSS.utils import convert_tensor_property_to_value_dict
from common import get_tri_color_lights_for_view, get_light_for_view

# torch.manual_seed(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# np.random.seed(0)


def get_names_and_paths(opt):
    points_paths = list(chain.from_iterable(glob(p) for p in opt.points))
    assert(len(points_paths) > 0), "Found no point clouds in with path {}".format(
        points_paths)

    if len(points_paths) > 1:
        points_dir = os.path.commonpath(points_paths)
        points_relpaths = [os.path.relpath(
            p, points_dir) for p in points_paths]
    else:
        points_relpaths = [os.path.basename(p) for p in points_paths]

    name_and_path = {(os.path.splitext(rel_path)[0].replace(os.path.sep, "_"), path)
                     for rel_path, path in zip(points_relpaths, points_paths)}
    return name_and_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Create Synthetic MVR data saving per-view: RGBA, camera matrix, depth")
    parser.add_argument("--points", required=True, nargs="+",
                        help="String to grob point clouds, e.g. \"data/**/*.ply\"")
    parser.add_argument("--num_cameras", type=int, default=120)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--output", type=str, default="data")
    parser.add_argument("--tri_color_light", action='store_true')
    parser.add_argument("--point_lights", action='store_true')
    parser.add_argument("--has_specular", action='store_true')
    parser.add_argument("--min_dist", type=float, default=1.2)
    parser.add_argument("--max_dist", type=float, default=2.2)
    parser.add_argument("--znear", type=float, default=0.1)
    opt, _ = parser.parse_known_args()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    names_and_path = get_names_and_paths(opt)
    for mesh_name, mesh_path in names_and_path:
        output_dir = os.path.join(opt.output, mesh_name +'_variational_light')
        rgb_dir = os.path.join(output_dir, "image")
        mask_dir = os.path.join(output_dir, "mask")
        depth_dir = os.path.join(output_dir, "depth")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(rgb_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)

        # load and normalize mesh
        if os.path.splitext(mesh_path)[1].lower() == ".ply":
            verts, faces = load_ply(mesh_path)
            verts_idx = faces
        elif os.path.splitext(mesh_path)[1].lower() == ".obj":
            verts, faces, aux = load_obj(mesh_path)
            verts_idx = faces.verts_idx
        else:
            raise NotImplementedError

        # # normalize to unit box
        # vert_range = (verts.max(dim=0)[0] - verts.min(dim=0)[0]).max()
        # vert_center = (verts.max(dim=0)[0] + verts.min(dim=0)[0]) / 2
        # verts -= vert_center
        # verts /= vert_range

        # normalize to unit sphere
        vert_center = torch.mean(verts, dim=0)
        verts -= vert_center
        vert_scale = torch.norm(verts, dim=1).max()
        verts /= vert_scale

        save_obj(os.path.join(output_dir, "mesh.obj"),
                 verts=verts, faces=verts_idx)
        textures = Textures(verts_rgb=torch.ones(
            1, verts.shape[0], 3)).to(device=device)
        meshes = Meshes(verts=[verts], faces=[verts_idx],
                        textures=textures).to(device=device)

        # Initialize an OpenGL perspective camera.
        batch_size = 1
        camera_params = {"znear": opt.znear}
        camera_sampler = CameraSampler(opt.num_cameras,
                                       batch_size, distance_range=torch.tensor(
                                           ((opt.min_dist, opt.max_dist),)),  # min distance should be larger than znear+obj_dim
                                       sort_distance=True,
                                       camera_type=FoVPerspectiveCameras,
                                       camera_params=camera_params)


        # Define the settings for rasterization and shading.
        # Refer to raster_points.py for explanations of these parameters.
        raster_settings = RasterizationSettings(
            image_size=opt.image_size,
            blur_radius=0.0,
            faces_per_pixel=5,
            # this setting controls whether naive or coarse-to-fine rasterization is used
            bin_size=None,
            max_faces_per_bin=None  # this setting is for coarse rasterization
        )

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                raster_settings=raster_settings),
            shader=HardFlatShader(device=device)
        )

        if opt.point_lights:
            template_lights = PointLights()
        else:
            template_lights = DirectionalLights()

        # pcl_dict = {'points': pointclouds.points_padded[0].cpu().numpy()}
        data_dict = {"cameras_type": '.'.join([camera_sampler.camera_type.__module__,
                                               camera_sampler.camera_type.__name__]),
                     "cameras_params": camera_params,
                     "lights_type": '.'.join([template_lights.__module__, template_lights.__class__.__name__]),
                     }
        num_points = 20000
        V, V_normal = sample_points_from_meshes(
            meshes, num_samples=num_points, return_normals=True)
        num_points = V.shape[1]
        data_dict['points'] = V[0].cpu().numpy()
        data_dict['normals'] = V_normal[0].cpu().numpy()
        data_dict['colors'] = np.ones_like(
            data_dict['points'], dtype=np.float32)
        data_dict['camera_mat'] = torch.empty(opt.num_cameras, 4, 4)

        # DVR data no projection step, assumes use SfMcamera
        cameras_dict = {}
        pcl_dict = {}
        pcl_dict['points'] = data_dict['points']
        pcl_dict['normals'] = data_dict['normals']
        pcl_dict['colors'] = data_dict['colors']

        idx = 0
        for cams in tqdm(camera_sampler):
            meshes_batch = meshes.extend(batch_size)
            cams = cams.to(device)

            # create tri-color lights and a specular+diffuse shader
            if opt.tri_color_light:
                lights = get_tri_color_lights_for_view(cams,
                    point_lights=opt.point_lights, has_specular=opt.has_specular)
            else:
                lights = get_light_for_view(cams, point_lights=opt.point_lights, has_specular=opt.has_specular)

            assert(type(lights) is type(template_lights))
            lights.to(device=device)

            # renderer function (flat shading)
            fragments = renderer.rasterizer(meshes_batch, cameras=cams)
            images = renderer.shader(
                fragments, meshes_batch, lights=lights, cameras=cams)

            mask = fragments.pix_to_face[..., :1] >= 0
            mask_imgs = mask.to(dtype=torch.uint8) * 255

            # use hard alpha values
            images = torch.cat([images[..., :3], mask.float()], dim=-1)
            dense_depths = cams.zfar.view(-1, 1,
                                          1, 1).clone().expand_as(mask_imgs)
            dense_depths = torch.where(
                mask, fragments.zbuf[..., :1], dense_depths)

            # cameras
            camera_mat = cams.get_projection_transform().get_matrix().cpu()
            world_mat = cams.get_world_to_view_transform().get_matrix().cpu()
            id_mat = np.eye(4)
            # DVR scales x,y and does the projection step manually (/z)
            dvr_camera_mat = eyes(4, camera_mat.shape[0]).to(camera_mat.device)
            dvr_camera_mat[:, :2, :2] = camera_mat[:, :2, :2]
            # dense depth read from rasterizer
            for b in range(images.shape[0]):
                # save camera data
                data_dict['camera_mat'][idx, ...] = world_mat[b]
                data_dict['lights_%d' % idx] = convert_tensor_property_to_value_dict(lights)

                # DVR camera data
                cameras_dict['world_mat_%d' %
                             idx] = world_mat[b].transpose(0, 1)
                cameras_dict['scale_mat_%d' % idx] = id_mat
                cameras_dict['camera_mat_%d' %
                             idx] = dvr_camera_mat[b].transpose(0, 1)
                # save dense depth
                imageio.imwrite(os.path.join(depth_dir, "%06d.exr" % idx),
                                dense_depths[b, ...].cpu())
                # save rgb
                imageio.imwrite(os.path.join(rgb_dir, "%06d.png" % idx),
                                (images[b].cpu().numpy() * 255.0).astype('uint8'),)
                # save mask
                imageio.imwrite(os.path.join(mask_dir, "%06d.png" % idx),
                                mask_imgs[b, ...].cpu())
                idx += 1

        data_dict['camera_mat'] = data_dict['camera_mat'].tolist()
        np.savez(os.path.join(output_dir, "data_dict.npz"),
                 allow_pickle=False, **data_dict)
        np.savez(os.path.join(output_dir, "cameras.npz"),
                 allow_pickle=False, **cameras_dict)
