from typing import Union
import torch
import torch.utils.data as data
import os
import imageio
import numpy as np
from DSS import get_logger
from DSS.core.cloud import PointClouds3D
from DSS.utils import get_class_from_string
from pytorch3d.io.obj_io import load_objs_as_meshes
from pytorch3d.structures import Meshes
import pytorch3d.renderer.cameras as cameras

logger_py = get_logger(__name__)


class MVRDataset(data.Dataset):
    """
    Dataset for MVR
    loads RGB, camera_mat, mask, points, lights, cameras

    Attributes:
        image_files
        mask_files
        data_dict
        point_clouds: point positions N,3 in object coordinates
        lights:
        cameras:
    """

    def __init__(self, data_dir, img_folder="image", mask_folder="mask",
                 depth_folder="depth", data_dict="data_dict.npz",
                 img_extension="png", mask_extension="png", depth_extension="exr",
                 load_dense_depth=False, **kwargs):

        image_files = os.listdir(os.path.join(data_dir, img_folder))
        image_files = list(filter(lambda x: os.path.splitext(
            x)[1].lower()[1:] == img_extension, image_files))
        image_files = sorted(image_files)
        self.image_files = [os.path.join(
            data_dir, img_folder, f) for f in image_files]

        mask_files = os.listdir(os.path.join(data_dir, mask_folder))
        mask_files = list(filter(lambda x: os.path.splitext(
            x)[1].lower()[1:] == mask_extension, mask_files))
        mask_files = sorted(mask_files)
        self.mask_files = [os.path.join(data_dir, mask_folder, f)
                           for f in mask_files]

        self.data_dict = np.load(os.path.join(
            data_dir, data_dict), allow_pickle=True)

        self.data_dir = data_dir

        if 'camera_mat' not in self.data_dict:
            logger_py.error("data_dict must contain camera_mat!")

        # data_dict contains camera_mat_%d, points, normals, colors, lights
        # check image, mask and camera_mat has the same length
        if(len(set(len(_) for _ in (self.image_files,
                                    self.mask_files, self.data_dict['camera_mat']))) > 1):
            logger_py.error(
                "Found unequal number of image, mask and camera matrices! ({}, {}, {})".format(
                    len(self.image_files), len(self.mask_files), len(self.data_dict['camera_mat']))
            )
            raise ValueError

        if load_dense_depth:
            depth_files = os.listdir(os.path.join(data_dir, depth_folder))
            depth_files = list(filter(lambda x: os.path.splitext(
                x)[1].lower()[1:] == depth_extension, depth_files))
            depth_files = sorted(depth_files)
            self.depth_files = [os.path.join(
                data_dir, depth_folder, f) for f in depth_files]
            if len(self.depth_files) != len(self):
                logger_py.error("Found invalid number of dense depth maps")
                raise ValueError
        else:
            self.depth_files = None

        self.load_all_files()

    def get_pointclouds(self, num_points=None) -> PointClouds3D:
        """ Returns points, normals and color in object coordinate """
        if hasattr(self, 'point_clouds'):
            if num_points is None and (self.point_clouds.num_points_per_cloud()== num_points).all():
                return self.point_clouds

        if num_points is None or num_points == self.data_dict['points'].shape[0]:
            points = torch.tensor(self.data_dict["points"]).to(dtype=torch.float32)
            normals = torch.tensor(self.data_dict["normals"]).to(
                dtype=torch.float32)
            colors = torch.tensor(self.data_dict["colors"]).to(dtype=torch.float32)
            self.point_clouds = PointClouds3D([points], [normals], [colors])
        else:
            import point_cloud_utils as pcu
            meshes = self.get_meshes()
            # sample on the mesh with poisson disc sampling
            points_list = []
            normals_list = []
            for i in range(len(meshes)):
                mesh = meshes[i]
                points, normals = pcu.sample_mesh_poisson_disk(
                    mesh.verts_packed().cpu().numpy(), mesh.faces_packed().cpu().numpy(),
                    mesh.verts_normals_packed().cpu().numpy(), num_points, use_geodesic_distance=True)
                points = torch.from_numpy(points)
                normals = torch.from_numpy(normals)
                points_list.append(points)
                normals_list.append(normals)
            self.point_clouds = PointClouds3D(points_list, normals_list)

        return self.point_clouds

    def get_meshes(self) -> Union[None, Meshes]:
        """ Returns ground truth mesh if exist """
        if hasattr(self, 'meshes'):
            return self.meshes

        mesh_file = os.path.join(self.data_dir, 'mesh.obj')
        if os.path.isfile(mesh_file):
            self.meshes = load_objs_as_meshes([mesh_file])
            return self.meshes
        else:
            return None

    def get_lights(self):
        """ Returns lights instance """
        Light = get_class_from_string(self.data_dict["lights_type"].item())
        self.lights = Light(**self.data_dict['lights'].item())
        return self.lights

    def get_cameras(self):
        """ Returns a cameras instance """
        Camera = get_class_from_string(self.data_dict["cameras_type"].item())
        self.cameras = Camera(**self.data_dict["cameras_params"].item())
        return self.cameras

    def load_all_files(self):
        """ load all data into memory to save time"""
        assert(len(self.image_files) == len(self.mask_files))
        self.item_list = []
        for i in range(len(self.image_files)):
            rgb = imageio.imread(self.image_files[i]).astype(np.float32)[..., :3] / 255.0
            mask = imageio.imread(self.mask_files[i], pilmode="L").astype(np.bool)[..., None]
            assert(rgb.shape[:2] == mask.shape[:2]
                   ), "rgb {} and mask {} images must have the same dimensions.".format(rgb.shape, mask.shape)
            assert(rgb.shape[2] == 3 and rgb.ndim ==
                   3), "Invalid RGB image shape {}".format(rgb.shape)
            assert(mask.shape[2] == 1 and mask.ndim ==
                   3), "Invalid Mask image shape {}".format(mask.shape)
            # transpose only changes the strides, doesn't touch the actual array
            rgb = np.ascontiguousarray(np.transpose(rgb, [2, 0, 1]))
            mask = np.ascontiguousarray(np.transpose(mask, [2, 0, 1]))
            camera_mat = np.array(self.data_dict['camera_mat'][i]).astype(np.float32)
            out_data = {"img.rgb": rgb, "img.mask": mask, "camera_mat": camera_mat}

            # load dense depth map
            if self.depth_files is not None:
                depth = np.array(imageio.imread(self.depth_files[i])).astype(np.float32)
                depth = depth.reshape(mask.shape)
                out_data['img.depth'] = depth

            self.item_list.append(out_data)
        return

    def __len__(self):
        ''' Returns the length of the dataset.
        '''
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Returns:
            data dict {"img.rgb": rgb (C,H,W),
                       "img.mask": mask (1,H,W),
                       "camera_mat": camera_mat (4,4),
                       "img.depth: depth (1,H,W)}
        """
        # use files loaded in memory
        if self.item_list is not None:
            return self.item_list[idx]
        # load rgb
        rgb = np.array(imageio.imread(
            self.image_files[idx])).astype(np.float32)[..., :3] / 255.0

        # load mask
        mask = np.array(imageio.imread(self.mask_files[idx], pilmode="L")).astype(
            np.bool)[..., None]
        assert(rgb.shape[:2] == mask.shape[:2]
               ), "rgb {} and mask {} images must have the same dimensions.".format(rgb.shape, mask.shape)
        assert(rgb.shape[2] == 3 and rgb.ndim ==
               3), "Invalid RGB image shape {}".format(rgb.shape)
        assert(mask.shape[2] == 1 and mask.ndim ==
               3), "Invalid Mask image shape {}".format(mask.shape)

        rgb = np.transpose(rgb, [2, 0, 1])
        mask = np.transpose(mask, [2, 0, 1])
        # load camera
        camera_mat = np.array(
            self.data_dict['camera_mat'][idx]).astype(np.float32)

        out_data = {"img.rgb": rgb, "img.mask": mask,
                    "camera_mat": camera_mat}

        # load dense depth map
        if self.depth_files is not None:
            depth = np.array(imageio.imread(
                self.depth_files[idx])).astype(np.float32)
            depth = depth.reshape(mask.shape)
            out_data['img.depth'] = depth

        return out_data
