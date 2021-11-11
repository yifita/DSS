from typing import Union
import torch
import torch.utils.data as data
import os
import imageio
import numpy as np
from .. import logger_py
from ..core.cloud import PointClouds3D
from . import get_class_from_string
from .mathHelper import decompose_to_R_and_t
from pytorch3d.io.obj_io import load_objs_as_meshes
from pytorch3d.structures import Meshes
import pytorch3d.renderer.cameras as cameras


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
                 load_dense_depth=False, n_imgs=None, **kwargs):

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
        if n_imgs is not None:
            self.n_imgs = n_imgs
        else:
            self.n_imgs = len(self.image_files)

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

        self.load_all_images()
        self.load_all_masks()
        self.resolution = self.rgb_images[0].shape[1:]

    def load_all_images(self):
        self.rgb_images = []
        for path in self.image_files:
            rgb = np.array(imageio.imread(path)).astype(np.float32)[..., :3] / 255.0
            rgb = np.transpose(rgb, [2, 0, 1])
            self.rgb_images.append(torch.from_numpy(rgb).float())

    def load_all_masks(self):
        self.object_masks = []
        for path in self.mask_files:
            mask = np.array(imageio.imread(path, pilmode="L")).astype(
            np.bool)[..., None]
            mask = np.transpose(mask, [2, 0, 1])
            self.object_masks.append(torch.from_numpy(mask).float())

    def get_pointclouds(self, num_points=None) -> PointClouds3D:
        """ Returns points, normals and color in object coordinate """
        if hasattr(self, 'point_clouds'):
            if num_points is None or (self.point_clouds.points_packed().shape[0] == num_points):
                return self.point_clouds

        if num_points is None or num_points == self.data_dict['points'].shape[0]:
            points = torch.tensor(self.data_dict["points"]).to(dtype=torch.float32)
            normals = torch.tensor(self.data_dict["normals"]).to(
                dtype=torch.float32)
            colors = torch.tensor(self.data_dict["colors"]).to(dtype=torch.float32)
            self.point_clouds = PointClouds3D([points], [normals], [colors])
        else:
            import pymeshlab
            meshes = self.get_meshes()
            # sample on the mesh with poisson disc sampling
            points_list = []
            normals_list = []
            for i in range(len(meshes)):
                mesh = meshes[i]
                m = pymeshlab.Mesh(mesh.verts_packed().cpu().numpy(), mesh.faces_packed().cpu().numpy())
                # breakpoint()
                ms = pymeshlab.MeshSet()
                ms.add_mesh(m, 'mesh0')
                ms.poisson_disk_sampling(samplenum=num_points, approximategeodesicdistance=True, exactnumflag=True)
                m = ms.current_mesh()
                points = m.vertex_matrix().astype(np.float32)
                normals = m.vertex_normal_matrix().astype(np.float32)
                points_list.append(torch.from_numpy(points))
                normals_list.append(torch.from_numpy(normals))
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

    def get_lights(self, **kwargs):
        """ Returns lights instance """
        Light = get_class_from_string(self.data_dict["lights_type"].item())
        self.lights = Light(**kwargs)
        return self.lights

    def get_cameras(self, camera_mat=None):
        """ Returns a cameras instance """
        if not hasattr(self, 'cameras'):
            Camera = get_class_from_string(self.data_dict["cameras_type"].item())
            self.cameras = Camera(**self.data_dict["cameras_params"].item())
        if camera_mat is not None:
            # set camera R and T
            self.cameras.R, self.cameras.T = decompose_to_R_and_t(camera_mat)
            self.cameras._N = self.cameras.R.shape[0]

        return self.cameras

    def __len__(self):
        ''' Returns the length of the dataset.
        '''
        return min(len(self.image_files), self.n_imgs)

    def __getitem__(self, idx):
        """
        Returns:
            data dict {"img.rgb": rgb (C,H,W),
                       "img.mask": mask (1,H,W),
                       "camera_mat": camera_mat (4,4),
                       "img.depth: depth (1,H,W)}
        """
        idx = idx % self.__len__()
        # load rgb
        rgb = self.rgb_images[idx]
        # load mask
        mask = self.object_masks[idx]

        assert(rgb.shape[-2:] == mask.shape[-2:]
               ), "rgb {} and mask {} images must have the same dimensions.".format(rgb.shape, mask.shape)
        assert(rgb.shape[0] == 3 and rgb.ndim ==
               3), "Invalid RGB image shape {}".format(rgb.shape)
        assert(mask.shape[0] == 1 and mask.ndim ==
               3), "Invalid Mask image shape {}".format(mask.shape)

        # load camera
        camera_mat = np.array(
            self.data_dict['camera_mat'][idx]).astype(np.float32)

        out_data = {"img.rgb": rgb, "img.mask": mask,
                    "camera_mat": camera_mat}

        # load light
        if light_properties := self.data_dict.get('lights_%d' % idx, None):
            out_data['lights'] = {k: np.array(v, dtype=np.float32)[0] for k, v in light_properties.item().items() if isinstance(v, (list, np.ndarray))}

        # load dense depth map
        if self.depth_files is not None:
            depth = np.array(imageio.imread(
                self.depth_files[idx])).astype(np.float32)
            depth = depth.reshape(mask.shape)
            out_data['img.depth'] = depth

        return out_data

class DTUDataset(MVRDataset):
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
                 depth_folder="depth", img_extension="png", mask_extension="png",
                 depth_extension="exr", load_dense_depth=False,
                 n_imgs=None, resolution=(1200, 1600), ignore_image_idx=[],
                 **kwargs):

        image_files = os.listdir(os.path.join(data_dir, img_folder))
        image_files = list(filter(lambda x: os.path.splitext(
            x)[1].lower()[1:] == img_extension, image_files))
        image_files = sorted(image_files)
        image_files = [image_files[i] for i in range(
            len(image_files)) if i not in ignore_image_idx]
        self.image_files = [os.path.join(
            data_dir, img_folder, f) for f in image_files]

        mask_files = os.listdir(os.path.join(data_dir, mask_folder))
        mask_files = list(filter(lambda x: os.path.splitext(
            x)[1].lower()[1:] == mask_extension, mask_files))
        mask_files = sorted(mask_files)
        mask_files = [mask_files[i] for i in range(
            len(mask_files)) if i not in ignore_image_idx]
        self.mask_files = [os.path.join(data_dir, mask_folder, f)
                           for f in mask_files]
        assert(len(self.mask_files) == len(self.image_files))

        camera_file = os.path.join(data_dir, 'cameras.npz')
        self.data_dict = np.load(camera_file)

        self.data_dir = data_dir
        if n_imgs is not None:
            self.n_imgs = n_imgs
        else:
            self.n_imgs = len(self.image_files)

        self.depth_range = kwargs.get('depth_range', [1, 2000])

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

        self.load_all_masks()
        self.load_all_images()
        self.resolution = self.rgb_images[0].shape[1:]

    def load_all_images(self):
        self.rgb_images = []
        for path in self.image_files:
            rgb = np.array(imageio.imread(path)).astype(np.float32)[..., :3] / 255.0
            rgb = np.transpose(rgb, [2, 0, 1])
            self.rgb_images.append(rgb.astype('float32'))

    def load_all_masks(self):
        self.object_masks = []
        for path in self.mask_files:
            mask = np.array(imageio.imread(path, pilmode="L")).astype(
            np.bool)[..., None]
            mask = np.transpose(mask, [2, 0, 1])
            self.object_masks.append(mask)

    def get_pointclouds(self, num_points=None) -> PointClouds3D:
        """ Returns points, normals and color in object coordinate """
        return None

    def get_curvatures(self):
        """ Returns the mesh vertex curvature (smoothed) (N,) """
        return None

    def get_meshes(self) -> Union[None, Meshes]:
        """ Returns ground truth mesh if exist """
        return None

    def get_lights(self, **kwargs):
        """ Returns lights instance """
        return None

    def get_cameras(self, camera_mat=None):
        """ Returns a cameras instance """
        if not hasattr(self, 'cameras'):
            focal_lengths = self.data_dict['camera_mat_0'][(0,1),(0,1)].reshape(1,2)
            principal_point = self.data_dict['camera_mat_0'][(0,1),(2,2)].reshape(1,2)
            self.cameras = cameras.PerspectiveCameras(focal_length=-focal_lengths, principal_point=-principal_point)
            self.cameras.znear = float(self.depth_range[0])
            self.cameras.zfar = float(self.depth_range[1])
        if camera_mat is not None:
            # set camera R and T
            self.cameras.R, self.cameras.T = decompose_to_R_and_t(camera_mat)
            self.cameras._N = self.cameras.R.shape[0]
        return self.cameras

    def get_scale_mat(self):
        return self.data_dict['scale_mat_0']

    def _get_idx(self, idx):
        file_idx = os.path.basename(self.image_files[idx])[:-4]
        return int(file_idx)

    def __len__(self):
        ''' Returns the length of the dataset.
        '''
        return min(len(self.image_files), self.n_imgs)

    def __getitem__(self, idx):
        """
        Returns:
            data dict {"img.rgb": rgb (C,H,W),
                       "img.mask": mask (1,H,W),
                       "camera_mat": camera_mat (4,4),
                       "img.depth: depth (1,H,W)}
        """
        file_idx = self._get_idx(idx)
        # load rgb
        rgb = self.rgb_images[idx]
        # load mask
        mask = self.object_masks[idx]

        assert(rgb.shape[-2:] == mask.shape[-2:]
               ), "rgb {} and mask {} images must have the same dimensions.".format(rgb.shape, mask.shape)
        assert(rgb.shape[0] == 3 and rgb.ndim ==
               3), "Invalid RGB image shape {}".format(rgb.shape)
        assert(mask.shape[0] == 1 and mask.ndim ==
               3), "Invalid Mask image shape {}".format(mask.shape)

        # load camera
        camera_mat = (self.data_dict['scale_mat_%d' % file_idx].T @ self.data_dict['world_mat_%d' % file_idx].T).astype('float32')

        out_data = {"img.rgb": rgb, "img.mask": mask,
                    "camera_mat": camera_mat}

        # load dense depth map
        if self.depth_files is not None:
            depth = np.array(imageio.imread(
                self.depth_files[idx])).astype(np.float32)
            depth = depth.reshape(mask.shape)
            out_data['img.depth'] = depth

        return out_data
