import os.path
import random
from data.base_dataset import BaseDataset, get_params, get_transform
import torchvision.transforms as transforms
from data.image_folder import make_dataset
from PIL import Image
from os import listdir
import numpy as np
from util.util import is_image_file, load_img, save_img_tensor, tensor2im, save_image
import torch
import glob
from os import walk


def npy_loader(path):
    sample = torch.from_numpy(np.load(path))
    return sample


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        # self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory

        # self.dir_A = os.path.join(opt.dataroot, 'trainA')
        # self.dir_B = os.path.join(opt.dataroot, 'trainB')
        self.dir_A = os.path.join(opt.dataroot, 'input_rendered')
        self.dir_B = os.path.join(opt.dataroot, 'target_rendered')

        # self.image_filenames = [x for x in listdir(self.dir_B) if is_image_file(x)]

        self.image_filenames = []
        # glob.glob(self.dir_B + "/*.npy")

        for (root, dirs, files) in walk(self.dir_B):
            for filename in files:
                if filename.endswith(('.npy')):
                    self.image_filenames.append(os.path.join(root, filename))
                    # print(os.path.join(root, filename))

        # self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # get image paths
        # self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))

        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index

        # AB_path = self.AB_paths[index]
        # AB = Image.open(AB_path).convert('RGB')
        # # split AB image into A and B
        # w, h = AB.size
        # w2 = int(w / 2)
        # A = AB.crop((0, 0, w2, h))
        # B = AB.crop((w2, 0, w, h))

        # A = Image.open(os.path.join(self.dir_A, self.image_filenames[index])).convert('RGB')
        # B = Image.open(os.path.join(self.dir_B, self.image_filenames[index])).convert('RGB')
        #
        # transform_params = get_params(self.opt, A.size)
        # A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        # B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        # print(self.image_filenames[index])
        # print(self.image_filenames[index].replace('_target', '_input').replace('target_rendered', 'input_rendered'))

        # A = np.load(os.path.join(self.dir_A, self.image_filenames[index].replace('_target', '_input')))
        # B = np.load(os.path.join(self.dir_B, self.image_filenames[index]))

        A = np.load(self.image_filenames[index].replace('_target', '_input').replace('target_rendered', 'input_rendered'))
        B = np.load(self.image_filenames[index])

        # print(A.shape)
        # print(A)
        # apply the same transform to both A and B

        A_transform = get_transform(self.opt)
        B_transform = get_transform(self.opt)

        ####
        import pdb
        pdb.set_trace()
        A = A_transform(A)
        B = B_transform(B)

        # save_img_tensor(A, './testA.png')
        # save_img_tensor(B, './testB.png')

        return {'A': A, 'B': B}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.image_filenames)
