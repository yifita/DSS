from setuptools import setup, find_packages
import unittest
from typing import List

from torch.utils.cpp_extension import BuildExtension, CUDAExtension
print(find_packages())
CUDA_FLAGS = []  # type: List[str]

# The modules are as small as possible to reduce incremental building time during development
ext_modules = [
    CUDAExtension('DSS.cuda.rasterize_forward', [
        'DSS/cuda/rasterize_forward_cuda.cpp',
        'DSS/cuda/rasterize_forward_cuda_kernel.cu'
    ]),
    CUDAExtension('DSS.cuda.rasterize_backward', [
        'DSS/cuda/rasterize_backward_cuda.cpp',
        'DSS/cuda/rasterize_backward_cuda_kernel.cu'
    ]),
]

INSTALL_REQUIREMENTS = ['numpy', 'torch', 'pypng', 'plyfile']

setup(
    name='DSS',
    description='Differentiable Surface Splatter',
    author='Yifan Wang and Felice Serena',
    author_email='yifan.wang@inf.ethz.ch and felice@serena-mueller.ch',
    license='MIT License',
    version='0.9',
    packages=['DSS'],
    install_requires=INSTALL_REQUIREMENTS,
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension}
)
