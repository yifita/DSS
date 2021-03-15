from setuptools import setup, find_packages
from typing import List
import os

import torch
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension, CppExtension

extra_compile_args = {"cxx": ["-std=c++14"]}
define_macros = []

force_cuda = os.getenv("FORCE_CUDA", "0") == "1"
if (torch.cuda.is_available() and CUDA_HOME is not None) or force_cuda:
    extension = CUDAExtension
    # sources += source_cuda
    define_macros += [("WITH_CUDA", None)]
    nvcc_args = [
        "-DCUDA_HAS_FP16=1",
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ]
    nvcc_flags_env = os.getenv("NVCC_FLAGS", "")
    if nvcc_flags_env != "":
        nvcc_args.extend(nvcc_flags_env.split(" "))

    # It's better if pytorch can do this by default ..
    CC = os.environ.get("CC", None)
    if CC is not None:
        CC_arg = "-ccbin={}".format(CC)
        if CC_arg not in nvcc_args:
            if any(arg.startswith("-ccbin") for arg in nvcc_args):
                raise ValueError("Inconsistent ccbins")
            nvcc_args.append(CC_arg)

    extra_compile_args["nvcc"] = nvcc_args
else:
    print('Cuda is not available!')

ext_modules = [
    CppExtension('DSS._C', [
        'DSS/csrc/ext.cpp',
        'DSS/csrc/rasterize_points_cpu.cpp',
    ])
]
include_dirs = torch.utils.cpp_extension.include_paths()
ext_modules += [
    CUDAExtension('DSS._C', [
        'DSS/csrc/ext.cpp',
        'DSS/csrc/rasterize_points.cu',
        'DSS/csrc/rasterize_points_backward.cu',
        'DSS/csrc/rasterize_points_cpu.cpp',
    ],
        include_dirs=['DSS/csrc'],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args
    )
]

INSTALL_REQUIREMENTS = ['numpy', 'torch', 'plyfile', 'pytorch3d', 'imageio', 'frnn']

setup(
    name='DSS',
    description='Differentiable Surface Splatter',
    author='Yifan Wang, Lixin Xue and Felice Serena',
    packages=find_packages(exclude=('tests')),
    license='MIT License',
    version='2.0',
    install_requires=INSTALL_REQUIREMENTS,
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension}
)
