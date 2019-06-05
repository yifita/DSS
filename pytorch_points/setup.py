from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

print(find_packages())

INSTALL_REQUIREMENTS = ['numpy', 'torch', 'plyfile', 'matplotlib']
setup(
    name='pytorch_points',
    description="pytorch extension for point cloud processing",
    author='Yifan Wang',
    author_email="yifan.wang@inf.ethz.ch",
    version='0.9',
    install_requires=INSTALL_REQUIREMENTS,
    packages=find_packages("."),
    ext_package="pytorch_points._ext",
    python_requires=">3.6",
    ext_modules=[
        CUDAExtension('linalg', [
            'pytorch_points/_ext/torch_batch_svd.cpp', ],
            libraries=["cusolver", "cublas"],
            extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']},
        ),
        CUDAExtension('losses', [
            'pytorch_points/_ext/nmdistance_cuda.cu', 'pytorch_points/_ext/nmdistance.cpp'],
            extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']},
        ),
        CUDAExtension('sampling', [
            'pytorch_points/_ext/sampling.cpp',
            'pytorch_points/_ext/sampling_cuda.cu', ],
            extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']},
        )
    ],

    cmdclass={
        'build_ext': BuildExtension
    })
