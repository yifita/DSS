# DSS: Differentiable Surface Splatting
| [Paper PDF](https://igl.ethz.ch/projects/differentiable-surface-splatting/DSS-2019-SA-Yifan-etal.pdf) | [Project page](https://igl.ethz.ch/projects/differentiable-surface-splatting/) |
| ----------------------------------------- | ------------------------------------------------------------------------------ |

![bunny](images/teaser.png)

code for paper Differentiable Surface Splatting for Point-based Geometry Processing

```diff
+ Mar 2021: major updates tag 2.0. Code shares the same structure as pytorch3d. See the Changelog
```

## installation
1. clone
````bash
git clone --recursive https://github.com/yifita/DSS.git
cd dss
````
2. install prequisitories. Our code uses python3.8, pytorch 1.6.1, pytorch3d. the installation instruction requires the latest anaconda.
````bash
# install cuda, cudnn, nccl from nvidia
# we tested with cuda 11.1, cudnn 7.5, nccl 1.3.5
# update conda
conda update -n base -c defaults conda
# install requirements
conda config --add channels pytorch
conda config --add channels conda-forge
conda create -n pytorch3d python=3.8
conda activate pytorch3d
conda install -c pytorch pytorch=1.6.0 torchvision cudatoolkit=10.2
conda install -c conda-forge -c fvcore -c iopath fvcore iopath
conda install --file requirements.txt
# plyfile package is not on conda
pip install plyfile
````
3. compile cuda libraries in external
````bash
cd external
cd prefix_sum
python setup.py install
cd ..
cd FRNN
python setup.py install
cd ..
cd torch-batch-svd
python setup.py install
cd ..
python setup.py develop
````
## Demos
### inverse rendering - shape deformation

```bash
python train_mvr.py --config configs/dss_proj.yml
```

### denoising
```bash
cd trained_models
# unix system can run this command directly
./download_data.sh
```
```bash
# 0.3% noise
python learn_image_filter.py example_data/scenes/pix2pix_denoise.json --cloud example_data/pointclouds/noisy03_points/a72-seated_jew_aligned_pca.ply
````
![denoise_0.3noise](images/seated_all.png)

```bash
# 1.0% noise
python learn_image_filter.py example_data/scenes/pix2pix_denoise_noise01.json --cloud example_data/noisy1_points/a72-seated_jew_aligned_pca.ply
```


![denoise_1noise](images/armadillo_2_all.png)

### other functions
#### render object 360 degree
```bash
python sequences.py example_data/scenes/teapot.json --points example_data/pointclouds/teapot_normal_dense.ply --width 512 --height 512 --output renders/teapot_360
# then you can create gif. on ubuntu this can be done with
convert -dispose 2 -delay 10 renders/teapot_360/*.png renders/teapot_360/animation.gif
```
![teapot_sequence](images/teapot_sequence.gif)

## video
[![accompanying video](images/video-thumb.png)](https://youtu.be/MIu59GiJZ2s "Accompanying video")
<!-- [Accompanying video](https://youtu.be/Q8iTkmIky0o) -->

## cite
Please cite us if you find the code useful!
```
@article{Yifan:DSS:2019,
author = {Yifan, Wang and
          Serena, Felice and
          Wu, Shihao and
          {\"{O}}ztireli, Cengiz and
         Sorkine{-}Hornung, Olga},
title = {Differentiable Surface Splatting for Point-based Geometry Processing},
journal = {ACM Transactions on Graphics (proceedings of ACM SIGGRAPH ASIA)},
volume = {38},
number = {6},
year = {2019},
}
```

## Acknowledgement
We would like to thank Federico Danieli for the insightful discussion, Phillipp Herholz for the timely feedack, Romann Weber for the video voice-over and Derek Liu for the help during the rebuttal.
This work was supported in part by gifts from Adobe, Facebook and Snap, Inc.
