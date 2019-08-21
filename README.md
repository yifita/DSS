## DSS: Differentiable Surface Splatting
![bunny](images/teaser.png)

### installation
1. clone
````bash
git clone --recursive https://gitlab.inf.ethz.ch/OU-SORKINE/dss.git
cd dss
````
2. install prequisitories. Our code uses python3.7, the installation instruction requires the latest anaconda.
````bash
# install cuda, cudnn, nccl from nvidia
# we tested with cuda 10.1, cudnn 7.5, nccl 1.3.5
# update conda
conda update -n base -c defaults conda
# install requirements
conda config --add channels pytorch
conda config --add channels conda-forge
conda create --name DSS --file requirements.txt
conda activate DSS
# plyfile package is not on conda
pip install plyfile
````
3. compile cuda library
````bash
cd pytorch_points
python setup.py install
cd ..
python setup.py develop
````
### Demos

#### inverse rendering - shape deformation
##### 2D grid to teapot
```bash
python learn_shape_from_target.py -t example_data/scenes/good_teapot.json
```
![teapot_2D](images/2D_teapot.gif)
##### sphere to teapot
````bash
# inverse rendering test: optimize point positions and normals to transform sphere to teapot
python learn_shape_from_target.py example_data/scenes/sphere.json -t example_data/scenes/teapot.json
````
![teapot](images/teapot_3D.gif)
##### cube to yoga
```bash
python learn_shape_from_target.py example_data/scenes/cube_20k.json  -t example_data/scenes/yoga6.json --name yoga6_z_paper_1
````
![yoga1](images/yoga6.gif)
```bash
python finetune_shape.py learn_examples/yoga6_z_paper_1/final_scene.json  -t example_data/scenes/yoga6.json --name yoga6_z_paper_1_1
```
![yoga2](images/yoga6-1.gif)

#### denoising
```bash
cd trained_models
# unix system can run this command directly
./download_data.sh

# 0.3% noise
python learn_image_filter.py example_data/scenes/pix2pix_denoise.json --cloud example_data/pointclouds/noisy03_points/a72-seated_jew_aligned_pca.ply

# 1.0% noise
python learn_image_filter.py example_data/scenes/pix2pix_denoise_noise01.json --cloud example_data/noisy1_points/a72-seated_jew_aligned_pca.ply
```
![denoise_0.3noise](images/seated_all.png)

#### other functions
##### render object 360 degree
```bash
python sequences.py example_data/scenes/teapot.json --points example_data/pointclouds/teapot_normal_dense.ply --width 512 --height 512 --output renders/teapot_360
# then you can create gif. on ubuntu this can be done with
convert -dispose 2 -delay 10 renders/teapot_360/*.png renders/teapot_360/animation.gif
```
![teapot_sequence](images/teapot_sequence.gif)