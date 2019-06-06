## DSS: Differentiable Surface Splatting
![bunny](images/teaser.png)

### Demo
1. clone
````bash
git clone --recursive 
cd DSS
````
2. install prequisitories. Our code uses python3.7, the installation instruction requires the latest anaconda.
````bash
# update conda
conda update -n base -c defaults conda
# install requirements
conda config --add channels pytorch
conda config --add channels conda-forge
conda create --name DSS --file requirements.txt
````
3. compile cuda library
````bash
cd pytorch_points
python setup.py install
cd ..
python setup.py install # or python setup develop for local compilation
````
4. download data
````bash
# downloads
cd trained_models
./download_data.sh
````
5. experiments
````bash
# inverse rendering test: optimize point positions and normals to transform sphere to teapot
python learn_shape_from_target.py example_data/scenes/sphere.json -t example_data/scenes/teapot.json

# denoising
python learn_image_filter.py example_data/scenes/pix2pix_denoise.json --cloud example_data/pointclouds/noisy03_points/a72-seated_jew_aligned_pca.ply

python learn_image_filter.py example_data/scenes/pix2pix_denoise_noise01.json --cloud example_data/noisy1_points/a72-seated_jew_aligned_pca.ply
````

### Results

#### 2D grid to teapot
![2DTeapot](images/teapot_2D.gif)

#### 3D sphere to teapot
![3DTeapot](images/teapot_3D.gif)

#### Denoising
![denoise_0.3noise](images/seated_all.png)