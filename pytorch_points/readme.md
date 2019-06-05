## pytorch_points - a python library for learning point clouds on pytorch

This library implements and collects some useful functions commonly used for point cloud processing.
Many of the functions are adapted from the code base of amazing researchers. Thank you! (see [related repositories](#related_repositories) for a comprehensive list.)

### structures

- `_ext`: cuda extensions, 
  - losses: "chamfer distance"
  - sampling: "farthest_sampling", "ball_query"
  - linalg: "batch_SVD"     
- `network`: common pytorch layers and operations for point cloud processing
  - operations: "group_KNN", "batch_normals"
  - layers
- `utils`: utility functions including functions for point cloud in/output etc
  - pc_utils

### install
```bash
# requires faiss-gpu and pytorch>=1.0
conda config --add_channels pytorch conda-forge
conda create --name pytorch-1.0 --file requirements.txt

python setup.py install
```

### related repositories:
- [AtlasNet](https://github.com/ThibaultGROUEIX/AtlasNet): Thanks Thibault for your AtlasNet!!! 
- [Pointnet-Pytorch](https://github.com/erikwijmans/Pointnet2_PyTorch): pytorch implementation of pointnet.
- [SO-Net](https://github.com/lijx10/SO-Net): CVPR 2018 spotlight paper
