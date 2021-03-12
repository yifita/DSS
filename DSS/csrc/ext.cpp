#include <torch/extension.h>
#include "rasterize_points.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // module docstring
  m.doc() = "pybind11 compute_visibility_maps plugin";
  m.def("splat_points", &RasterizePoints);
  m.def("_splat_points_naive", &RasterizePointsNaive);
  m.def("_splat_points_occ_backward", &RasterizePointsOccBackward);
  m.def("_rasterize_coarse", &RasterizePointsCoarse);
  m.def("_rasterize_fine", &RasterizePointsFine);
#ifdef WITH_CUDA
  m.def("_splat_points_occ_fast_cuda_backward", &RasterizePointsBackwardCudaFast);
#endif
  m.def("_splat_points_occ_backward", &RasterizePointsOccBackward);
  m.def("_backward_zbuf", &RasterizeZbufBackward);
}
