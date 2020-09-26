#include <torch/extension.h>
#include "rasterize_points.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // module docstring
  m.doc() = "pybind11 compute_visibility_maps plugin";
  m.def("splat_points", &RasterizePoints);
  m.def("_splat_points_naive", &RasterizePointsNaive);
  m.def("_splat_points_weights_backward", &RasterizePointsWeightsBackward);
  m.def("_splat_points_occ_rbf_backward", &RasterizePointsOccRBFBackward);
  m.def("_splat_points_occ_backward", &RasterizePointsOccBackward);
  // m.def("_splat_disc_points_occ_backward", &RasterizeDiscPointsOccBackward);
}
