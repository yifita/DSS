#include "macros.hpp"
#include <pybind11/pybind11.h>
#include <torch/extension.h>

std::vector<at::Tensor>
visibility_backward_cuda(const double focalLength, const double mergeThres,
                         const bool considerZ, const int localHeight,
                         const int localWidth,
                         const at::Tensor &colorGrads,    // BxHxWx3
                         const at::Tensor &pointIdxMap,   // BxHxWxtopK
                         const at::Tensor &rhoMap,        // BxHxWxtopK
                         const at::Tensor &wsMap,         // BxHxWxtopK
                         const at::Tensor &depthMap,      // BxHxWxtopK
                         const at::Tensor &isBehind,      // BxHxWxtopK
                         const at::Tensor &pixelValues,   // BxHxWx3
                         const at::Tensor &boundingBoxes, // BxNx4
                         const at::Tensor &projPoints,    // BxNx[2or3]
                         const at::Tensor &pointColors,   // BxNx3
                         const at::Tensor &depthValues,   // BxNx1
                         const at::Tensor &rhoValues,     // BxNx1
                         at::Tensor &dIdp, at::Tensor &dIdz);

std::vector<at::Tensor>
visibility_backward(const double focalLength, const double mergeThres,
                    const bool considerZ, const int localHeight,
                    const int localWidth,
                    const at::Tensor &colorGrads,    // BxHxWx3
                    const at::Tensor &pointIdxMap,   // BxHxWxtopK
                    const at::Tensor &rhoMap,        // BxHxWxtopK
                    const at::Tensor &wsMap,         // BxHxWxtopK
                    const at::Tensor &depthMap,      // BxHxWxtopK
                    const at::Tensor &isBehind,      // BxHxWxtopK
                    const at::Tensor &pixelValues,   // BxHxWx3
                    const at::Tensor &boundingBoxes, // BxNx4
                    const at::Tensor &projPoints,    // BxNx[2or3]
                    const at::Tensor &pointColors,   // BxNx3
                    const at::Tensor &depthValues,   // BxNx1
                    const at::Tensor &rhoValues,     // BxNx1
                    at::Tensor &dIdp, at::Tensor &dIdz) {

  CHECK_INPUT(colorGrads);
  CHECK_INPUT(pointIdxMap);
  CHECK_INPUT(rhoMap);
  CHECK_INPUT(wsMap);
  CHECK_INPUT(depthMap);
  CHECK_INPUT(isBehind);
  CHECK_INPUT(pixelValues);
  CHECK_INPUT(boundingBoxes);
  CHECK_INPUT(projPoints);
  CHECK_INPUT(pointColors);
  CHECK_INPUT(depthValues);
  CHECK_INPUT(rhoValues);
  CHECK_INPUT(dIdp);
  CHECK_INPUT(dIdz);

  CHECK_CUDA(colorGrads);
  CHECK_CUDA(pointIdxMap);
  CHECK_CUDA(rhoMap);
  CHECK_CUDA(wsMap);
  CHECK_CUDA(depthMap);
  CHECK_CUDA(isBehind);
  CHECK_CUDA(pixelValues);
  CHECK_CUDA(boundingBoxes);
  CHECK_CUDA(projPoints);
  CHECK_CUDA(pointColors);
  CHECK_CUDA(depthValues);
  CHECK_CUDA(rhoValues);
  CHECK_CUDA(dIdp);
  CHECK_CUDA(dIdz);

  return visibility_backward_cuda(
      focalLength, mergeThres, considerZ, localHeight, localWidth, colorGrads,
      pointIdxMap, rhoMap, wsMap, depthMap, isBehind, pixelValues,
      boundingBoxes, projPoints, pointColors, depthValues, rhoValues, dIdp,
      dIdz);
}

PYBIND11_MODULE(rasterize_backward, m) {
  // module docstring
  m.doc() = "splatter rasterize backward";
  m.def("visibility_backward", &visibility_backward,
        "Performs backward pass to propagate the gradient back to the "
        "projected point coordinates and depth.");
}
