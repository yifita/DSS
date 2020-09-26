#include <torch/extension.h>
#include <ATen/ATen.h>
#include <pybind11/pybind11.h>
#include "macros.hpp"

// data BxNxC, indices BxHxWxK value (0~N-1), output BxHxWxKxC
at::Tensor gather_maps_cuda(const at::Tensor &data, const at::Tensor &indices,
                            const double defaultValue);
at::Tensor scatter_maps_cuda(const int64_t numPoint, const at::Tensor &data,
                             const at::Tensor &indices);
// data BxHxWxK outputs BxnumPointsxK, indices BxHxWxk
at::Tensor scatter_maps(const int64_t numPoint, const at::Tensor &data,
                        const at::Tensor &indices) {
  TORCH_CHECK(indices.scalar_type() == at::ScalarType::Long,
           "indices must be a LongTensor");
  return scatter_maps_cuda(numPoint, data, indices);
}
at::Tensor guided_scatter_maps_cuda(const int64_t numPoint,
                                    const at::Tensor &src,
                                    const at::Tensor &indices,
                                    const at::Tensor &boundingBoxes);

at::Tensor guided_scatter_maps(const int64_t numPoint, const at::Tensor &data,
                               const at::Tensor &indices,
                               const at::Tensor &boundingBoxes) {
  TORCH_CHECK(indices.scalar_type() == at::ScalarType::Long,
           "indices must be a LongTensor");
  TORCH_CHECK(boundingBoxes.size(-1) == 4 && boundingBoxes.size(1) == numPoint &&
               boundingBoxes.dim() == 3,
           "boundingBoxes must be a (B, N, 4) tensor");
  return guided_scatter_maps_cuda(numPoint, data, indices, boundingBoxes);
}
at::Tensor gather_maps(const at::Tensor &data, const at::Tensor &indices,
                       const double defaultValue) {
  TORCH_CHECK(data.dim() == 3, "data must a 3 dimensional tensor");
  CHECK_INPUT(data);
  CHECK_INPUT(indices);
  return gather_maps_cuda(data, indices, defaultValue);
}
void compute_visiblity_maps_cuda(const at::Tensor &boundingBoxes,
                                 const at::Tensor &inPlane,
                                 at::Tensor &pointIdxMap,
                                 at::Tensor &bbPositionMap,
                                 at::Tensor &depthMap);

void compute_visibility_maps(const at::Tensor &boundingBoxes,
                             const at::Tensor &inPlane, at::Tensor &pointIdxMap,
                             at::Tensor &bbPositionMap, at::Tensor &depthMap) {
  const int numPoint = inPlane.size(1);
  CHECK_INPUT(boundingBoxes);
  CHECK_INPUT(inPlane);
  CHECK_INPUT(pointIdxMap);
  CHECK_INPUT(bbPositionMap);
  CHECK_INPUT(depthMap);
  TORCH_CHECK(depthMap.dim() == 4);
  TORCH_CHECK(pointIdxMap.dim() == 4);
  TORCH_CHECK(bbPositionMap.dim() == 5);
  TORCH_CHECK(inPlane.dim() == 5);

  TORCH_CHECK(bbPositionMap.size(-1) == 2);
  TORCH_CHECK(pointIdxMap.size(0) == bbPositionMap.size(0) &&
               pointIdxMap.size(1) == bbPositionMap.size(1) &&
               pointIdxMap.size(2) == bbPositionMap.size(2) &&
               pointIdxMap.size(3) == bbPositionMap.size(3),
           "pointIdxMap and bbPositionMap should be (b, h, w, k) and (b, h, w, "
           "k, 2)");
  TORCH_CHECK(boundingBoxes.dim() == 3 && boundingBoxes.size(2) == 2 &&
               boundingBoxes.size(1) == numPoint,
           "boundingBoxes must be (B, P, 2)");
  TORCH_CHECK(depthMap.size(0) == pointIdxMap.size(0) &&
               depthMap.size(1) == pointIdxMap.size(1) &&
               depthMap.size(2) == pointIdxMap.size(2) &&
               depthMap.size(3) == pointIdxMap.size(3),
           "depthMap must be (B, H, W, topK)");
  TORCH_CHECK(inPlane.size(0) == pointIdxMap.size(0) &&
               inPlane.size(1) == numPoint && inPlane.size(4) == 3,
           "inPlane must be (B, N, h, w, 3)");
  return compute_visiblity_maps_cuda(boundingBoxes, inPlane, pointIdxMap,
                                     bbPositionMap, depthMap);
}

PYBIND11_MODULE(rasterize_forward, m) {
  // module docstring
  m.doc() = "pybind11 compute_visibility_maps plugin";
  m.def("_compute_visibility_maps", &compute_visibility_maps, "");
  m.def("_gather_maps", &gather_maps, "");
  m.def("_scatter_maps", &scatter_maps, "");
  m.def("_guided_scatter_maps", &guided_scatter_maps, "");
}
