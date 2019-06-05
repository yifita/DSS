#include "utils.h"
#include <torch/extension.h>
#include <iostream>

// CUDA forward declarations

at::Tensor furthest_sampling_cuda_forward(
    const int m, 
    const int seedIdx,
    at::Tensor input,
    at::Tensor temp,
    at::Tensor idx);

at::Tensor gather_points_cuda_forward(int b, int c, int n, int npoints,
                                      at::Tensor points, at::Tensor idx,
                                      at::Tensor out);

at::Tensor gather_points_cuda_backward(int b, int c, int n, int npoints,
                                       at::Tensor grad_out, at::Tensor idx, at::Tensor grad_points);

at::Tensor furthest_sampling_forward(
  const int m,
  const int seedIdx,
  at::Tensor input,
  at::Tensor temp,
  at::Tensor idx
)
{
  CHECK_INPUT(input);
  CHECK_INPUT(temp);
  return furthest_sampling_cuda_forward(m, seedIdx, input, temp, idx);
}

at::Tensor gather_points_forward(int b, int c, int n, int npoints,
                                 at::Tensor points_tensor,
                                 at::Tensor idx_tensor,
                                 at::Tensor out_tensor)
{
  CHECK_INPUT(points_tensor);
  CHECK_INPUT(idx_tensor);
  return gather_points_cuda_forward(b, c, n, npoints, points_tensor, idx_tensor, out_tensor);
}

at::Tensor gather_points_backward(int b, int c, int n, int npoints,
                                  at::Tensor grad_out_tensor,
                                  at::Tensor idx_tensor,
                                  at::Tensor grad_points_tensor)
{
  return gather_points_cuda_backward(b, c, n, npoints, grad_out_tensor, idx_tensor, grad_points_tensor);
}

at::Tensor ball_query_cuda_forward(float radius, int nsample, at::Tensor new_xyz,
                                   at::Tensor xyz, at::Tensor out_idx);

at::Tensor ball_query_forward(at::Tensor query, at::Tensor xyz, const float radius,
                              const int nsample)
{
  CHECK_INPUT(query);
  CHECK_INPUT(xyz);
  CHECK_CUDA(xyz);
  CHECK_CUDA(query);

  at::Tensor idx =
      torch::zeros({query.size(0), query.size(1), nsample},
                   at::device(query.device()).dtype(at::ScalarType::Int));

  if (query.type().is_cuda())
  {
    ball_query_cuda_forward(radius, nsample, query,
                            xyz, idx);
  }
  return idx;
}
void group_points_kernel_wrapper(int b, int c, int n, int npoints, int nsample,
                                 const float *points, const int *idx,
                                 float *out);

void group_points_grad_kernel_wrapper(int b, int c, int n, int npoints,
                                      int nsample, const float *grad_out,
                                      const int *idx, float *grad_points);

at::Tensor group_points(at::Tensor points, at::Tensor idx) {
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(idx);
  CHECK_IS_FLOAT(points);
  CHECK_IS_INT(idx);

  if (points.type().is_cuda()) {
    CHECK_CUDA(idx);
  }

  at::Tensor output =
      torch::zeros({points.size(0), points.size(1), idx.size(1), idx.size(2)},
                   at::device(points.device()).dtype(at::ScalarType::Float));

  if (points.type().is_cuda()) {
    group_points_kernel_wrapper(points.size(0), points.size(1), points.size(2),
                                idx.size(1), idx.size(2), points.data<float>(),
                                idx.data<int>(), output.data<float>());
  } else {
    AT_CHECK(false, "CPU not supported");
  }

  return output;
}

at::Tensor group_points_grad(at::Tensor grad_out, at::Tensor idx, const int n) {
  CHECK_CONTIGUOUS(grad_out);
  CHECK_CONTIGUOUS(idx);
  CHECK_IS_FLOAT(grad_out);
  CHECK_IS_INT(idx);

  if (grad_out.type().is_cuda()) {
    CHECK_CUDA(idx);
  }

  at::Tensor output =
      torch::zeros({grad_out.size(0), grad_out.size(1), n},
                   at::device(grad_out.device()).dtype(at::ScalarType::Float));

  if (grad_out.type().is_cuda()) {
    group_points_grad_kernel_wrapper(
        grad_out.size(0), grad_out.size(1), n, idx.size(1), idx.size(2),
        grad_out.data<float>(), idx.data<int>(), output.data<float>());
  } else {
    AT_CHECK(false, "CPU not supported");
  }

  return output;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("furthest_sampling", &furthest_sampling_forward, "furthest point sampling (no gradient)");
  m.def("gather_forward", &gather_points_forward, "gather npoints points along an axis");
  m.def("gather_backward", &gather_points_backward, "gather npoints points along an axis backward");
  m.def("ball_query", &ball_query_forward, "ball query");
  m.def("group_points", &group_points);
  m.def("group_points_grad", &group_points_grad);
}