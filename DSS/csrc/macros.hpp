#pragma once
#include <torch/extension.h>

#define CHECK_INPUT(x)                                                         \
  TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor");                   \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor");