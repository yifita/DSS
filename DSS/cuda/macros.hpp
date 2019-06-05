#pragma once
#include <torch/extension.h>

#define CHECK_INPUT(x)                                                         \
  AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor");                   \
  AT_CHECK(x.is_contiguous(), #x " must be contiguous")

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor");