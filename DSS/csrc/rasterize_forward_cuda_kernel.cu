#include "cuda_utils.h"
#include "macros.hpp"
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <iostream>
#include <stdio.h>
#include <torch/extension.h>

/*
        Given point cloud's screen position, local depth values and local
        filter values rho,
        output a PxHxW point index map indicating which point contributes
        to which pixel by how much.
        (pytorch tensors are row-major)
 */
template <typename scalar_t, typename indice_t>
__global__ void
gather_maps_kernel(int batchSize, int numPoint, int imgWidth, int imgHeight,
                   int topK, int C,
                   const indice_t *__restrict__ indices, // BxHxWxtopK
                   const scalar_t *__restrict__ data,    // BxNxC
                   const scalar_t defaultValue,
                   scalar_t *output) // BxHxWxtopKxC
{
  const int numPixels = imgWidth * imgHeight;
  // loop all pixels
  for (int b = blockIdx.x; b < batchSize; b += gridDim.x) {
    for (int p = threadIdx.x + blockIdx.y * blockDim.x; p < numPixels;
         p += blockDim.x * gridDim.y) {
      const int pixID = b * numPixels + p;
      // loop over topK dimension
      for (int i = 0; i < topK; i++) {
        const indice_t pid = indices[pixID * topK + i];
        for (int c = 0; c < C; c++) {
          // dereference point from the N dimension of data
          if (pid < 0 || pid > numPoint)
            output[pixID * topK * C + i * C + c] = defaultValue;
          else
            output[pixID * topK * C + i * C + c] =
                data[(b * numPoint + pid) * C + c];
        }
      }
    }
  }
}
/* put gradient BxHxWxKxC to the correct position in BxNxC */
template <typename scalar_t, typename indice_t>
__global__ void
scatter_maps_kernel(int batchSize, int numPoint, int imgWidth, int imgHeight,
                    int topK, int C,
                    const scalar_t *__restrict__ outGrad, // BxHxWxtopKxC
                    const indice_t *__restrict__ indices, // BxHxWxtopK
                    scalar_t *__restrict__ dataGrad)      // BxNxC
{
  // const int numPixels = imgWidth * imgHeight;
  // loop all points
  // TODO instead of looping over points then pixels, loop over pixels only and write
  // in points with cudasync?
  for (int b = blockIdx.x; b < batchSize; b += gridDim.x) {
    for (indice_t p = threadIdx.x + blockIdx.y * blockDim.x; p < numPoint;
         p += blockDim.x * gridDim.y) {
      // loop over all pixels
      for (int i = 0; i < imgHeight; i++) {
        for (int j = 0; j < imgWidth; j++) {
          int pixID = b * imgWidth * imgHeight + i * imgWidth + j;
          const indice_t *iOffset = indices + pixID * topK;
          for (int k = 0; k < topK; k++) {
            indice_t pid = iOffset[k];
            if (pid == p) {
              for (int c = 0; c < C; c++) {
                dataGrad[(b * numPoint + pid) * C + c] +=
                    outGrad[(pixID * topK + k) * C + c];
              }
            }
          }
        }
      }
    }
  }
}
/* put gradient BxHxWxKxC to the correct position in BxNxC */
template <typename scalar_t, typename indice_t>
__global__ void
guided_scatter_maps_kernel(int batchSize, int numPoint, int imgWidth,
                           int imgHeight, int topK, int C,
                           const scalar_t *__restrict__ outGrad, // BxHxWxtopKxC
                           const indice_t *__restrict__ indices, // BxHxWxtopK
                           const indice_t *__restrict__ boundingBoxes, // BxNx4
                           scalar_t *__restrict__ dataGrad)            // BxNxC
{
  // const int numPixels = imgWidth * imgHeight;
  // loop all points
  for (indice_t b = blockIdx.x; b < batchSize; b += gridDim.x) {
    for (indice_t p = threadIdx.x + blockIdx.y * blockDim.x; p < numPoint;
         p += blockDim.x * gridDim.y) {
      const indice_t curPointIdx = b * numPoint + p;
      scalar_t xmin = max(boundingBoxes[curPointIdx * 4], indice_t(0));
      indice_t ymin = max(boundingBoxes[curPointIdx * 4 + 1], indice_t(0));
      indice_t xmax =
          min(indice_t(boundingBoxes[curPointIdx * 4 + 2]), indice_t(imgWidth));
      indice_t ymax = min(indice_t(boundingBoxes[curPointIdx * 4 + 3]),
                          indice_t(imgHeight));
      // loop over all pixels
      for (indice_t i = ymin; i < ymax; i++) {
        for (indice_t j = xmin; j < xmax; j++) {
          indice_t pixID = b * imgWidth * imgHeight + i * imgWidth + j;
          const indice_t *iOffset = indices + pixID * topK;
          for (indice_t k = 0; k < topK; k++) {
            indice_t pid = iOffset[k];
            if (pid == p) {
              for (indice_t c = 0; c < C; c++) {
                dataGrad[(b * numPoint + pid) * C + c] +=
                    outGrad[(pixID * topK + k) * C + c];
              }
            }
          }
        }
      }
    }
  }
}
template <typename scalar_t, typename indice_t>
__device__ void
update_IndexMap(const scalar_t depth, const int pointId, const int yInBB,
                const int xInBB, const int topK, indice_t *pointIdxList,
                indice_t *bbPositionList, scalar_t *pointDepthList) {
  // compare depth with topK depth list of the current pixel
  for (int i = 0; i < topK; i++) {
    if (depth < pointDepthList[i]) {
      // insert current pointID, yInBB and xInBB and depth to the list
      // by shifting [i, topK-1] part of the topK list
      for (int j = topK - 1; j > i; j--) {
        pointDepthList[j] = pointDepthList[j - 1];
        pointIdxList[j] = pointIdxList[j - 1];
        bbPositionList[j * 2] = bbPositionList[(j - 1) * 2];
        bbPositionList[j * 2 + 1] = bbPositionList[(j - 1) * 2 + 1];
      }
      pointIdxList[i] = indice_t(pointId);
      bbPositionList[i * 2] = indice_t(yInBB);
      bbPositionList[i * 2 + 1] = indice_t(xInBB);
      pointDepthList[i] = depth;
      break;
    }
  }
}
// visibility kernel, outputs pointIdxMap and depthMap that saves 5 points per
// pixel which are order by the increasing z-value. bbPositionMap is the
// relative position of the current pixel within the point's bounding box
template <typename scalar_t, typename indice_t>
__global__ void compute_visiblity_maps_kernel(
    int batchSize, int numPoint, int imgWidth, int imgHeight, int bbWidth,
    int bbHeight, int topK,
    const indice_t *__restrict__ boundingBoxes, // BxPx2
    const scalar_t *__restrict__ inPlane,       // BxPxhxwx3
    indice_t *pointIdxMap,                      // BxHxWxK
    indice_t *bbPositionMap,                    // BxHxWxKx2
    scalar_t *depthMap                          // BxHxWxK
) {
  if (numPoint <= 0)
    return;
  int numPixels = imgWidth * imgHeight;
  int bbSize = bbWidth * bbHeight;
  // loop in the batch
  for (int b = blockIdx.x; b < batchSize; b += gridDim.x) {
    // loop all pixels
    for (int p = threadIdx.x + blockIdx.y * blockDim.x; p < numPixels;
         p += blockDim.x * gridDim.y) {
      // current pixel position (h,w)
      const int h = p / imgWidth;
      const int w = p % imgWidth;
      const int pixID = b * numPixels + p;
      assert(pixID < batchSize * numPixels && pixID >= 0);
      // loop all points
      for (int k = 0; k < numPoint; k++) {
        const int pointId = b * numPoint + k;
        const indice_t xmin = boundingBoxes[pointId * 2];
        const indice_t ymin = boundingBoxes[pointId * 2 + 1];
        // if current pixel is inside the point's boundingbox,
        // compare with depth. Update pointIdxMap and depthMap
        if (xmin <= w && ymin <= h && (xmin + bbWidth > w) &&
            (ymin + bbHeight > h)) {
          // relative position inside the bounding box
          const int yInBB = h - ymin;
          const int xInBB = w - xmin;
          assert(yInBB >= 0 && yInBB < bbHeight);
          assert(xInBB >= 0 && xInBB < bbWidth);
          const scalar_t depth = inPlane[pointId * bbSize * 3 +
                                         yInBB * bbWidth * 3 + xInBB * 3 + 2];
          update_IndexMap(
              depth, k, yInBB, xInBB, topK, pointIdxMap + pixID * topK,
              bbPositionMap + pixID * topK * 2, depthMap + pixID * topK);
        }
      }
    }
  }
}

void compute_visiblity_maps_cuda(const at::Tensor &boundingBoxes,
                                 const at::Tensor &inPlane,
                                 at::Tensor &pointIdxMap,
                                 at::Tensor &bbPositionMap,
                                 at::Tensor &depthMap) {
  TORCH_CHECK(inPlane.dim() == 5);
  const int batchSize = inPlane.size(0);
  const int numPoint = inPlane.size(1);
  const int bbHeight = inPlane.size(2);
  const int bbWidth = inPlane.size(3);
  const int imgHeight = pointIdxMap.size(1);
  const int imgWidth = pointIdxMap.size(2);
  const int topK = pointIdxMap.size(-1);

  int device;
  cudaGetDevice(&device);
  // printf("compute_visiblity_maps_cuda using device %d\n", device);
  unsigned int n_threads, n_blocks;
  int numPixels = imgWidth * imgHeight;
  n_threads = opt_n_threads(numPixels);
  n_blocks = min(32, (numPixels * batchSize + n_threads - 1) / n_threads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      inPlane.scalar_type(), "compute_visiblity_maps_kernel", ([&] {
        compute_visiblity_maps_kernel<scalar_t, int64_t>
            <<<dim3(batchSize, n_blocks, 1), n_threads, 0, stream>>>(
                batchSize, numPoint, imgWidth, imgHeight, bbWidth, bbHeight,
                topK,
                boundingBoxes.data_ptr<int64_t>(),
                inPlane.data_ptr<scalar_t>(), pointIdxMap.data_ptr<int64_t>(),
                bbPositionMap.data_ptr<int64_t>(), depthMap.data_ptr<scalar_t>());
      }));
  cudaError_t err = cudaGetLastError();
  // cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("compute_visiblity_maps_cuda kernel failed: %s\n",
           cudaGetErrorString(err));
    exit(-1);
  }
  return;
}

// data BxNxC, indices BxHxWxK value (0~N-1), output BxHxKxC
at::Tensor gather_maps_cuda(const at::Tensor &data, const at::Tensor &indices,
                            const double defaultValue) {
  const int batchSize = data.size(0);
  const int numPoint = data.size(1);
  const int imgHeight = indices.size(1);
  const int imgWidth = indices.size(2);
  const int topK = indices.size(3);
  const int channels = data.size(2);
  at::Scalar dv = at::Scalar(defaultValue);
  auto output = at::full({batchSize, imgHeight, imgWidth, topK, channels}, dv,
                         data.options());
  unsigned int n_threads, n_blocks;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int pixelNumber = imgWidth * imgHeight;
  n_threads = opt_n_threads(pixelNumber);
  n_blocks = min(32, (pixelNumber * batchSize + n_threads - 1) / n_threads);
  // printf("gather_maps_cuda: kernel config (%d, %d, %d)\n", batchSize,
  // n_blocks,
  //        n_threads);
  AT_DISPATCH_ALL_TYPES(
      data.scalar_type(), "gather_maps_kernel", ([&] {
        gather_maps_kernel<scalar_t, int64_t>
            <<<dim3(batchSize, n_blocks, 1), n_threads, 0, stream>>>(
                batchSize, numPoint, imgWidth, imgHeight, topK, channels,
                indices.data_ptr<int64_t>(), data.data_ptr<scalar_t>(),
                dv.to<scalar_t>(), output.data_ptr<scalar_t>());
      }));
  // cudaError_t err = cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("gather_maps_cuda kernel failed: %s\n", cudaGetErrorString(err));
    exit(-1);
  }
  return output;
}

// the inverse of gather_maps
// src BxHxWxKxC, indices BxHxWxK value (0~N-1), output BxHxC
at::Tensor scatter_maps_cuda(const int64_t numPoint, const at::Tensor &src,
                             const at::Tensor &indices) {
  const int batchSize = indices.size(0);
  const int imgHeight = indices.size(1);
  const int imgWidth = indices.size(2);
  const int topK = indices.size(3);
  const int channels = src.size(-1);
  unsigned int n_threads, n_blocks;
  const int nP = int(numPoint);
  n_threads = opt_n_threads(nP);
  n_blocks = min(32, (nP * batchSize + n_threads - 1) / n_threads);
  // initialize with zeros
  auto dataGrad = at::zeros({batchSize, nP, channels}, src.options());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Bool,
      src.scalar_type(), "scatter_maps_kernel", ([&] {
        scatter_maps_kernel<scalar_t, int64_t>
            <<<dim3(batchSize, n_blocks, 1), n_threads, 0, stream>>>(
                batchSize, nP, imgWidth, imgHeight, topK, channels,
                src.data_ptr<scalar_t>(), indices.data_ptr<int64_t>(),
                dataGrad.data_ptr<scalar_t>());
      }));
  cudaError_t err = cudaDeviceSynchronize();
  // cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("scatter_maps_cuda kernel failed: %s\n", cudaGetErrorString(err));
    exit(-1);
  }
  return dataGrad;
}

// the inverse of gather_maps, use boundingboxes to restrict search areas
// src BxHxWxKxC, indices BxHxWxK value (0~N-1), output BxHxC
at::Tensor guided_scatter_maps_cuda(const int64_t numPoint,
                                    const at::Tensor &src,
                                    const at::Tensor &indices,
                                    const at::Tensor &boundingBoxes) {
  const int batchSize = indices.size(0);
  const int imgHeight = indices.size(1);
  const int imgWidth = indices.size(2);
  const int topK = indices.size(3);
  CHECK_EQ(src.dim(), 5);
  const int channels = src.size(-1);
  unsigned int n_threads, n_blocks;
  const int nP = int(numPoint);
  n_threads = opt_n_threads(nP);
  // 2D grid (batchSize, n_blocks)
  n_blocks = min(32, (nP * batchSize + n_threads - 1) / n_threads);
  // initialize with zeros
  auto dataGrad = at::zeros({batchSize, nP, channels}, src.options());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Bool,
      src.scalar_type(), "guided_scatter_maps_kernel", ([&] {
        guided_scatter_maps_kernel<scalar_t, int64_t>
            <<<dim3(batchSize, n_blocks, 1), n_threads, 0, stream>>>(
                batchSize, nP, imgWidth, imgHeight, topK, channels,
                src.data_ptr<scalar_t>(),
                indices.data_ptr<int64_t>(),
                boundingBoxes.data_ptr<int64_t>(),
                dataGrad.data_ptr<scalar_t>());
      }));
  cudaError_t err = cudaDeviceSynchronize();
  // cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("guided_scatter_maps_cuda kernel failed: %s\n",
           cudaGetErrorString(err));
    exit(-1);
  }
  return dataGrad;
}