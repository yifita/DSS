#include <ATen/ATen.h>
#include <ATen/TensorAccessor.h>
#include <ATen/TensorUtils.h>
#include <THC/THCAtomics.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <tuple>
#include "rasterization_utils.cuh"

#define GRID_3D_MIN_X 0
#define GRID_3D_MIN_Y 1
#define GRID_3D_MIN_Z 2
#define GRID_3D_DELTA 3
#define GRID_3D_RES_X 4
#define GRID_3D_RES_Y 5
#define GRID_3D_RES_Z 6
#define GRID_3D_TOTAL 7
#define GRID_3D_PARAMS_SIZE 8
#define GRID_3D_MAX_RES 128

#define GRID_2D_MIN_X 0
#define GRID_2D_MIN_Y 1
#define GRID_2D_DELTA 2
#define GRID_2D_RES_X 3
#define GRID_2D_RES_Y 4
#define GRID_2D_TOTAL 5
#define GRID_2D_PARAMS_SIZE 6
#define GRID_2D_MAX_RES 1024

__global__ void RasterizePointsBackwardCudaFastKernel(
    const float* __restrict__ points_sorted,        // (P,3)
    const float* __restrict__ radii_sorted,         // (P,2)
    const float* __restrict__ rs,                   // (N,)
    const long* __restrict__ num_points_per_cloud,  // (N,)
    const long* __restrict__ cloud_to_packed_first_idx,  // (N,)
    const int32_t* __restrict__ points_grid_off,        // (N,G) offset for the entire pack
    const float* __restrict__ grid_params,
    const float* __restrict__ grad_occ,             // (N,H,W)
    // const int32_t * __restrict__ point_idxs,        // (N,H,W,K)
    // const float* __restrict__ grad_zbuf,            // (N,H,W,K)
    const int N,
    const int H,
    const int W,
    // const int K,
    const int B,
    const int G,
    float* grad_points
) {
  // loop over all pixels. indexing s.t. neighboring threads get pixels inside the same grid
  const int BIN_SIZE_Y = (H + B/2) / B;
  const int BIN_SIZE_X = (W + B/2) / B;

  const float PIXEL_SIZE_X = 2.0f / W;
  const float PIXEL_SIZE_Y = 2.0f / H;
  const float PIXEL_SIZE = (PIXEL_SIZE_X+PIXEL_SIZE_Y) / 2.0;

  const int num_pixels = N * BIN_SIZE_X * BIN_SIZE_Y * B * B;
  const int num_threads = gridDim.x * blockDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (int pid = tid; pid < num_pixels; pid += num_threads) {
    int i = pid;

    // Convert linear index into bin and pixel indices.
    const int n = i / (BIN_SIZE_X * BIN_SIZE_Y * B * B);  // batch
    i %= BIN_SIZE_X * BIN_SIZE_Y * B * B;
    const int by = i / (B * BIN_SIZE_X * BIN_SIZE_Y);  // bin_y
    i %= B * BIN_SIZE_X * BIN_SIZE_Y;
    const int bx = i / (BIN_SIZE_X * BIN_SIZE_Y);  // bin_x
    assert(n < N && n >= 0);
    assert(bx < B && bx >= 0);
    assert(by < B && by >= 0);
    // lixin
    // i %= B * BIN_SIZE_X;
    // const int bx = i / BIN_SIZE_X;
    i %= BIN_SIZE_X * BIN_SIZE_Y;  // index inside the bin

    // Pixel indices
    const int yidx = i / BIN_SIZE_X + by * BIN_SIZE_Y;
    const int xidx = i % BIN_SIZE_X + bx * BIN_SIZE_X;

    if (yidx >= H || xidx >= W) {
      continue;
    }
    const float grad_occ_pix = grad_occ[n*H*W + yidx*W + xidx];
    if (grad_occ_pix != 0.0f) {
      // reverse because NDC assuming +y is up and +x is left
      const int yi = H - 1 - yidx;
      const int xi = W - 1 - xidx;
      assert(xi >= 0 && xi < W);
      assert(yi >= 0 && yi < H);

      // Pixel in NDC coordinates
      const float xf = PixToNdc(xi, W);
      const float yf = PixToNdc(yi, H);
      assert(abs(xf) <= 1.0 && abs(yf) <= 1.0);

      const long cur_first_idx = cloud_to_packed_first_idx[n];
      const float cur_r = rs[n];  // search radius
      const float cur_r2 = cur_r * cur_r;

      const float grid_min_x = grid_params[n*GRID_2D_PARAMS_SIZE+GRID_2D_MIN_X];
      const float grid_min_y = grid_params[n*GRID_2D_PARAMS_SIZE+GRID_2D_MIN_Y];
      const float grid_delta = grid_params[n*GRID_2D_PARAMS_SIZE+GRID_2D_DELTA];  // 1/cell_size
      const int grid_res_x = grid_params[n*GRID_2D_PARAMS_SIZE+GRID_2D_RES_X];
      const int grid_res_y = grid_params[n*GRID_2D_PARAMS_SIZE+GRID_2D_RES_Y];
      const int grid_total = grid_params[n*GRID_2D_PARAMS_SIZE+GRID_2D_TOTAL];
      // const float grad_occ_pix = grad_occ[i];
      assert(n*H*W + yi*W + xi < N*H*W);

      int min_gc_x = (int) floor((xf-grid_min_x-cur_r) * grid_delta);
      int min_gc_y = (int) floor((yf-grid_min_y-cur_r) * grid_delta);
      int max_gc_x = (int) floor((xf-grid_min_x+cur_r) * grid_delta);
      int max_gc_y = (int) floor((yf-grid_min_y+cur_r) * grid_delta);

      // Search the relevant grid
      for (int x=max(min_gc_x, 0); x<=min(max_gc_x, grid_res_x-1); ++x) {
        for (int y=max(min_gc_y, 0); y<=min(max_gc_y, grid_res_y-1); ++y) {
          int cell_idx = x*grid_res_y + y;
          assert(cell_idx < grid_total);
          // Get the relevant index range of points
          const int64_t p2_start = points_grid_off[n*G + cell_idx];
          int p2_end;
          if (cell_idx+1 == grid_total) {
            p2_end = num_points_per_cloud[n];
          }
          else {
            p2_end = points_grid_off[n*G+cell_idx+1];
          }
          if (p2_end > cur_first_idx+num_points_per_cloud[n]){
              printf("points_grid_off[%d, %d] = %d, grid_total = %d, p2_end = %d, num_points_per_cloud[%d] = %d, cur_first_idx = %d, pid = %d\n", n, cell_idx, p2_start, grid_total, n, num_points_per_cloud[n], p2_end, cur_first_idx, pid);
            }
          assert(p2_end <= cur_first_idx+num_points_per_cloud[n]);
          // Loop over the relevant points, aggregate gradients
          for (int p_idx=p2_start; p_idx<p2_end; ++p_idx) {
            // check radii
            if (p_idx < cur_first_idx){
              printf("points_grid_off[%d, %d] = %d, p_idx = %d, cur_first_idx = %d, pid = %d\n", n, cell_idx, p2_start, p_idx, cur_first_idx, pid);
            }
            assert(p_idx >= cur_first_idx);
            const float px = points_sorted[p_idx * 3 + 0];
            const float py = points_sorted[p_idx * 3 + 1];
            const float pz = points_sorted[p_idx * 3 + 2];
            // outside renderable area
            if (pz < 0 || abs(py) > 1.0 || abs(px) > 1.0)
              continue;
            const float dx = xf - px;
            const float dy = yf - py;

            const float radiix = radii_sorted[p_idx * 2 + 0];
            const float radiiy = radii_sorted[p_idx * 2 + 1];

            const float dist2 = dx * dx + dy * dy;

            // inside backpropagation radius?
            if (dist2 > cur_r2)
              continue; // Skip if pixel out of precomputed radii range

            // inside splat? NOTE: this is not as accurate as check qvalue < cutoffthreshold
            // but it's a close approximation
            const bool pix_outside_splat = (abs(dx) > radiix) || (abs(dy) > radiiy);

            // if grad_occ_pix > 0, it means that this pixel shouldn't be occluded
            // but if it's outside the splat, it doesn't generate meaninigful information
            // for in which direction the point should move.
            if (grad_occ_pix > 0.0f && pix_outside_splat)
                // if (grad_occ_pix > 0.0f)
                continue;

            const float denom = eps_denom(dist2, 1e-10f);
            const float grad_px = dx / denom * grad_occ_pix;
            const float grad_py = dy / denom * grad_occ_pix;
            // const float grad_px = clamp(dx / denom, -10/PIXEL_SIZE, 10/PIXEL_SIZE) * grad_occ_pix;
            // const float grad_py = clamp(dy / denom, -10/PIXEL_SIZE, 10/PIXEL_SIZE) * grad_occ_pix;

            // printf("grad_pts[%d] = [%g, %g]\n", p_idx, grad_px, grad_py);
            gpuAtomicAdd(grad_points + p_idx * 2 + 0, grad_px);
            gpuAtomicAdd(grad_points + p_idx * 2 + 1, grad_py);

    }

          // // If inside splat, copy the grad_pz
          // if (!pix_outside_splat) {
          //   const int ik = (n*H*W+yi*W+xi) * K;   // pid is for (B*BIN_SIZE_Y)x(B*BIN_SIZE_X)
          //   assert(n < N);
          //   assert(yi < H);
          //   assert(xi < W);
          //   assert(n*H*W+yi*W+xi < N*H*W);
          //   assert(ik+K-1 < N*H*W*K);
          //   // if (ik >= N*H*W*K) {
          //   //     printf("N: %d, n: %d\n", N, n);
          //   //     // printf("H: %d, h: %d\n", H, yi);
          //   //     // printf("W: %d, w: %d\n", W, xi);
          //   //     // printf("N*H*W: %d, n*H*W+yi*W+xi: %d\n", N*H*W, n*H*W+yi*W+xi);
          //   //     printf("N*H*W*K: %d, (n*H*W+yi*W+xi)*K: %d\n", N*H*W*K, (n*H*W+yi*W+xi)*K);
          //   //     assert(ik < N*H*W*K);
          //   // }
          //   for (int k = 0; k < K; k++)
          //   {
          //       const int z_idx = point_idxs[ik + k];
          //       if (z_idx < 0)
          //           break;
          //       const float grad_pz = grad_zbuf[ik + k];
          //       gpuAtomicAdd(grad_points + z_idx * 3 + 2, grad_pz);
          //   }
          // }
        }

      }
    }
  }
}
/*
Args:
  points,    // (P, 3)
  radii,     // (P, 2)
  idx,       // (N, H, W, K)
  rs,        // (N, )
  grad_occ,  // (N, H, W)
  grad_zbuf, // (N, H, W, K)
  num_points_per_cloud, // (N,)
  cloud_to_packed_first_idx,  // (N,)
  points_grid_off (N, G) The packed index of the first point stored in a grid
Returns:
  grad_points: (P, 3)
 */
at::Tensor RasterizePointsBackwardCudaFast(
  const at::Tensor &points_sorted,    // (P, 3)
  const at::Tensor &radii_sorted,     // (P, 2)
  // const at::Tensor &idxs,       // (N, H, W, K)
  const at::Tensor &rs,        // (N, )
  const at::Tensor &grad_occ,  // (N, H, W)
  // const at::Tensor &grad_zbuf, // (N, H, W, K)
  const at::Tensor &num_points_per_cloud, // (N,)
  const at::Tensor &cloud_to_packed_first_idx,  // (N,)
  const at::Tensor &points_grid_off, // (N, G)
  const at::Tensor &grid_params // (N, GRID_2D_PARAMS_SIZE)
) {
  // Check inputs are on the same device
  at::TensorArg points_t{points_sorted, "points", 1},
      radii_t{radii_sorted, "radii", 2},
      // idxs_t{idxs, "idxs", 3},
      rs_t{rs, "rs", 3},
      grad_occ_t{grad_occ, "grad_occ", 4},
      // grad_zbuf_t{grad_zbuf, "grad_zbuf", 6},
      num_points_per_cloud_t{num_points_per_cloud, "num_points_per_cloud", 5},
      cloud_to_packed_first_idx_t{
          cloud_to_packed_first_idx, "cloud_to_packed_first_idx", 6},
      points_grid_off_t{points_grid_off, "points_grid_off", 7},
      grid_params_t{grid_params, "grid_params", 8};
  at::CheckedFrom c = "RasterizePointsBackwardCudaFast";
  at::checkDim(c, points_t, 2);
  at::checkDim(c, radii_t, 2);
  // at::checkDim(c, idxs_t, 4);
  at::checkDim(c, rs_t, 1);
  at::checkDim(c, grad_occ_t, 3);
  // at::checkDim(c, grad_zbuf_t, 4);
  at::checkDim(c, num_points_per_cloud_t, 1);
  at::checkDim(c, cloud_to_packed_first_idx_t, 1);
  at::checkDim(c, points_grid_off_t, 2);
  at::checkDim(c, grid_params_t, 2);
  at::checkSize(c, grid_params_t, 1, GRID_2D_PARAMS_SIZE);
  at::checkSize(c, points_t, 1, 3);
  at::checkSize(c, radii_t, {points_t->size(0), 2});
  // at::checkAllSameSize(c, {rs_t, num_points_per_cloud_t, cloud_to_packed_first_idx_t});
  at::checkSameSize(c, rs_t, cloud_to_packed_first_idx_t);
  at::checkSameSize(c, rs_t, num_points_per_cloud_t);
  // at::checkSameSize(c, grad_zbuf_t, idxs_t);

  at::checkAllSameGPU(
      c, {points_t, radii_t, rs_t, grad_occ_t,
      num_points_per_cloud_t, cloud_to_packed_first_idx_t, points_grid_off_t});

  // Set the device for the kernel launch based on the device of the input
  at::cuda::CUDAGuard device_guard(points_sorted.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int P = points_sorted.size(0);
  const int G = points_grid_off.size(1);
  const int N = grad_occ.size(0);
  const int H = grad_occ.size(1);
  const int W = grad_occ.size(2);
  int B = 1;
  const int S = min(H, W);

  if (S >= 64)
    B = 8;
  if (S >= 128)
    B = 16;
  if (S >= 256)
    B = 32;
  if (S >= 512)
    B = 64;

  // call backward fast kernel on sorted points_sorted and sorted radii_sorted, this will return a gradient [P, 3] of the *sorted* points_sorted
  const size_t blocks = 1024;
  const size_t threads = 64;
  at::Tensor grad_points_sorted = at::zeros({P, 2}, points_sorted.options());
  RasterizePointsBackwardCudaFastKernel<<<blocks, threads, 0, stream>>>(
      points_sorted.contiguous().data_ptr<float>(),       // (P,3)
      radii_sorted.contiguous().data_ptr<float>(),        // (P,2)
      rs.contiguous().data_ptr<float>(),                  // (N,)
      num_points_per_cloud.contiguous().data_ptr<int64_t>(),  // (N,)
      cloud_to_packed_first_idx.contiguous().data_ptr<int64_t>(),  // (N,)
      points_grid_off.contiguous().data_ptr<int32_t>(),    // (N,G)
      grid_params.contiguous().data_ptr<float>(),          // (N,8)
      grad_occ.contiguous().data_ptr<float>(),             // (N,H,W)
      // idxs.contiguous().data_ptr<int32_t>(),               // (N,H,W)
      // grad_zbuf.contiguous().data_ptr<float>(),            // (N,H,W)
      N,
      H,
      W,
      // K,
      B,
      G,   // grid_res_x * grid_res_y
      grad_points_sorted.contiguous().data_ptr<float>()
  );

  AT_CUDA_CHECK(cudaGetLastError());

  return grad_points_sorted;
}