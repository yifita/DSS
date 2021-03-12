// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x "must be a CUDA tensor.")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x "must be contiguous.")
#define CHECK_CONTIGUOUS_CUDA(x) \
  CHECK_CUDA(x);                 \
  CHECK_CONTIGUOUS(x)

// ****************************************************************************
// *                          NAIVE RASTERIZATION                             *
// ****************************************************************************

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> RasterizePointsNaiveCpu(
    const torch::Tensor &points,
    const torch::Tensor &ellipse_params,
    const torch::Tensor &cutoff_thres,
    const torch::Tensor &radii,
    const torch::Tensor &cloud_to_packed_first_idx,
    const torch::Tensor &num_points_per_cloud,
    const float depth_merging_thres,
    const int image_size,
    const int points_per_pixel);

#ifdef WITH_CUDA
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
RasterizePointsNaiveCuda(
    const torch::Tensor &points,
    const torch::Tensor &ellipse_params,
    const torch::Tensor &cutoff_thres,
    const torch::Tensor &radii,
    const torch::Tensor &cloud_to_packed_first_idx,
    const torch::Tensor &num_points_per_cloud,
    const float depth_merging_thres,
    const int image_size,
    const int points_per_pixel);
#endif
// Naive (forward) pointcloud rasterization: For each pixel, for each point,
// check whether that point hits the pixel.
//
// Args:
//  points: Tensor of shape (P, 3) giving (packed) positions for
//          points in all N pointclouds in the batch where P is the total
//          number of points in the batch across all pointclouds. These points
//          are expected to be in NDC coordinates in the range [-1, 1].
//  ellipse_params: Tensor of shape (P, 3) giving the quadratic coefficients for each splatted point
//                  i.e. the (a,b,c) in ax^2 + bxy + cy^2
// !TODO(lixin): maybe we can reduce this to a single float as the threshold can be encoded in ellipse_params
//  cutoff_thres: Tensor of shape (P,) giving the cut off threshold for each ellipse splat
//  radii: FloatTensor of (N,2) splat radius in x and y direction (in NDC)
//  cloud_to_packed_first_idx: LongTensor of shape (N) giving the index in
//                          points_packed of the first point in each pointcloud
//                          in the batch where N is the batch size.
//  num_points_per_cloud: LongTensor of shape (N) giving the number of points
//                        for each pointcloud in the batch.
//  depth_merging_thres: the threshold for merging multiple splats if the difference of depth is below this threshold
//  image_size: (S) Size of the image to return (in pixels)
//  points_per_pixel: (K) The number closest of points to return for each pixel
//
// Returns:
//  A 4 element tuple of:
//  idxs: int32 Tensor of shape (N, S, S, K) giving the indices of the
//        closest K points along the z-axis for each pixel, padded with -1 for
//        pixels hit by fewer than K points. The indices refer to points in
//        points packed i.e a tensor of shape (P, 3) representing the flattened
//        points for all pointclouds in the batch.
//  zbuf: float32 Tensor of shape (N, S, S, K) giving the depth of each
//        closest point for each pixel.
//  qvalue: float32 Tensor of shape (N, S, S, K) giving the exponent to
//          compute the composition weight exp(-Qvalue)
//  occupancy: float32  (N, S, S) giving the occupancy 0/1 at the pixel
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> RasterizePointsNaive(
    const torch::Tensor &points,
    const torch::Tensor &ellipse_params,
    const torch::Tensor &cutoff_thres,
    const torch::Tensor &radii,
    const torch::Tensor &cloud_to_packed_first_idx,
    const torch::Tensor &num_points_per_cloud,
    const float depth_merging_thres,
    const int image_size,
    const int points_per_pixel)
{
  if (points.is_cuda())
  {
#ifdef WITH_CUDA
    CHECK_CUDA(points);
    CHECK_CUDA(radii);
    CHECK_CUDA(ellipse_params);
    CHECK_CUDA(cutoff_thres);
    CHECK_CUDA(cloud_to_packed_first_idx);
    CHECK_CUDA(num_points_per_cloud);
    return RasterizePointsNaiveCuda(
        points,
        ellipse_params,
        cutoff_thres,
        radii,
        cloud_to_packed_first_idx,
        num_points_per_cloud,
        depth_merging_thres,
        image_size,
        points_per_pixel);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  else
  {
    return RasterizePointsNaiveCpu(
        points,
        ellipse_params,
        cutoff_thres,
        radii,
        cloud_to_packed_first_idx,
        num_points_per_cloud,
        depth_merging_thres,
        image_size,
        points_per_pixel);
  }
}

// ****************************************************************************
// *                          COARSE RASTERIZATION                            *
// ****************************************************************************
// TODO(lixin)
torch::Tensor RasterizePointsCoarseCpu(
    const torch::Tensor &points,                    // (P, 3)
    const torch::Tensor &radii,                     // (P, 2)
    const torch::Tensor &cloud_to_packed_first_idx, // (N)
    const torch::Tensor &num_points_per_cloud,      // (N)
    const int image_size,
    const int bin_size,
    const int max_points_per_bin);
torch::Tensor RasterizePointsCoarseCuda(
    const torch::Tensor &points,
    const torch::Tensor &radii,
    const torch::Tensor &cloud_to_packed_first_idx,
    const torch::Tensor &num_points_per_cloud,
    const int image_size,
    const int bin_size,
    const int max_points_per_bin);
// Args:
//  points: Tensor of shape (P, 3) giving (packed) positions for
//          points in all N pointclouds in the batch where P is the total
//          number of points in the batch across all pointclouds. These points
//          are expected to be in NDC coordinates in the range [-1, 1].
//  radii: Tensor of shape (N, 2) the axis-aligned radii for each splat
//  cloud_to_packed_first_idx: LongTensor of shape (N) giving the index in
//                          points_packed of the first point in each pointcloud
//                          in the batch where N is the batch size.
//  num_points_per_cloud: LongTensor of shape (N) giving the number of points
//                        for each pointcloud in the batch.
//  image_size: Size of the image to generate (in pixels)
//  bin_size: Size of each bin within the image (in pixels)
//  max_points_per_bin: Max number of points in a single bin (M)
//
// Returns:
//  TODO(lixin): remove this points_per_bin as this is useless / not returned for now
//  points_per_bin: IntTensor of shape (N, num_bins, num_bins) giving the number
//                  of points that fall in each bin
//  bin_points: IntTensor of shape (N, num_bins, num_bins, M) giving the indices
//              of points that fall into each bin.
torch::Tensor RasterizePointsCoarse(
    const torch::Tensor &points,
    const torch::Tensor &radii,
    const torch::Tensor &cloud_to_packed_first_idx,
    const torch::Tensor &num_points_per_cloud,
    const int image_size,
    const int bin_size,
    const int max_points_per_bin)
{
  if (points.is_cuda())
  {
    CHECK_CUDA(points);
    CHECK_CUDA(radii);
    CHECK_CUDA(cloud_to_packed_first_idx);
    CHECK_CUDA(num_points_per_cloud);
    return RasterizePointsCoarseCuda(
        points,
        radii,
        cloud_to_packed_first_idx,
        num_points_per_cloud,
        image_size,
        bin_size,
        max_points_per_bin);
  }
  else
  {
    // AT_ERROR("RasterizeCoarse CPU version unimplemented");
    return RasterizePointsCoarseCpu(
        points,
        radii,
        cloud_to_packed_first_idx,
        num_points_per_cloud,
        image_size,
        bin_size,
        max_points_per_bin);
  }
}

// ****************************************************************************
// *                            FINE RASTERIZATION                            *
// ****************************************************************************

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> RasterizePointsFineCpu(
    const torch::Tensor &points,            // (P, 3)
    const torch::Tensor &ellipse_params,    // (P, 3)
    const torch::Tensor &cutoff_thres,      // (P,)
    const torch::Tensor &radii,             // (P,2)
    const torch::Tensor &bin_points,        // (N, B, B, M)
    const float depth_merging_thres,
    const int image_size,
    const int bin_size,
    const int points_per_pixel);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> RasterizePointsFineCuda(
    const torch::Tensor &points,
    const torch::Tensor &ellipse_params,
    const torch::Tensor &cutoff_thres,
    const torch::Tensor &radii,
    const torch::Tensor &bin_points,
    const float depth_merging_thres,
    const int image_size,
    const int bin_size,
    const int points_per_pixel);
// Args:
//  points: Tensor of shape (P, 3) giving (packed) positions for
//          points in all N pointclouds in the batch where P is the total
//          number of points in the batch across all pointclouds. These points
//          are expected to be in NDC coordinates in the range [-1, 1].
//  ellipse_params: (N, 3) the parabolic parameters.
//  cutoff_thres: define the region of each ellipse by cutting it off
//                once its qvalue is larger than this threshold
//  radii: (N, 2) the axis-aligned radii for each splat
//  bin_points: int32 Tensor of shape (N, B, B, M) giving the indices of points
//              that fall into each bin (output from coarse rasterization)
//  depth_merging_thres: only points that are near enough will be blended
//  image_size: Size of image to generate (in pixels)
//  bin_size: Size of each bin (in pixels)
//  points_per_pixel: How many points to rasterize for each pixel
//
// Returns (same as rasterize_points):
//  idxs: int32 Tensor of shape (N, S, S, K) giving the indices of the
//        closest K points along the z-axis for each pixel, padded with -1 for
//        pixels hit by fewer than K points. The indices refer to points in
//        points packed i.e a tensor of shape (P, 3) representing the flattened
//        points for all pointclouds in the batch.
//  zbuf: float32 Tensor of shape (N, S, S, K) giving the depth of each of each
//        closest point for each pixel
//  qvalue_map: float32 Tensor of shape (N, S, S, K) giving  the value of
//              the ellipse_function Q=(a dx^2+b dxdy+c dy^2) of the points
//              corresponding to idx map. EWA ~ exp(-0.5Q)
//  occupance_map: a map indicating whether this pixel is occupied or not
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> RasterizePointsFine(
    const torch::Tensor &points,
    const torch::Tensor &ellipse_params,
    const torch::Tensor &cutoff_thres,
    const torch::Tensor &radii,
    const torch::Tensor &bin_points,
    const float depth_merging_thres,
    const int image_size,
    const int bin_size,
    const int points_per_pixel)
{
  if (points.is_cuda())
  {
    CHECK_CUDA(points);
    CHECK_CUDA(radii);
    CHECK_CUDA(cutoff_thres);
    CHECK_CUDA(ellipse_params);
    CHECK_CUDA(cutoff_thres);
    CHECK_CUDA(bin_points);
    return RasterizePointsFineCuda(
        points, ellipse_params, cutoff_thres, radii, bin_points, depth_merging_thres, image_size, bin_size, points_per_pixel);
  }
  else
  {
    // AT_ERROR("RasterizeFine CPU version unimplemented.");
    return RasterizePointsFineCpu(
        points, ellipse_params, cutoff_thres, radii, bin_points, depth_merging_thres, image_size, bin_size, points_per_pixel);
  }
}


// ****************************************************************************
// *                            BACKWARD PASS                                 *
// ****************************************************************************
/*
Args:
 points: Tensor of shape (P, 3) giving (packed) positions for
         points in all N pointclouds in the batch where P is the total
         number of points in the batch across all pointclouds. These points
         are expected to be in NDC coordinates in the range [-1, 1].
 radii:  (P, 2) per-point axis-aligned radius
 ellipse_params: (P, 3) the parabolic parameters.
 zbuf0:  (N, H, W) smallest value in the zbuffer
 grad_occ: (N, H, W) gradients from occupancy loss
 radii_s
 depth_merging_thres
Returns:
  grad_points: float32 Tensor of shape (N, P, 3) giving downstream gradients
 */
torch::Tensor RasterizePointsOccBackwardCpu(
    const torch::Tensor &points,         // (P, 3)
    const torch::Tensor &radii,          // (P, 2)
    const torch::Tensor &grad_occ,       // (N, H, W)
    const torch::Tensor &cloud_to_packed_first_idx,
    const torch::Tensor &num_points_per_cloud,
    const float radii_s,
    const float depth_merging_thres);

void RasterizeZbufBackwardCpu(const at::Tensor& idx, const at::Tensor& zbuf_grad, at::Tensor& point_z_grad);

#ifdef WITH_CUDA
torch::Tensor RasterizePointsOccBackwardCuda(
    const torch::Tensor &points,     // (P, 3)
    const torch::Tensor &radii,      // (P, 2)
    const torch::Tensor &grad_occ,  //  (N, H, W)
    const torch::Tensor &cloud_to_packed_first_idx,
    const torch::Tensor &num_points_per_cloud,
    const float radii_s,
    const float depth_merging_thres);

at::Tensor RasterizePointsBackwardCudaFast(
  const at::Tensor &points_sorted,    // (P, 3)
  const at::Tensor &radii_sorted,     // (P, 2)
  const at::Tensor &rs,        // (N, )
  const at::Tensor &grad_occ,  // (N, H, W)
  const at::Tensor &num_points_per_cloud, // (N,)
  const at::Tensor &cloud_to_packed_first_idx,  // (N,)
  const at::Tensor &points_grid_off, // (N, G)
  const at::Tensor &grid_params // (N, GRID_2D_PARAMS_SIZE)
);

void RasterizeZbufBackwardCuda(const at::Tensor& idx, const at::Tensor& zbuf_grad, at::Tensor& point_z_grad);
#endif

torch::Tensor RasterizePointsOccBackward(
    const torch::Tensor &points,         // (P, 3)
    const torch::Tensor &radii, // (P, 2)
    const torch::Tensor &grad_occ,  //  (N, H, W)
    const torch::Tensor &cloud_to_packed_first_idx,
    const torch::Tensor &num_points_per_cloud,
    const float radii_s,
    const float depth_merging_thres)
{
  // Check inputs are on the same device
  torch::TensorArg points_t{points, "points", 1},
      radii_t{radii, "radii", 2},
      grad_occ_t{grad_occ, "grad_occ", 3},
      cloud_to_packed_first_idx_t{
          cloud_to_packed_first_idx, "cloud_to_packed_first_idx", 4},
      num_points_per_cloud_t{num_points_per_cloud, "num_points_per_cloud", 5};
  torch::CheckedFrom c = "RasterizePointsOccBackward";
  torch::checkDim(c, points_t, 2);
  torch::checkSize(c, points_t, 1, 3);
  torch::checkSize(c, radii_t, {points_t->size(0), 2});
  torch::checkSameSize(c, cloud_to_packed_first_idx_t, num_points_per_cloud_t);
  torch::checkAllSameType(c, {points_t, radii_t, grad_occ_t});
  if (points.is_cuda())
  {
#ifdef WITH_CUDA
    CHECK_CUDA(points);
    CHECK_CUDA(radii);
    CHECK_CUDA(cloud_to_packed_first_idx);
    CHECK_CUDA(num_points_per_cloud);
    CHECK_CUDA(grad_occ);
    return RasterizePointsOccBackwardCuda(points, radii,
                                          grad_occ,
                                          cloud_to_packed_first_idx, num_points_per_cloud,
                                          radii_s, depth_merging_thres);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  else
  {
    return RasterizePointsOccBackwardCpu(points, radii,
                                         grad_occ,
                                         cloud_to_packed_first_idx, num_points_per_cloud,
                                         radii_s, depth_merging_thres);
  }
}

void RasterizeZbufBackward(
    const torch::Tensor &idx,       //  (N, H, W, K)
    const torch::Tensor &grad_zbuf, // (N, H, W, K)
    torch::Tensor &point_z_grad    // (P, 1)
    )
{
  // Check inputs are on the same device
  torch::TensorArg idx_t{idx, "idx", 1},
      grad_zbuf_t{grad_zbuf, "grad_zbuf", 2},
      point_z_grad_t{point_z_grad, "grad_occ", 3};
  torch::CheckedFrom c = "RasterizeZbufBackward";
  torch::checkDim(c, idx_t, 4);
  torch::checkSameSize(c, idx_t, grad_zbuf_t);
  torch::checkDim(c, point_z_grad_t, 2);
  torch::checkAllSameType(c, {point_z_grad_t, grad_zbuf_t});
  const int P = point_z_grad_t->size(0);
  torch::checkSize(c, point_z_grad_t, {P, 1});
  if (point_z_grad.is_cuda())
  {
#ifdef WITH_CUDA
    CHECK_CUDA(grad_zbuf);
    CHECK_CUDA(idx);
    return RasterizeZbufBackwardCuda(idx, grad_zbuf, point_z_grad);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  else
  {
    return RasterizeZbufBackwardCpu(idx, grad_zbuf, point_z_grad);
  }
}

// ****************************************************************************
// *                         MAIN ENTRY POINT                                 *
// ****************************************************************************

// This is the main entry point for the forward pass of the point rasterizer;
//
// Args:
//  points: Tensor of shape (P, 3) giving (packed) positions for
//          points in all N pointclouds in the batch where P is the total
//          number of points in the batch across all pointclouds. These points
//          are expected to be in NDC coordinates in the range [-1, 1].
//  radii: (N, 2) the axis-aligned radii for each splat
//  ellipse_params: (N, 3) the parabolic parameters.
//  cloud_to_packed_first_idx: LongTensor of shape (N) giving the index in
//                          points_packed of the first point in each pointcloud
//                          in the batch where N is the batch size.
//  num_points_per_cloud: LongTensor of shape (N) giving the number of points
//                        for each pointcloud in the batch.
//  cutoff_thres: define the region of each ellipse by cutting it off
//                once its qvalue is larger than this threshold
//  depth_merging_thres: only points that are near enough will be blended
//  image_size:  (S) Size of the image to return (in pixels)
//  points_per_pixel: (K) The number of points to return for each pixel
//  bin_size: Bin size (in pixels) for coarse-to-fine rasterization. Setting
//            bin_size=0 uses naive rasterization instead.
//  max_points_per_bin: The maximum number of points allowed to fall into each
//                      bin when using coarse-to-fine rasterization.
//
// Returns:
//  idxs: int32 Tensor of shape (N, S, S, K) giving the indices of the
//        closest K points along the z-axis for each pixel, padded with -1 for
//        pixels hit by fewer than K points. The indices refer to points in
//        points packed i.e a tensor of shape (P, 3) representing the flattened
//        points for all pointclouds in the batch.
//  zbuf: float32 Tensor of shape (N, S, S, K) giving the depth of each of each
//        closest point for each pixel
//  qvalue_map: float32 Tensor of shape (N, S, S, K) giving  the value of
//              the ellipse_function Q=(a dx^2+b dxdy+c dy^2) of the points
//              corresponding to idx map. EWA ~ exp(-0.5Q)
//  occupance_map: a map indicating whether this pixel is occupied or not
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> RasterizePoints(
    const torch::Tensor &points,
    const torch::Tensor &ellipse_params,
    const torch::Tensor &cutoff_thres,
    const torch::Tensor &radii,
    const torch::Tensor &cloud_to_packed_first_idx,
    const torch::Tensor &num_points_per_cloud,
    const float depth_merging_thres,
    const int image_size,
    const int points_per_pixel,
    const int bin_size,
    const int max_points_per_bin)
{
  // Check inputs are on the same device
  at::TensorArg points_t{points, "points", 1},
      ellipse_params_t{ellipse_params, "ellipse_params", 2},
      cutoff_thres_t{cutoff_thres, "cutoff_thres", 3},
      radii_t{radii, "radii", 4},
      cloud_to_packed_first_idx_t{
          cloud_to_packed_first_idx, "cloud_to_packed_first_idx", 5},
      num_points_per_cloud_t{num_points_per_cloud, "num_points_per_cloud", 6};
  at::CheckedFrom c = "RasterizePoints";
  torch::checkDim(c, points_t, 2);
  torch::checkSize(c, points_t, 1, 3);
  torch::checkSize(c, radii_t, {points_t->size(0), 2});
  torch::checkSize(c, ellipse_params_t, {points_t->size(0), 3});
  torch::checkSize(c, cutoff_thres_t, {points_t->size(0)});
  at::checkSameSize(c, num_points_per_cloud_t, cloud_to_packed_first_idx_t);
  if (bin_size == 0)
  {
    // Use the naive per-pixel implementation
    return RasterizePointsNaive(
        points,
        ellipse_params,
        cutoff_thres,
        radii,
        cloud_to_packed_first_idx,
        num_points_per_cloud,
        depth_merging_thres,
        image_size,
        points_per_pixel);
  }
  else
  {
    // Use coarse-to-fine rasterization
    const auto bin_points = RasterizePointsCoarse(
        points,
        radii,
        cloud_to_packed_first_idx,
        num_points_per_cloud,
        image_size,
        bin_size,
        max_points_per_bin);
    return RasterizePointsFine(
        points,
        ellipse_params,
        cutoff_thres,
        radii,
        bin_points,
        depth_merging_thres,
        image_size,
        bin_size,
        points_per_pixel);
  }
}
