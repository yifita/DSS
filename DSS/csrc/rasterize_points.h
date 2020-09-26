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

// std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> RasterizeDiscPointsNaiveCpu(
//     const torch::Tensor &points,
//     const torch::Tensor &radii,
//     const torch::Tensor &cloud_to_packed_first_idx,
//     const torch::Tensor &num_points_per_cloud,
//     const float cutoff_thres,
//     const float depth_merging_thres,
//     const int image_size,
//     const int points_per_pixel);

// #ifdef WITH_CUDA
// std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
// RasterizeDiscPointsNaiveCuda(
//     const torch::Tensor &points,
//     const torch::Tensor &radii,
//     const torch::Tensor &cloud_to_packed_first_idx,
//     const torch::Tensor &num_points_per_cloud,
//     const float cutoff_thres,
//     const float depth_merging_thres,
//     const int image_size,
//     const int points_per_pixel);
// #endif

// // Naive (forward) pointcloud rasterization: For each pixel, for each point,
// // check whether that point hits the pixel.
// //
// // Args:
// //  points: Tensor of shape (P, 3) giving (packed) positions for
// //          points in all N pointclouds in the batch where P is the total
// //          number of points in the batch across all pointclouds. These points
// //          are expected to be in NDC coordinates in the range [-1, 1].
// //  cloud_to_packed_first_idx: LongTensor of shape (N) giving the index in
// //                          points_packed of the first point in each pointcloud
// //                          in the batch where N is the batch size.
// //  num_points_per_cloud: LongTensor of shape (N) giving the number of points
// //                        for each pointcloud in the batch.
// //  radius: Radius of each point (in NDC units)
// //  image_size: (S) Size of the image to return (in pixels)
// //  points_per_pixel: (K) The number closest of points to return for each pixel
// //
// // Returns:
// //  A 4 element tuple of:
// //  idxs: int32 Tensor of shape (N, S, S, K) giving the indices of the
// //        closest K points along the z-axis for each pixel, padded with -1 for
// //        pixels hit by fewer than K points. The indices refer to points in
// //        points packed i.e a tensor of shape (P, 3) representing the flattened
// //        points for all pointclouds in the batch.
// //  zbuf: float32 Tensor of shape (N, S, S, K) giving the depth of each
// //        closest point for each pixel.
// //  dists: float32 Tensor of shape (N, S, S, K) giving squared Euclidean
// //          distance in the (NDC) x/y plane between each pixel and its K closest
// //          points along the z axis.
// std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> RasterizeDiscPointsNaive(
//     const torch::Tensor &points,
//     const torch::Tensor &radii,
//     const torch::Tensor &cloud_to_packed_first_idx,
//     const torch::Tensor &num_points_per_cloud,
//     const float cutoff_thres,
//     const float depth_merging_thres,
//     const int image_size,
//     const int points_per_pixel)
// {
//   if (points.is_cuda())
//   {
// #ifdef WITH_CUDA
//     CHECK_CUDA(points);
//     CHECK_CUDA(radii);
//     CHECK_CUDA(cloud_to_packed_first_idx);
//     CHECK_CUDA(num_points_per_cloud);
//     return RasterizeDiscPointsNaiveCuda(
//         points,
//         radii,
//         cloud_to_packed_first_idx,
//         num_points_per_cloud,
//         cutoff_thres,
//         depth_merging_thres,
//         image_size,
//         points_per_pixel);
// #else
//     AT_ERROR("Not compiled with GPU support");
// #endif
//   }
//   else
//   {
//     return RasterizeDiscPointsNaiveCpu(
//         points,
//         radii,
//         ellipse_params,
//         cloud_to_packed_first_idx,
//         num_points_per_cloud,
//         cutoff_thres,
//         depth_merging_thres,
//         image_size,
//         points_per_pixel);
//   }
// }

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
//  cloud_to_packed_first_idx: LongTensor of shape (N) giving the index in
//                          points_packed of the first point in each pointcloud
//                          in the batch where N is the batch size.
//  num_points_per_cloud: LongTensor of shape (N) giving the number of points
//                        for each pointcloud in the batch.
//  radius: Radius of each point (in NDC units)
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
//  radii: (N, 2) the axis-aligned radii for each splat
//  cloud_to_packed_first_idx: LongTensor of shape (N) giving the index in
//                          points_packed of the first point in each pointcloud
//                          in the batch where N is the batch size.
//  num_points_per_cloud: LongTensor of shape (N) giving the number of points
//                        for each pointcloud in the batch.
//  image_size: Size of the image to generate (in pixels)
//  bin_size: Size of each bin within the image (in pixels)
//
// Returns:
//  points_per_bin: Tensor of shape (N, num_bins, num_bins) giving the number
//                  of points that fall in each bin
//  bin_points: Tensor of shape (N, num_bins, num_bins, K) giving the indices
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
    AT_ERROR("RasterizeCoarse CPU version unimplemented");
  }
}

// ****************************************************************************
// *                            FINE RASTERIZATION                            *
// ****************************************************************************

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
//  radii: (N, 2) the axis-aligned radii for each splat
//  ellipse_params: (N, 3) the parabolic parameters.
//  bin_points: int32 Tensor of shape (N, B, B, M) giving the indices of points
//              that fall into each bin (output from coarse rasterization)
//  cutoff_thres: define the region of each ellipse by cutting it off
//                once its qvalue is larger than this threshold
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
    AT_ERROR("RasterizeFine CPU version unimplemented.");
  }
}

// std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> RasterizeDiscPointsFineCuda(
//     const torch::Tensor &points,
//     const torch::Tensor &radii,
//     const torch::Tensor &bin_points,
//     const float cutoff_thres,
//     const float depth_merging_thres,
//     const int image_size,
//     const int bin_size,
//     const int points_per_pixel);
// // Args:
// //  points: Tensor of shape (P, 3) giving (packed) positions for
// //          points in all N pointclouds in the batch where P is the total
// //          number of points in the batch across all pointclouds. These points
// //          are expected to be in NDC coordinates in the range [-1, 1].
// //  radii: (N, 2) the axis-aligned radii for each splat
// //  ellipse_params: (N, 3) the parabolic parameters.
// //  bin_points: int32 Tensor of shape (N, B, B, M) giving the indices of points
// //              that fall into each bin (output from coarse rasterization)
// //  cutoff_thres: define the region of each ellipse by cutting it off
// //                once its qvalue is larger than this threshold
// //  depth_merging_thres: only points that are near enough will be blended
// //  image_size: Size of image to generate (in pixels)
// //  bin_size: Size of each bin (in pixels)
// //  points_per_pixel: How many points to rasterize for each pixel
// //
// // Returns (same as rasterize_points):
// //  idxs: int32 Tensor of shape (N, S, S, K) giving the indices of the
// //        closest K points along the z-axis for each pixel, padded with -1 for
// //        pixels hit by fewer than K points. The indices refer to points in
// //        points packed i.e a tensor of shape (P, 3) representing the flattened
// //        points for all pointclouds in the batch.
// //  zbuf: float32 Tensor of shape (N, S, S, K) giving the depth of each of each
// //        closest point for each pixel
// //  qvalue_map: float32 Tensor of shape (N, S, S, K) giving  the value of
// //              the ellipse_function Q=(a dx^2+b dxdy+c dy^2) of the points
// //              corresponding to idx map. EWA ~ exp(-0.5Q)
// //  occupance_map: a map indicating whether this pixel is occupied or not
// std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> RasterizeDiscPointsFine(
//     const torch::Tensor &points,
//     const torch::Tensor &radii,
//     const torch::Tensor &bin_points,
//     const float cutoff_thres,
//     const float depth_merging_thres,
//     const int image_size,
//     const int bin_size,
//     const int points_per_pixel)
// {
//   if (points.is_cuda())
//   {
//     CHECK_CUDA(points);
//     CHECK_CUDA(radii);
//     CHECK_CUDA(bin_points);
//     return RasterizeDiscPointsFineCuda(
//         points, radii, bin_points, cutoff_thres, depth_merging_thres, image_size, bin_size, points_per_pixel);
//   }
//   else
//   {
//     AT_ERROR("RasterizeDiscPointsFine CPU version unimplemented.");
//   }
// }

// ****************************************************************************
// *                            BACKWARD PASS                                 *
// ****************************************************************************

/* NOTE(yifan) Three options. We should implement them all for comparison. I'm going to implement 2:
(RasterizePointsWeightsBackward)
1. first one similar to pytorch3d, gradients defined only where the weights are nonzero. In this case,
we need idx and ellipse_params. radii is not required
(RasterizePointsOccRBFBackward)
2. compute gradients also when weights = 0, but unlike DSS, the gradient value is based on exp(-0.5Q).
This means, like forward pass, we need to re-evaluate the neighborhood of each point
(maybe a multiple of radii) and compute the gradients using
dx, dy and ellipse_params.
(RasterizePointsOccBackward)
3. dss gradients, like 2 we need to do a forward pass style sweep over all pixels and all points,
the gradient is based on 1/dist
*/
/*
Args:
 points: Tensor of shape (P, 3) giving (packed) positions for
         points in all N pointclouds in the batch where P is the total
         number of points in the batch across all pointclouds. These points
         are expected to be in NDC coordinates in the range [-1, 1].
 ellipse_params: (P, 3) the parabolic parameters.
 idxs:   Tensor (N,H,W,K) from forward pass
 grad_zbuf: Gradients for the zbuf
 grad_qvalues: gradients qvalues received
Returns:
  grad_points: float32 Tensor of shape (N, P, 3) giving downstream gradients
 */
std::tuple<torch::Tensor, torch::Tensor> RasterizePointsWeightsBackwardCpu(
    const torch::Tensor &points,         // (P, 3)
    const torch::Tensor &ellipse_params, // (P, 3)
    const torch::Tensor &cutoff,         // (P,) or (1,)
    const torch::Tensor &idxs,           // (N, H, W, K)
    const torch::Tensor &grad_zbuf,      // (N, H, W, K)
    const torch::Tensor &grad_qvalues);
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
 radii_s: a scaler to increase the range of backpropagation pixels by radii*s, and qvalue*radii_s*radii_s
 depth_merging_thres
Returns:
  grad_points: float32 Tensor of shape (N, P, 3) giving downstream gradients
 */
std::tuple<torch::Tensor, torch::Tensor> RasterizePointsOccRBFBackwardCpu(
    const torch::Tensor &points,         // (P, 3)
    const torch::Tensor &ellipse_params, // (P, 3)
    const torch::Tensor &cutoff_thres,
    const torch::Tensor &radii,     // (P, 2)
    const torch::Tensor &idx,       // (N, H, W, K)
    const torch::Tensor &zbuf0,     // (N, H, W)
    const torch::Tensor &grad_occ,  //  (N, H, W)
    const torch::Tensor &grad_zbuf, // (N, H, W)
    const torch::Tensor &cloud_to_packed_first_idx,
    const torch::Tensor &num_points_per_cloud,
    const float radii_s,
    const float depth_merging_thres);

/*
Similar to OccRBF but use 1/d as the gradient
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
std::tuple<torch::Tensor, torch::Tensor> RasterizePointsOccBackwardCpu(
    const torch::Tensor &points,         // (P, 3)
    const torch::Tensor &ellipse_params, // (P, 3)
    const torch::Tensor &cutoff_thres,   // (P,) or (1,)
    const torch::Tensor &radii,          // (P, 2)
    const torch::Tensor &idx,            // (N, H, W, K)
    const torch::Tensor &zbuf0,          // (N, H, W)
    const torch::Tensor &grad_occ,       //  (N, H, W)
    const torch::Tensor &grad_zbuf,      // (N, H, W)
    const torch::Tensor &cloud_to_packed_first_idx,
    const torch::Tensor &num_points_per_cloud,
    const float radii_s,
    const float depth_merging_thres);

#ifdef WITH_CUDA
std::tuple<torch::Tensor, torch::Tensor> RasterizePointsWeightsBackwardCuda(
    const torch::Tensor &points,         // (P, 3)
    const torch::Tensor &ellipse_params, // (P, 3)
    const torch::Tensor &cutoff,         // (P,) or (1,)
    const torch::Tensor &idxs,           // (N, H, W, K)
    const torch::Tensor &grad_zbuf,      // (N, H, W, K)
    const torch::Tensor &grad_qvalues);
#endif

#ifdef WITH_CUDA
std::tuple<torch::Tensor, torch::Tensor> RasterizePointsOccRBFBackwardCuda(
    const torch::Tensor &points,         // (P, 3)
    const torch::Tensor &ellipse_params, // (P, 3)
    const torch::Tensor &cutoff_thres,   // (P,) or (1,)
    const torch::Tensor &radii,          // (P, 2)
    const torch::Tensor &idx,            // (N, H, W, K)
    const torch::Tensor &zbuf0,          // (N, H, W)
    const torch::Tensor &grad_occ,       //  (N, H, W)
    const torch::Tensor &grad_zbuf,      // (N, H, W)
    const torch::Tensor &cloud_to_packed_first_idx,
    const torch::Tensor &num_points_per_cloud,
    const float radii_s,
    const float depth_merging_thres);
#endif

#ifdef WITH_CUDA
std::tuple<torch::Tensor, torch::Tensor> RasterizePointsOccBackwardCuda(
    const torch::Tensor &points,         // (P, 3)
    const torch::Tensor &ellipse_params, // (P, 3)
    const torch::Tensor &cutoff_thres,
    const torch::Tensor &radii,     // (P, 2)
    const torch::Tensor &idx,       // (N, H, W, K)
    const torch::Tensor &zbuf0,     // (N, H, W)
    const torch::Tensor &grad_occ,  //  (N, H, W)
    const torch::Tensor &grad_zbuf, // (N, H, W)
    const torch::Tensor &cloud_to_packed_first_idx,
    const torch::Tensor &num_points_per_cloud,
    const float radii_s,
    const float depth_merging_thres);
#endif

// torch::Tensor RasterizeDiscPointsOccBackwardCpu(
//     const torch::Tensor &points, // (P, 3)
//     const torch::Tensor &radii,  // (P, 2)
//     // const torch::Tensor &ellipse_params, // (P, 3)
//     const torch::Tensor &idx,       // (N, H, W, K)
//     const torch::Tensor &zbuf0,     // (N, H, W)
//     const torch::Tensor &grad_occ,  //  (N, H, W)
//     const torch::Tensor &grad_zbuf, // (N, H, W)
//     const torch::Tensor &cloud_to_packed_first_idx,
//     const torch::Tensor &num_points_per_cloud,
//     const float radii_s,
//     const float cutoff_thres,
//     const float depth_merging_thres);

// #ifdef WITH_CUDA
// torch::Tensor RasterizeDiscPointsOccBackwardCuda(
//     const torch::Tensor &points,    // (P, 3)
//     const torch::Tensor &radii,     // (P, 2)
//     const torch::Tensor &idx,       // (N, H, W, K)
//     const torch::Tensor &zbuf0,     // (N, H, W)
//     const torch::Tensor &grad_occ,  //  (N, H, W)
//     const torch::Tensor &grad_zbuf, // (N, H, W)
//     const torch::Tensor &cloud_to_packed_first_idx,
//     const torch::Tensor &num_points_per_cloud,
//     const float radii_s,
//     const float cutoff_thres,
//     const float depth_merging_thres);
// #endif

std::tuple<torch::Tensor, torch::Tensor> RasterizePointsWeightsBackward(
    const torch::Tensor &points,         // (P, 3)
    const torch::Tensor &ellipse_params, // (P, 3)
    const torch::Tensor &cutoff,         // (P,) (1,)
    const torch::Tensor &idxs,           // (N, H, W, K)
    const torch::Tensor &grad_zbuf,      // (N, H, W, K)
    const torch::Tensor &grad_qvalues)
{
  if (points.is_cuda())
  {
#ifdef WITH_CUDA
    CHECK_CUDA(points);
    CHECK_CUDA(idxs);
    CHECK_CUDA(cutoff);
    CHECK_CUDA(grad_zbuf);
    CHECK_CUDA(grad_qvalues);
    return RasterizePointsWeightsBackwardCuda(points, ellipse_params, cutoff, idxs, grad_zbuf, grad_qvalues);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  else
  {
    return RasterizePointsWeightsBackwardCpu(points, ellipse_params, cutoff, idxs, grad_zbuf, grad_qvalues);
  }
}

std::tuple<torch::Tensor, torch::Tensor> RasterizePointsOccRBFBackward(
    const torch::Tensor &points,         // (P, 3)
    const torch::Tensor &ellipse_params, // (P, 3)
    const torch::Tensor &cutoff_thres,
    const torch::Tensor &radii, // (P, 2)
    const torch::Tensor &idx,   // (N, H, W, K)
    const torch::Tensor &zbuf0, // (N, H, W)
    // the closest depth rendered at this pixel
    const torch::Tensor &grad_occ,  //  (N, H, W)
    const torch::Tensor &grad_zbuf, // (N, H, W)
    const torch::Tensor &cloud_to_packed_first_idx,
    const torch::Tensor &num_points_per_cloud,
    const float radii_s,
    const float depth_merging_thres)
{
  // Check inputs are on the same device
  torch::TensorArg points_t{points, "points", 1},
      ellipse_params_t{ellipse_params, "ellipse_params", 2},
      cutoff_thres_t{cutoff_thres, "cutoff_thres", 3},
      radii_t{radii, "radii", 4},
      idx_t{idx, "idx", 5},
      zbuf0_t{zbuf0, "zbuf0", 6},
      grad_occ_t{grad_occ, "grad_occ", 7},
      grad_zbuf_t{grad_zbuf, "grad_zbuf", 8},
      cloud_to_packed_first_idx_t{
          cloud_to_packed_first_idx, "cloud_to_packed_first_idx", 9},
      num_points_per_cloud_t{num_points_per_cloud, "num_points_per_cloud", 10};
  torch::CheckedFrom c = "RasterizePointsOccRBFBackward";
  torch::checkDim(c, points_t, 2);
  torch::checkSize(c, points_t, 1, 3);
  torch::checkSize(c, radii_t, {points_t->size(0), 2});
  torch::checkSize(c, ellipse_params_t, {points_t->size(0), 3});
  torch::checkSameSize(c, zbuf0_t, grad_occ_t);
  torch::checkSameSize(c, idx_t, grad_zbuf_t);
  torch::checkDim(c, idx_t, 4);
  torch::checkDim(c, zbuf0_t, 3);
  torch::checkAllSameType(c, {points_t, radii_t, ellipse_params_t, zbuf0_t, grad_occ_t, grad_zbuf_t});
  torch::checkSameSize(c, num_points_per_cloud_t, cloud_to_packed_first_idx_t);
  if (points.is_cuda())
  {
#ifdef WITH_CUDA
    CHECK_CUDA(points);
    CHECK_CUDA(ellipse_params);
    CHECK_CUDA(cutoff_thres);
    CHECK_CUDA(radii);
    CHECK_CUDA(zbuf0);
    CHECK_CUDA(cloud_to_packed_first_idx);
    CHECK_CUDA(num_points_per_cloud);
    CHECK_CUDA(grad_occ);
    return RasterizePointsOccRBFBackwardCuda(points, ellipse_params, cutoff_thres, radii, idx,
                                             zbuf0, grad_occ, grad_zbuf,
                                             cloud_to_packed_first_idx, num_points_per_cloud,
                                             radii_s, depth_merging_thres);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  else
  {
    return RasterizePointsOccRBFBackwardCpu(points, ellipse_params, cutoff_thres, radii, idx,
                                            zbuf0, grad_occ, grad_zbuf,
                                            cloud_to_packed_first_idx, num_points_per_cloud,
                                            radii_s, depth_merging_thres);
  }
}
std::tuple<torch::Tensor, torch::Tensor> RasterizePointsOccBackward(
    const torch::Tensor &points,         // (P, 3)
    const torch::Tensor &ellipse_params, // (P, 3)
    const torch::Tensor &cutoff_thres,
    const torch::Tensor &radii, // (P, 2)
    const torch::Tensor &idx,   // (N, H, W, K)
    const torch::Tensor &zbuf0, // (N, H, W)
    // the closest depth rendered at this pixel
    const torch::Tensor &grad_occ,  //  (N, H, W)
    const torch::Tensor &grad_zbuf, // (N, H, W)
    const torch::Tensor &cloud_to_packed_first_idx,
    const torch::Tensor &num_points_per_cloud,
    const float radii_s,
    const float depth_merging_thres)
{
  // Check inputs are on the same device
  torch::TensorArg points_t{points, "points", 1},
      ellipse_params_t{ellipse_params, "ellipse_params", 2},
      cutoff_thres_t{cutoff_thres, "cutoff_thres", 3},
      radii_t{radii, "radii", 4},
      idx_t{idx, "idx", 5},
      zbuf0_t{zbuf0, "zbuf0", 6},
      grad_occ_t{grad_occ, "grad_occ", 7},
      grad_zbuf_t{grad_zbuf, "grad_zbuf", 8},
      cloud_to_packed_first_idx_t{
          cloud_to_packed_first_idx, "cloud_to_packed_first_idx", 9},
      num_points_per_cloud_t{num_points_per_cloud, "num_points_per_cloud", 10};
  torch::CheckedFrom c = "RasterizePointsOccBackward";
  torch::checkDim(c, points_t, 2);
  torch::checkSize(c, points_t, 1, 3);
  torch::checkSize(c, radii_t, {points_t->size(0), 2});
  torch::checkSize(c, ellipse_params_t, {points_t->size(0), 3});
  torch::checkSameSize(c, zbuf0_t, grad_occ_t);
  torch::checkSameSize(c, idx_t, grad_zbuf_t);
  torch::checkSameSize(c, cloud_to_packed_first_idx_t, num_points_per_cloud_t);
  torch::checkDim(c, idx_t, 4);
  torch::checkDim(c, zbuf0_t, 3);
  torch::checkAllSameType(c, {points_t, radii_t, zbuf0_t, cutoff_thres_t, grad_occ_t, grad_zbuf_t});
  if (points.is_cuda())
  {
#ifdef WITH_CUDA
    CHECK_CUDA(points);
    CHECK_CUDA(radii);
    CHECK_CUDA(ellipse_params);
    CHECK_CUDA(zbuf0);
    CHECK_CUDA(cloud_to_packed_first_idx);
    CHECK_CUDA(num_points_per_cloud);
    CHECK_CUDA(grad_occ);
    return RasterizePointsOccBackwardCuda(points, ellipse_params, cutoff_thres, radii,
                                          idx, zbuf0, grad_occ, grad_zbuf,
                                          cloud_to_packed_first_idx, num_points_per_cloud,
                                          radii_s, depth_merging_thres);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  else
  {
    return RasterizePointsOccBackwardCpu(points, ellipse_params, cutoff_thres, radii,
                                         idx, zbuf0, grad_occ, grad_zbuf,
                                         cloud_to_packed_first_idx, num_points_per_cloud,
                                         radii_s, depth_merging_thres);
  }
}

// torch::Tensor RasterizeDiscPointsOccBackward(
//     const torch::Tensor &points, // (P, 3)
//     const torch::Tensor &radii,  // (P, 2)
//     const torch::Tensor &idx,    // (N, H, W, K)
//     const torch::Tensor &zbuf0,  // (N, H, W)
//     // the closest depth rendered at this pixel
//     const torch::Tensor &grad_occ,  //  (N, H, W)
//     const torch::Tensor &grad_zbuf, // (N, H, W)
//     const torch::Tensor &cloud_to_packed_first_idx,
//     const torch::Tensor &num_points_per_cloud,
//     const float radii_s,
//     const float cutoff_thres,
//     const float depth_merging_thres)
// {
//   // Check inputs are on the same device
//   torch::TensorArg points_t{points, "points", 1},
//       radii_t{radii, "radii", 2},
//       idx_t{idx, "idx", 4},
//       zbuf0_t{zbuf0, "zbuf0", 5},
//       grad_occ_t{grad_occ, "grad_occ", 6},
//       grad_zbuf_t{grad_zbuf, "grad_zbuf", 7},
//       cloud_to_packed_first_idx_t{
//           cloud_to_packed_first_idx, "cloud_to_packed_first_idx", 8},
//       num_points_per_cloud_t{num_points_per_cloud, "num_points_per_cloud", 9};
//   torch::CheckedFrom c = "RasterizePointsOccBackward";
//   torch::checkDim(c, points_t, 2);
//   torch::checkSize(c, points_t, 1, 3);
//   torch::checkSize(c, radii_t, {points_t->size(0), 2});
//   // torch::checkSize(c, ellipse_params_t, {points_t->size(0), 3});
//   torch::checkSameSize(c, zbuf0_t, grad_occ_t);
//   torch::checkSameSize(c, idx_t, grad_zbuf_t);
//   torch::checkSameSize(c, cloud_to_packed_first_idx_t, num_points_per_cloud_t);
//   torch::checkDim(c, idx_t, 4);
//   torch::checkDim(c, zbuf0_t, 3);
//   torch::checkAllSameType(c, {points_t, radii_t, zbuf0_t, grad_occ_t, grad_zbuf_t});
//   if (points.is_cuda())
//   {
// #ifdef WITH_CUDA
//     CHECK_CUDA(points);
//     CHECK_CUDA(radii);
//     // CHECK_CUDA(ellipse_params);
//     CHECK_CUDA(zbuf0);
//     CHECK_CUDA(cloud_to_packed_first_idx);
//     CHECK_CUDA(num_points_per_cloud);
//     CHECK_CUDA(grad_occ);
//     return RasterizeDiscPointsOccBackwardCuda(points, radii, idx,
//                                               zbuf0, grad_occ, grad_zbuf,
//                                               cloud_to_packed_first_idx, num_points_per_cloud,
//                                               radii_s, cutoff_thres, depth_merging_thres);
// #else
//     AT_ERROR("Not compiled with GPU support");
// #endif
//   }
//   else
//   {
//     return RasterizeDiscPointsOccBackwardCpu(points, radii,
//                                              idx, zbuf0, grad_occ, grad_zbuf,
//                                              cloud_to_packed_first_idx, num_points_per_cloud,
//                                              radii_s, cutoff_thres, depth_merging_thres);
//   }
// }
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

// std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> RasterizeDiscPoints(
//     const torch::Tensor &points,
//     const torch::Tensor &radii,
//     const torch::Tensor &cloud_to_packed_first_idx,
//     const torch::Tensor &num_points_per_cloud,
//     const float cutoff_thres,
//     const float depth_merging_thres,
//     const int image_size,
//     const int points_per_pixel,
//     const int bin_size,
//     const int max_points_per_bin)
// {
//   // Check inputs are on the same device
//   at::TensorArg points_t{points, "points", 1},
//       radii_t{radii, "radii", 2},
//       cloud_to_packed_first_idx_t{
//           cloud_to_packed_first_idx, "cloud_to_packed_first_idx", 4},
//       num_points_per_cloud_t{num_points_per_cloud, "num_points_per_cloud", 5};
//   at::CheckedFrom c = "RasterizePoints";
//   torch::checkDim(c, points_t, 2);
//   torch::checkSize(c, points_t, 1, 3);
//   torch::checkSize(c, radii_t, {points_t->size(0), 2});
//   at::checkSameSize(c, num_points_per_cloud_t, cloud_to_packed_first_idx_t);
//   if (bin_size == 0)
//   {
//     // Use the naive per-pixel implementation
//     return RasterizeDiscPointsNaive(
//         points,
//         radii,
//         cloud_to_packed_first_idx,
//         num_points_per_cloud,
//         cutoff_thres,
//         depth_merging_thres,
//         image_size,
//         points_per_pixel);
//   }
//   else
//   {
//     // Use coarse-to-fine rasterization
//     const auto bin_points = RasterizePointsCoarse(
//         points,
//         radii,
//         cloud_to_packed_first_idx,
//         num_points_per_cloud,
//         image_size,
//         bin_size,
//         max_points_per_bin);
//     return RasterizeDiscPointsFine(
//         points,
//         radii,
//         bin_points,
//         cutoff_thres,
//         depth_merging_thres,
//         image_size,
//         bin_size,
//         points_per_pixel);
//   }
// }