// TODO(lixin)
#include <torch/extension.h>
#include <queue>
#include <tuple>
#include <algorithm>

// Given a pixel coordinate 0 <= i < S, convert it to a normalized device
// coordinate in the range [-1, 1]. The NDC range is divided into S evenly-sized
// pixels, and assume that each pixel falls in the *center* of its range.
static float PixToNdc(const int i, const int S)
{
  // NDC x-offset + (i * pixel_width + half_pixel_width)
  return -1 + (2 * i + 1.0f) / S;
}

template <typename T>
int sgn(T val)
{
  return (T(0) < val) - (val < T(0));
}

inline float Qvalue(const float dx, const float dy, const float a, const float b, const float c)
{
  return a * dx * dx + b * dx * dy + c * dy * dy;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> RasterizePointsNaiveCpu(
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
  const int32_t N = cloud_to_packed_first_idx.size(0); // batch_size.

  const int S = image_size;
  const int K = points_per_pixel;

  // Initialize output tensors.
  auto int_opts = num_points_per_cloud.options().dtype(torch::kInt32);
  auto float_opts = points.options().dtype(torch::kFloat32);
  torch::Tensor point_idxs = torch::full({N, S, S, K}, -1, int_opts);
  torch::Tensor zbuf = torch::full({N, S, S, K}, -1, float_opts);
  torch::Tensor qvalue = torch::full({N, S, S, K}, -1, float_opts);
  torch::Tensor occupancy = torch::full({N, S, S}, 0, float_opts);

  auto points_a = points.accessor<float, 2>();
  auto radii_a = radii.accessor<float, 2>();
  auto ellipse_params_a = ellipse_params.accessor<float, 2>();
  auto cutoff_a = cutoff_thres.accessor<float, 1>();

  auto point_idxs_a = point_idxs.accessor<int32_t, 4>();
  auto zbuf_a = zbuf.accessor<float, 4>();
  auto qvalue_a = qvalue.accessor<float, 4>();
  auto occupancy_a = occupancy.accessor<float, 3>();

  for (int n = 0; n < N; ++n)
  {
    // Loop through each pointcloud in the batch.
    // Get the start index of the points in points_packed and the num points
    // in the point cloud.
    const int point_start_idx =
        cloud_to_packed_first_idx[n].item().to<int32_t>();
    const int point_stop_idx =
        (point_start_idx + num_points_per_cloud[n].item().to<int32_t>());

    for (int yi = 0; yi < S; ++yi)
    {
      // Reverse the order of yi so that +Y is pointing upwards in the image.
      const int yidx = S - 1 - yi;
      const float yf = PixToNdc(yidx, S);

      for (int xi = 0; xi < S; ++xi)
      {
        // Reverse the order of xi so that +X is pointing to the left in the
        // image.
        const int xidx = S - 1 - xi;
        const float xf = PixToNdc(xidx, S);

        // Use a priority queue to hold (z, idx, qvalue)
        std::priority_queue<std::tuple<float, int, float>> q;
        for (int p = point_start_idx; p < point_stop_idx; ++p)
        {
          const float px = points_a[p][0];
          const float py = points_a[p][1];
          const float pz = points_a[p][2];
          if (pz < 0)
          {
            continue;
          }
          const float dx = xf - px;
          const float dy = yf - py;
          const float radiix = radii_a[p][0];
          const float radiiy = radii_a[p][1];
          if (abs(dx) > radiix && abs(dy) > radiiy)
            continue;
          // The current point hit the current pixel
          float qvalue_p = Qvalue(dx, dy, ellipse_params_a[p][0], ellipse_params_a[p][1], ellipse_params_a[p][2]);
          if (qvalue_p > cutoff_a[p])
            continue;
          q.emplace(pz, p, qvalue_p);
          // printf("dx dy = (%g, %g), radii = (%g, %g), qsize = %lu, queue top = (%g, %d, %g)\n", dx, dy, radiix, radiiy,
          //        q.size(), std::get<0>(q.top()), std::get<1>(q.top()), std::get<2>(q.top()));
          if ((int)q.size() > K)
          {
            q.pop();
          }
        }
        // Now all the points have been seen, so pop elements off the queue
        // one by one and write them into the output tensors.
        while (!q.empty())
        {
          auto t = q.top();
          q.pop();
          int i = q.size();
          zbuf_a[n][yi][xi][i] = std::get<0>(t);
          point_idxs_a[n][yi][xi][i] = std::get<1>(t);
          qvalue_a[n][yi][xi][i] = std::get<2>(t);
        }
        // traverse zbuf again to remove elements according to depth_merging_thres
        if (point_idxs_a[n][yi][xi][0] >= 0 && zbuf_a[n][yi][xi][0] >= 0)
        {
          // set occupancy_map
          occupancy_a[n][yi][xi] = 1.0f;
          float closest_z = zbuf_a[n][yi][xi][0];
          for (int i = 1; i < K; i++)
          {
            if ((zbuf_a[n][yi][xi][i] - closest_z) > depth_merging_thres)
            {
              point_idxs_a[n][yi][xi][i] = -1;
              zbuf_a[n][yi][xi][i] = -1;
              qvalue_a[n][yi][xi][i] = -1;
            }
          }
        }
      }
    }
  }
  return std::make_tuple(point_idxs, zbuf, qvalue, occupancy);
}

// TODO(lixin)
torch::Tensor RasterizePointsCoarseCpu(
    const torch::Tensor &points,                    // (P, 3)
    const torch::Tensor &radii,                     // (P, 2)
    const torch::Tensor &cloud_to_packed_first_idx, // (N)
    const torch::Tensor &num_points_per_cloud,      // (N)
    const int image_size,
    const int bin_size,
    const int max_points_per_bin)
{
  const int32_t N = cloud_to_packed_first_idx.size(0); // batch_size.

  const int B = 1 + (image_size - 1) / bin_size; // Integer division round up
  const int M = max_points_per_bin;
  auto opts = num_points_per_cloud.options().dtype(torch::kInt32);
  torch::Tensor points_per_bin = torch::zeros({N, B, B}, opts);
  torch::Tensor bin_points = torch::full({N, B, B, M}, -1, opts);

  auto points_a = points.accessor<float, 2>();
  auto radii_a = radii.accessor<float, 2>();
  auto points_per_bin_a = points_per_bin.accessor<int32_t, 3>();
  auto bin_points_a = bin_points.accessor<int32_t, 4>();

  const float pixel_width = 2.0f / image_size;
  const float bin_width = pixel_width * bin_size;

  for (int n = 0; n < N; ++n)
  {
    // Loop through each pointcloud in the batch.
    // Get the start index of the points in points_packed and the num points
    // in the point cloud.
    const int point_start_idx =
        cloud_to_packed_first_idx[n].item().to<int32_t>();
    const int point_stop_idx =
        (point_start_idx + num_points_per_cloud[n].item().to<int32_t>());

    float bin_y_min = -1.0f;
    float bin_y_max = bin_y_min + bin_width;

    // Iterate through the horizontal bins from top to bottom.
    for (int by = 0; by < B; by++)
    {
      float bin_x_min = -1.0f;
      float bin_x_max = bin_x_min + bin_width;

      // Iterate through bins on this horizontal line, left to right.
      for (int bx = 0; bx < B; bx++)
      {
        int32_t points_hit = 0;
        for (int p = point_start_idx; p < point_stop_idx; ++p)
        {
          float px = points_a[p][0];
          float py = points_a[p][1];
          float pz = points_a[p][2];
          if (pz < 0)
          {
            continue;
          }
          float point_x_min = px - radii_a[p][0];
          float point_x_max = px + radii_a[p][0];
          float point_y_min = py - radii_a[p][1];
          float point_y_max = py + radii_a[p][1];

          // Use a half-open interval so that points exactly on the
          // boundary between bins will fall into exactly one bin.
          bool x_hit = (point_x_min <= bin_x_max) && (bin_x_min <= point_x_max);
          bool y_hit = (point_y_min <= bin_y_max) && (bin_y_min <= point_y_max);
          if (x_hit && y_hit)
          {
            // Got too many points for this bin, so throw an error.
            if (points_hit >= max_points_per_bin)
            {
              AT_ERROR("Got too many points per bin");
            }
            // The current point falls in the current bin, so
            // record it.
            bin_points_a[n][by][bx][points_hit] = p;
            points_hit++;
          }
        }
        // Record the number of points found in this bin
        points_per_bin_a[n][by][bx] = points_hit;

        // Shift the bin to the right for the next loop iteration
        bin_x_min = bin_x_max;
        bin_x_max = bin_x_min + bin_width;
      }
      // Shift the bin down for the next loop iteration
      bin_y_min = bin_y_max;
      bin_y_max = bin_y_min + bin_width;
    }
  }
  return bin_points;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> RasterizePointsFineCpu(
    const torch::Tensor &points,         // (P, 3)
    const torch::Tensor &ellipse_params, // (P, 3)
    const torch::Tensor &cutoff_thres,   // (P,)
    const torch::Tensor &radii,          // (P,2)
    const torch::Tensor &bin_points,     // (N, B, B, M)
    const float depth_merging_thres,
    const int image_size,
    const int bin_size,
    const int points_per_pixel)
{
  const int N = bin_points.size(0); // batch_size.
  const int M = bin_points.size(3);
  const int S = image_size;
  const int K = points_per_pixel;

  // Initialize output tensors.
  auto int_opts = bin_points.options().dtype(torch::kInt32);
  auto float_opts = points.options().dtype(torch::kFloat32);
  torch::Tensor point_idxs = torch::full({N, S, S, K}, -1, int_opts);
  torch::Tensor zbuf = torch::full({N, S, S, K}, -1, float_opts);
  torch::Tensor qvalue = torch::full({N, S, S, K}, -1, float_opts);
  torch::Tensor occupancy = torch::full({N, S, S}, 0, float_opts);

  auto points_a = points.accessor<float, 2>();
  auto radii_a = radii.accessor<float, 2>();
  auto ellipse_params_a = ellipse_params.accessor<float, 2>();
  auto cutoff_a = cutoff_thres.accessor<float, 1>();
  auto bin_points_a = bin_points.accessor<int, 4>();

  auto point_idxs_a = point_idxs.accessor<int32_t, 4>();
  auto zbuf_a = zbuf.accessor<float, 4>();
  auto qvalue_a = qvalue.accessor<float, 4>();
  auto occupancy_a = occupancy.accessor<float, 3>();

  for (int n = 0; n < N; ++n)
  {
    // Loop through each pointcloud in the batch.

    for (int yi = 0; yi < S; ++yi)
    {
      // TODO: for consistency with bin_points, here we use yi/xi for yf/xf.
      // Only reverse it when we write back
      const int by = yi / bin_size;
      const float yf = PixToNdc(yi, S);
      const int yidx = S - 1 - yi;

      // Reverse the order of yi so that +Y is pointing upwards in the image.
      // const int yidx = S - 1 - yi;
      // const float yf = PixToNdc(yidx, S);

      for (int xi = 0; xi < S; ++xi)
      {
        // TODO: for consistency with bin_points, here we use yi/xi for yf/xf.
        // Only reverse it when we write back
        const int bx = xi / bin_size;
        const float xf = PixToNdc(xi, S);
        const int xidx = S - 1 - xi;

        // Reverse the order of xi so that +X is pointing to the left in the
        // image.
        // const int xidx = S - 1 - xi;
        // const float xf = PixToNdc(xidx, S);

        // Use a priority queue to hold (z, idx, qvalue)
        std::priority_queue<std::tuple<float, int, float>> q;
        // loop over all points in this bin
        for (int i = 0; i < M; ++i)
        {
          const int p = bin_points_a[n][by][bx][i];
          if (p < 0)
          {
            // bin_points uses -1 as a sentinal value
            break;
          }
          const float px = points_a[p][0];
          const float py = points_a[p][1];
          const float pz = points_a[p][2];
          if (pz < 0)
          {
            continue;
          }
          const float dx = xf - px;
          const float dy = yf - py;
          const float radiix = radii_a[p][0];
          const float radiiy = radii_a[p][1];
          if (abs(dx) > radiix || abs(dy) > radiiy)
            continue;
          // The current point hit the current pixel
          float qvalue_p = Qvalue(dx, dy, ellipse_params_a[p][0], ellipse_params_a[p][1], ellipse_params_a[p][2]);
          if (qvalue_p > cutoff_a[p])
            continue;
          q.emplace(pz, p, qvalue_p);
          // printf("dx dy = (%g, %g), radii = (%g, %g), qsize = %lu, queue top = (%g, %d, %g)\n", dx, dy, radiix, radiiy,
          //        q.size(), std::get<0>(q.top()), std::get<1>(q.top()), std::get<2>(q.top()));
          if ((int)q.size() > K)
          {
            q.pop();
          }
        }
        // Now all the points have been seen, so pop elements off the queue
        // one by one and write them into the output tensors.
        while (!q.empty())
        {
          auto t = q.top();
          q.pop();
          int i = q.size();
          zbuf_a[n][yidx][xidx][i] = std::get<0>(t);
          point_idxs_a[n][yidx][xidx][i] = std::get<1>(t);
          qvalue_a[n][yidx][xidx][i] = std::get<2>(t);
        }
        // traverse zbuf again to remove elements according to depth_merging_thres
        if (point_idxs_a[n][yi][xi][0] >= 0 && zbuf_a[n][yi][xi][0] >= 0)
        {
          // set occupancy_map
          occupancy_a[n][yi][xi] = 1.0f;
          float closest_z = zbuf_a[n][yi][xi][0];
          for (int i = 1; i < K; i++)
          {
            if ((zbuf_a[n][yi][xi][i] - closest_z) > depth_merging_thres)
            {
              point_idxs_a[n][yi][xi][i] = -1;
              zbuf_a[n][yi][xi][i] = -1;
              qvalue_a[n][yi][xi][i] = -1;
            }
          }
        }
      }
    }
  }
  return std::make_tuple(point_idxs, zbuf, qvalue, occupancy);
}


/*
Args:
  radii_s: a scaler for radii. only compute gradient if dx <= radii[0]*radii_s, dy <= radii[1]*radii_s
  Q < radii_s^2 * cutoff
 */
torch::Tensor RasterizePointsOccBackwardCpu(
    const torch::Tensor &points, // (P, 3)
    const torch::Tensor &radii,  // (P, 2)
    const torch::Tensor &grad_occ, //  (N, H, W)
    const torch::Tensor &cloud_to_packed_first_idx,
    const torch::Tensor &num_points_per_cloud,
    const float radii_s,
    const float depth_merging_thres)
{
  const int N = grad_occ.size(0);
  const int P = points.size(0);
  const int H = grad_occ.size(1);
  const int W = grad_occ.size(2);

  // For now only support square images.
  // TODO(jcjohns): Extend to non-square images.
  if (H != W)
  {
    AT_ERROR("RasterizePointsBackwardCpu only supports square images");
  }
  const int S = H;

  torch::Tensor grad_points = torch::zeros({P, 2}, points.options());

  // inputs
  auto points_a = points.accessor<float, 2>();
  auto grad_occ_a = grad_occ.accessor<float, 3>();
  auto radii_a = radii.accessor<float, 2>();
  // outputs
  auto grad_points_a = grad_points.accessor<float, 2>();

  /*
  This is gradient from the weights to the point position
  */
  for (int n = 0; n < N; ++n)
  {
    // Loop through each pointcloud in the batch.
    // Get the start index of the points in points_packed and the num points
    // in the point cloud.
    const int point_start_idx =
        cloud_to_packed_first_idx[n].item().to<int32_t>();
    const int point_stop_idx =
        (point_start_idx + num_points_per_cloud[n].item().to<int32_t>());

    for (int yi = 0; yi < S; ++yi)
    {
      // Reverse the order of yi so that +Y is pointing upwards in the image.
      const int yidx = S - 1 - yi;
      const float yf = PixToNdc(yidx, S);

      for (int xi = 0; xi < S; ++xi)
      {
        // Reverse the order of xi so that +X is pointing to the left in the
        // image.
        const int xidx = S - 1 - xi;
        const float xf = PixToNdc(xidx, S);
        // const float pixz = zbuf0_a[n][yi][xi];
        const float grad_occ_pix = grad_occ_a[n][yi][xi];

        // if occupancy_grad is 0, then we don't need to accomulate gradient for this pixel
        if (grad_occ_pix == 0.0f)
          continue;
        // still need to search all the points since we want to have larger gradient support
        for (int p = point_start_idx; p < point_stop_idx; ++p)
        {
          const float px = points_a[p][0];
          const float py = points_a[p][1];
          const float pz = points_a[p][2];

          if (pz < 0 || abs(py) > 1.0 || abs(px) > 1.0)
            continue;

          const float dx = xf - px;
          const float dy = yf - py;

          const float radiix = radii_a[p][0] * radii_s;
          const float radiiy = radii_a[p][1] * radii_s;

          // if grad_occ_pix > 0, it means this pixel shouldn't be occupied, but it's where the splat should
          // be (moving away from this pixel doesn't improve loss), so set the gradient to zero
          const bool pix_outside_splat = (abs(dx) > radiix / radii_s) || (abs(dy) > radiiy / radii_s);
          if (grad_occ_pix > 0.0f && pix_outside_splat)
            continue;

          if (abs(dx) > radiix && abs(dy) > radiiy)
            continue;

          const float dist2 = dx * dx + dy * dy;
          const float denom = std::max(dist2, 1e-8f);

          grad_points_a[p][0] += dx / denom * grad_occ_pix;
          grad_points_a[p][1] += dy / denom * grad_occ_pix;
        }
      }
    }
  }
  return grad_points;
}

void RasterizeZbufBackwardCpu(const at::Tensor &idx, const at::Tensor &grad_zbuf,
                           at::Tensor &point_z_grad)
{
  auto grad_zbuf_a = grad_zbuf.accessor<float, 4>();
  auto idxs_a = idx.accessor<int, 4>();
  auto point_z_grad_a = point_z_grad.accessor<float, 2>();

  const int N = grad_zbuf.size(0);
  const int H = grad_zbuf.size(1);
  const int W = grad_zbuf.size(2);
  const int K = grad_zbuf.size(3);

  for (int n = 0; n < N; ++n)
  {
    // Loop through each pointcloud in the batch.
    // Get the start index of the points in points_packed and the num points
    // in the point cloud.

    for (int yi = 0; yi < H; ++yi)
    {
      for (int xi = 0; xi < W; ++xi)
      {
        // pass the zbuf_grad same as WeightsBackward
        for (int k = 0; k < K; ++k)
        { // Loop over points for the pixel
          const float grad_pz = grad_zbuf_a[n][yi][xi][k];
          if (grad_pz == 0.0) continue;
          const int p = idxs_a[n][yi][xi][k];
          if (p < 0)
            break;
          point_z_grad_a[p][0] += grad_zbuf_a[n][yi][xi][k];
        }
      } // xi
    } // yi
  }
}