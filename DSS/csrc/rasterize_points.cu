#include <ATen/ATen.h>
#include <ATen/TensorAccessor.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCNumerics.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <math.h>
#include <stdio.h>
#include <cstdio>
#include <sstream>
#include <tuple>
#include "bitmask.cuh"
#include "rasterization_utils.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace
{
    // A little structure for holding details about a pixel.
    struct Pix
    {
        float z;     // Depth of the reference point.
        int32_t idx; // Index of the reference point.
        // the value of the ellipse_function
        // Q=(a dx^2+b dxdy+c dy^2) of the points corresponding to idx map.
        // Note: EWA ~ exp(-0.5Q)
        float qvalue;
    };

    __device__ inline bool operator<(const Pix &a, const Pix &b)
    {
        return a.z < b.z;
    }

    // // The derivative of 1/(1+exp(-x))
    // __device__ inline float SigmoidDiff(const float x)
    // {
    //     if (abs(x) > 12)
    //         return 0.0f;
    //     return exp(-x) / ((1 + exp(-x)) * (1 + exp(-x)));
    // }

    // __device__ inline int sgn(float val)
    // {
    //     return (float(0) < val) - (val < float(0));
    // }

    // This function checks if a pixel given by xy location pxy lies within the
    // point with index p and batch index n. One of the inputs is a list (q)
    // which contains Pixel structs with the indices of the points which intersect
    // with this pixel sorted by closest z distance. If the pixel pxy lies in the
    // point, the list (q) is updated and re-orderered in place. In addition
    // the auxillary variables q_size, q_max_z and q_max_idx are also modified.
    // This code is shared between RasterizePointsNaiveCudaKernel and
    // RasterizePointsFineCudaKernel.

    // modification: instead of const distance threshold,
    // now we check if the q value is below the threshold
    // here we don't take depth_merging threshold into account,
    // we deal with it at the final step when update the final output
    // TODO: we might use a q_min_z to keep track of the frontmost pixel
    // and reject points whose distance in z are larger than depth_merging_threshold
    // problem: we might need to update the whole list when the new point is the frontmost pixel
    template <typename PointQ>
    __device__ void CheckPixelInsidePoint(
        const float *points,         // (P, 3)
        const float *radii,          // (P, 2)
        const float *ellipse_params, // (P, 3)
        const int p_idx,
        const float cutoff_thres,
        int &q_size,
        float &q_max_z,
        int &q_max_idx,
        PointQ &q,
        const float xf,
        const float yf,
        const int K)
    {
        const float px = points[p_idx * 3 + 0];
        const float py = points[p_idx * 3 + 1];
        const float pz = points[p_idx * 3 + 2];
        const float a = ellipse_params[p_idx * 3 + 0];
        const float b = ellipse_params[p_idx * 3 + 1];
        const float c = ellipse_params[p_idx * 3 + 2];
        const float radiix = radii[p_idx * 2 + 0];
        const float radiiy = radii[p_idx * 2 + 1];
        if (pz < 0)
            return; // Don't render points behind the camera
        const float dx = xf - px;
        const float dy = yf - py;
        // this function is used by both coarse-to-fine and naive version
        if (abs(dx) > radiix || abs(dy) > radiiy)
            return; // Skip if pixel out of precomputed radii range
        const float qvalue = a * dx * dx + b * dx * dy + c * dy * dy;
        // outside cutoff, do nothing
        if (qvalue > cutoff_thres)
            return;

        if (q_size < K)
        {
            // Just insert it
            q[q_size] = {pz, p_idx, qvalue};
            if (pz > q_max_z)
            {
                q_max_z = pz;
                q_max_idx = q_size;
            }
            q_size++;
        }
        else if (pz < q_max_z)
        {
            // Overwrite the old max, and find the new max
            q[q_max_idx] = {pz, p_idx, qvalue};
            q_max_z = pz;
            for (int i = 0; i < K; i++)
            {
                if (q[i].z > q_max_z)
                {
                    q_max_z = q[i].z;
                    q_max_idx = i;
                }
            }
        }
    }
} // namespace

// ****************************************************************************
// *                          NAIVE RASTERIZATION                             *
// ****************************************************************************

__global__ void RasterizePointsNaiveCudaKernel(
    const float *points,         // (P, 3)
    const float *ellipse_params, // (P, 3)
    const at::PackedTensorAccessor64<float, 1> cutoff_thres,
    const float *radii,                       // (P, 2)
    const int64_t *cloud_to_packed_first_idx, // (N)
    const int64_t *num_points_per_cloud,      // (N)
    const float depth_merging_thres,
    const int N,
    const int S,
    const int K,
    int32_t *point_idxs, // (N, S, S, K)
    float *zbuf,         // (N, S, S, K)
    float *qvalues,      // (N, S, S, K)
    float *occupancies)  // (N, S, S)
{

    // (N, S, S, K)
    // Simple version: One thread per output pixel
    const int num_threads = gridDim.x * blockDim.x;
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < N * S * S; i += num_threads)
    {
        // Convert linear index to 3D index
        const int n = i / (S * S); // Batch index
        const int pix_idx = i % (S * S);

        // Reverse ordering of the X and Y axis as the camera coordinates
        // assume that +Y is pointing up and +X is pointing left.
        const int yi = S - 1 - pix_idx / S;
        const int xi = S - 1 - pix_idx % S;

        const float xf = PixToNdc(xi, S);
        const float yf = PixToNdc(yi, S);

        // For keeping track of the K closest points we want a data structure
        // that (1) gives O(1) access to the closest point for easy comparisons,
        // and (2) allows insertion of new elements. In the CPU version we use
        // std::priority_queue; then (2) is O(log K). We can't use STL
        // containers in CUDA; we could roll our own max heap in an array, but
        // that would likely have a lot of warp divergence so we do something
        // simpler instead: keep the elements in an unsorted array, but keep
        // track of the max value and the index of the max value. Then (1) is
        // still O(1) time, while (2) is O(K) with a clean loop. Since K <= 8
        // this should be fast enough for our purposes.
        // TODO(jcjohns) Abstract this out into a standalone data structure
        Pix q[kMaxPointsPerPixel];
        int q_size = 0;
        float q_max_z = -1000;
        int q_max_idx = -1;

        // Using the batch index of the thread get the start and stop
        // indices for the points.
        const int64_t point_start_idx = cloud_to_packed_first_idx[n];
        const int64_t point_stop_idx = point_start_idx + num_points_per_cloud[n];

        for (int p_idx = point_start_idx; p_idx < point_stop_idx; ++p_idx)
        {
            CheckPixelInsidePoint(
                points, radii, ellipse_params, p_idx, cutoff_thres[p_idx], q_size, q_max_z, q_max_idx, q, xf, yf, K);
        }

        BubbleSort(q, q_size);

        int idx = n * S * S * K + pix_idx * K;
        if (q_max_z >= 0)
        {
            // there is a point located at this pixel
            occupancies[i] = 1;
        }
        for (int k = 0; k < q_size; ++k)
        {
            if (q[k].z - q[0].z > depth_merging_thres)
            {
                break;
            }
            point_idxs[idx + k] = q[k].idx;
            zbuf[idx + k] = q[k].z;
            qvalues[idx + k] = q[k].qvalue;
        }
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
RasterizePointsNaiveCuda(
    const at::Tensor &points,
    const at::Tensor &ellipse_params,
    const at::Tensor &cutoff, // (P,) or (1,)
    const at::Tensor &radii,
    const at::Tensor &cloud_to_packed_first_idx,
    const at::Tensor &num_points_per_cloud,
    const float depth_merging_thres,
    const int image_size,
    const int points_per_pixel)
{
    // Check inputs are on the same device
    at::TensorArg points_t{points, "points", 1},
        ellipse_params_t{ellipse_params, "ellipse_params", 2},
        cutoff_t{cutoff, "cutoff", 3},
        radii_t{radii, "radii", 4},
        cloud_to_packed_first_idx_t{
            cloud_to_packed_first_idx, "cloud_to_packed_first_idx", 5},
        num_points_per_cloud_t{num_points_per_cloud, "num_points_per_cloud", 6};
    at::CheckedFrom c = "RasterizePointsNaiveCuda";
    at::checkAllSameGPU(
        c, {points_t, ellipse_params_t, cutoff_t, radii_t, cloud_to_packed_first_idx_t, num_points_per_cloud_t});

    // Set the device for the kernel launch based on the device of the input
    at::cuda::CUDAGuard device_guard(points.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int N = num_points_per_cloud.size(0); // batch size.
    const int S = image_size;
    const int K = points_per_pixel;

    if (K > kMaxPointsPerPixel)
    {
        std::stringstream ss;
        ss << "Must have points_per_pixel <= " << kMaxPointsPerPixel;
        AT_ERROR(ss.str());
    }
    auto int_opts = points.options().dtype(at::kInt);
    auto float_opts = points.options().dtype(at::kFloat);
    at::Tensor point_idxs = at::full({N, S, S, K}, -1, int_opts);
    at::Tensor zbuf = at::full({N, S, S, K}, -1, float_opts);
    at::Tensor qvalues = at::full({N, S, S, K}, -1, float_opts);
    at::Tensor occupancies = at::full({N, S, S}, 0, float_opts);

    if (point_idxs.numel() == 0)
    {
        AT_CUDA_CHECK(cudaGetLastError());
        return std::make_tuple(point_idxs, zbuf, qvalues, occupancies);
    }

    const size_t blocks = 1024;
    const size_t threads = 64;
    RasterizePointsNaiveCudaKernel<<<blocks, threads, 0, stream>>>(
        points.contiguous().data_ptr<float>(),
        ellipse_params.contiguous().data_ptr<float>(),
        cutoff.packed_accessor64<float, 1>(),
        radii.contiguous().data_ptr<float>(),
        cloud_to_packed_first_idx.contiguous().data_ptr<int64_t>(),
        num_points_per_cloud.contiguous().data_ptr<int64_t>(),
        depth_merging_thres,
        N,
        S,
        K,
        point_idxs.contiguous().data_ptr<int32_t>(),
        zbuf.contiguous().data_ptr<float>(),
        qvalues.contiguous().data_ptr<float>(),
        occupancies.contiguous().data_ptr<float>());

    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(point_idxs, zbuf, qvalues, occupancies);
}

// ****************************************************************************
// *                          COARSE RASTERIZATION                            *
// ****************************************************************************

// modification: only change the px0/1 and py0/1 here:
// now the bounding box might be a rectangle instead of square
__global__ void RasterizePointsCoarseCudaKernel(
    const float *points,                      // (P, 3)
    const float *radii,                       // (P, 2), precomputed axis-aligned bbox
    const int64_t *cloud_to_packed_first_idx, // (N)
    const int64_t *num_points_per_cloud,      // (N)
    const int N,
    const int P,
    const int S,
    const int bin_size,
    const int chunk_size,
    const int max_points_per_bin,
    int *points_per_bin,
    int *bin_points)
{
    extern __shared__ char sbuf[];
    const int M = max_points_per_bin;
    const int num_bins = 1 + (S - 1) / bin_size; // Integer divide round up
    const float half_pix = 1.0f / S;             // Size of half a pixel in NDC units

    // This is a boolean array of shape (num_bins, num_bins, chunk_size)
    // stored in shared memory that will track whether each point in the chunk
    // falls into each bin of the image.
    BitMask binmask((unsigned int *)sbuf, num_bins, num_bins, chunk_size);

    // Have each block handle a chunk of points and build a 3D bitmask in
    // shared memory to mark which points hit which bins.  In this first phase,
    // each thread processes one point at a time. After processing the chunk,
    // one thread is assigned per bin, and the thread counts and writes the
    // points for the bin out to global memory.
    const int chunks_per_batch = 1 + (P - 1) / chunk_size;
    const int num_chunks = N * chunks_per_batch;
    for (int chunk = blockIdx.x; chunk < num_chunks; chunk += gridDim.x)
    {
        const int batch_idx = chunk / chunks_per_batch;
        const int chunk_idx = chunk % chunks_per_batch;
        const int point_start_idx = chunk_idx * chunk_size;

        binmask.block_clear();

        // Using the batch index of the thread get the start and stop
        // indices for the points.
        const int64_t cloud_point_start_idx = cloud_to_packed_first_idx[batch_idx];
        const int64_t cloud_point_stop_idx =
            cloud_point_start_idx + num_points_per_cloud[batch_idx];

        // Have each thread handle a different point within the chunk
        for (int p = threadIdx.x; p < chunk_size; p += blockDim.x)
        {
            const int p_idx = point_start_idx + p;

            // Check if point index corresponds to the cloud in the batch given by
            // batch_idx.
            if (p_idx >= cloud_point_stop_idx || p_idx < cloud_point_start_idx)
            {
                continue;
            }

            const float px = points[p_idx * 3 + 0];
            const float py = points[p_idx * 3 + 1];
            const float pz = points[p_idx * 3 + 2];
            if (pz < 0)
                continue; // Don't render points behind the camera.
            const float px0 = px - radii[p_idx * 2 + 0];
            const float px1 = px + radii[p_idx * 2 + 0];
            const float py0 = py - radii[p_idx * 2 + 1];
            const float py1 = py + radii[p_idx * 2 + 1];

            // Brute-force search over all bins; TODO something smarter?
            // For example we could compute the exact bin where the point falls,
            // then check neighboring bins. This way we wouldn't have to check
            // all bins (however then we might have more warp divergence?)
            for (int by = 0; by < num_bins; ++by)
            {
                // Get y extent for the bin. PixToNdc gives us the location of
                // the center of each pixel, so we need to add/subtract a half
                // pixel to get the true extent of the bin.
                const float by0 = PixToNdc(by * bin_size, S) - half_pix;
                const float by1 = PixToNdc((by + 1) * bin_size - 1, S) + half_pix;
                const bool y_overlap = ((py0 <= by1) && (by0 <= py1));

                if (!y_overlap)
                {
                    continue;
                }
                for (int bx = 0; bx < num_bins; ++bx)
                {
                    // Get x extent for the bin; again we need to adjust the
                    // output of PixToNdc by half a pixel.
                    const float bx0 = PixToNdc(bx * bin_size, S) - half_pix;
                    const float bx1 = PixToNdc((bx + 1) * bin_size - 1, S) + half_pix;
                    const bool x_overlap = (px0 <= bx1) && (bx0 <= px1);

                    if (x_overlap)
                    {
                        binmask.set(by, bx, p);
                    }
                }
            }
        }
        __syncthreads();
        // Now we have processed every point in the current chunk. We need to
        // count the number of points in each bin so we can write the indices
        // out to global memory. We have each thread handle a different bin.
        for (int byx = threadIdx.x; byx < num_bins * num_bins; byx += blockDim.x)
        {
            const int by = byx / num_bins;
            const int bx = byx % num_bins;
            const int count = binmask.count(by, bx);
            const int points_per_bin_idx =
                batch_idx * num_bins * num_bins + by * num_bins + bx;

            // This atomically increments the (global) number of points found
            // in the current bin, and gets the previous value of the counter;
            // this effectively allocates space in the bin_points array for the
            // points in the current chunk that fall into this bin.
            // bug here: the add is atmoic but not together with the read
            // when read the value of start, the value in points_per_bin might already be changed
            // gpuAtomicAdd(points_per_bin + points_per_bin_idx, count);
            // const int start = points_per_bin[points_per_bin_idx];
            const int start = atomicAdd(points_per_bin + points_per_bin_idx, count);

            // Now loop over the binmask and write the active bits for this bin
            // out to bin_points.
            int next_idx = batch_idx * num_bins * num_bins * M + by * num_bins * M +
                           bx * M + start;
            for (int p = 0; p < chunk_size; ++p)
            {
                if (binmask.get(by, bx, p))
                {
                    // TODO: Throw an error if next_idx >= M -- this means that
                    // we got more than max_points_per_bin in this bin
                    // TODO: check if gpuAtomicAdd is needed in line 265.
                    bin_points[next_idx] = point_start_idx + p;
                    next_idx++;
                }
            }
        }
        __syncthreads();
    }
}

at::Tensor RasterizePointsCoarseCuda(
    const at::Tensor &points,                    // (P, 3)
    const at::Tensor &radii,                     // (P, 2)
    const at::Tensor &cloud_to_packed_first_idx, // (N)
    const at::Tensor &num_points_per_cloud,      // (N)
    const int image_size,
    const int bin_size,
    const int max_points_per_bin)
{
    // Check inputs are on the same device
    at::TensorArg points_t{points, "points", 1},
        radii_t{radii, "radii", 2},
        cloud_to_packed_first_idx_t{
            cloud_to_packed_first_idx, "cloud_to_packed_first_idx", 3},
        num_points_per_cloud_t{num_points_per_cloud, "num_points_per_cloud", 4};
    at::CheckedFrom c = "RasterizePointsCoarseCuda";
    at::checkAllSameGPU(
        c, {points_t, radii_t, cloud_to_packed_first_idx_t, num_points_per_cloud_t});

    // Set the device for the kernel launch based on the device of the input
    at::cuda::CUDAGuard device_guard(points.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int P = points.size(0);
    const int N = num_points_per_cloud.size(0);
    const int num_bins = 1 + (image_size - 1) / bin_size; // divide round up
    const int M = max_points_per_bin;

    if (num_bins >= 22)
    {
        // Make sure we do not use too much shared memory.
        std::stringstream ss;
        ss << "Got " << num_bins << "; that's too many!";
        AT_ERROR(ss.str());
    }
    auto opts = points.options().dtype(at::kInt);
    at::Tensor points_per_bin = at::zeros({N, num_bins, num_bins}, opts);
    at::Tensor bin_points = at::full({N, num_bins, num_bins, M}, -1, opts);

    if (bin_points.numel() == 0)
    {
        AT_CUDA_CHECK(cudaGetLastError());
        return bin_points;
    }

    const int chunk_size = 512;
    const size_t shared_size = num_bins * num_bins * chunk_size / 8;
    const size_t blocks = 64;
    const size_t threads = 512;

    RasterizePointsCoarseCudaKernel<<<blocks, threads, shared_size, stream>>>(
        points.contiguous().data_ptr<float>(),
        radii.contiguous().data_ptr<float>(),
        cloud_to_packed_first_idx.contiguous().data_ptr<int64_t>(),
        num_points_per_cloud.contiguous().data_ptr<int64_t>(),
        N,
        P,
        image_size,
        bin_size,
        chunk_size,
        M,
        points_per_bin.contiguous().data_ptr<int32_t>(),
        bin_points.contiguous().data_ptr<int32_t>());

    AT_CUDA_CHECK(cudaGetLastError());
    return bin_points;
}

// ****************************************************************************
// *                            FINE RASTERIZATION                            *
// ****************************************************************************

__global__ void RasterizePointsFineCudaKernel(
    const float *points,         // (P, 3)
    const float *ellipse_params, // (P, 3), the parabolic parameters
    const float *cutoff_thres,   // (P,)
    const float *radii,          // (P, 2)
    const int32_t *bin_points,   // (N, B, B, T)
    const float depth_merging_thres,
    const int bin_size,
    const int N,
    const int B, // num_bins
    const int M,
    const int S,
    const int K,
    int32_t *point_idxs,  // (N, S, S, K)
    float *zbuf,          // (N, S, S, K)
    float *qvalue_map,    // (N, S, S, K)
    float *occupancy_map) // (N, S, S)
{
    // This can be more than S^2 if S is not dividable by bin_size.
    const int num_pixels = N * B * B * bin_size * bin_size;
    const int num_threads = gridDim.x * blockDim.x;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int pid = tid; pid < num_pixels; pid += num_threads)
    {
        // Convert linear index into bin and pixel indices. We make the within
        // block pixel ids move the fastest, so that adjacent threads will fall
        // into the same bin; this should give them coalesced memory reads when
        // they read from points and bin_points.
        int i = pid;
        const int n = i / (B * B * bin_size * bin_size); // batch
        i %= B * B * bin_size * bin_size;                // index of the batch
        const int by = i / (B * bin_size * bin_size);    // bin_y
        i %= B * bin_size * bin_size;
        const int bx = i / (bin_size * bin_size); // bin_x
        i %= bin_size * bin_size;                 // index inside the bin

        const int yi = i / bin_size + by * bin_size;
        const int xi = i % bin_size + bx * bin_size;

        if (yi >= S || xi >= S)
            continue;

        const float xf = PixToNdc(xi, S);
        const float yf = PixToNdc(yi, S);

        // This part looks like the naive rasterization kernel, except we use
        // bin_points to only look at a subset of points already known to fall
        // in this bin. TODO abstract out this logic into some data structure
        // that is shared by both kernels?
        Pix q[kMaxPointsPerPixel];
        int q_size = 0;
        float q_max_z = -1000;
        int q_max_idx = -1;
        for (int m = 0; m < M; ++m)
        {
            const int p = bin_points[n * B * B * M + by * B * M + bx * M + m];
            if (p < 0)
            {
                // bin_points uses -1 as a sentinal value
                continue;
            }
            CheckPixelInsidePoint(
                points, radii, ellipse_params, p, cutoff_thres[p], q_size, q_max_z, q_max_idx, q, xf, yf, K);
        }
        // Now we've looked at all the points for this bin, so we can write
        // output for the current pixel.
        BubbleSort(q, q_size);

        // Reverse ordering of the X and Y axis as the camera coordinates
        // assume that +Y is pointing up and +X is pointing left.
        const int yidx = S - 1 - yi;
        const int xidx = S - 1 - xi;

        const int pix_idx = n * S * S * K + yidx * S * K + xidx * K;
        if (q_max_z > 0)
        {
            // there is a point located at this pixel
            occupancy_map[n * S * S + yidx * S + xidx] = 1;
        }
        for (int k = 0; k < q_size; ++k)
        {
            if (q[k].z - q[0].z > depth_merging_thres)
            {
                break;
            }
            point_idxs[pix_idx + k] = q[k].idx;
            zbuf[pix_idx + k] = q[k].z;
            qvalue_map[pix_idx + k] = q[k].qvalue;
        }
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> RasterizePointsFineCuda(
    const at::Tensor &points,         // (P, 3)
    const at::Tensor &ellipse_params, // (P, 3)
    const at::Tensor &cutoff,         // (P,) or (1,)
    const at::Tensor &radii,          // (P, 2)
    const at::Tensor &bin_points,     // (N, B, B, M)
    const float depth_merging_thres,
    const int image_size,
    const int bin_size,
    const int points_per_pixel)
{
    // Check inputs are on the same device
    at::TensorArg points_t{points, "points", 1},
        ellipse_params_t{ellipse_params, "ellipse_params", 2},
        cutoff_t{cutoff, "cutoff", 3},
        radii_t{radii, "radii", 4},
        bin_points_t{bin_points, "bin_points", 4};
    at::CheckedFrom c = "RasterizePointsFineCuda";
    at::checkAllSameGPU(c, {points_t, radii_t, ellipse_params_t, cutoff_t, bin_points_t});

    // Set the device for the kernel launch based on the device of the input
    at::cuda::CUDAGuard device_guard(points.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int N = bin_points.size(0);
    const int B = bin_points.size(1); // num_bins
    const int M = bin_points.size(3);
    const int S = image_size;
    const int K = points_per_pixel;
    if (K > kMaxPointsPerPixel)
    {
        AT_ERROR("Must have num_closest <= 150");
    }
    auto int_opts = bin_points.options().dtype(at::kInt);
    auto float_opts = points.options().dtype(at::kFloat);
    at::Tensor point_idxs = at::full({N, S, S, K}, -1, int_opts);
    at::Tensor zbuf = at::full({N, S, S, K}, -1, float_opts);
    at::Tensor qvalue_map = at::full({N, S, S, K}, -1, float_opts);
    at::Tensor occupancy_map = at::zeros({N, S, S}, float_opts);

    if (point_idxs.numel() == 0)
    {
        AT_CUDA_CHECK(cudaGetLastError());
        return std::make_tuple(point_idxs, zbuf, qvalue_map, occupancy_map);
    }

    const size_t blocks = 1024;
    const size_t threads = 64;
    RasterizePointsFineCudaKernel<<<blocks, threads, 0, stream>>>(
        points.contiguous().data_ptr<float>(),
        ellipse_params.contiguous().data_ptr<float>(),
        cutoff.contiguous().data_ptr<float>(),
        radii.contiguous().data_ptr<float>(),
        bin_points.contiguous().data_ptr<int32_t>(),
        depth_merging_thres,
        bin_size,
        N,
        B,
        M,
        S,
        K,
        point_idxs.contiguous().data_ptr<int32_t>(),
        zbuf.contiguous().data_ptr<float>(),
        qvalue_map.contiguous().data_ptr<float>(),
        occupancy_map.contiguous().data_ptr<float>());

    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(point_idxs, zbuf, qvalue_map, occupancy_map);
}

// ****************************************************************************
// *                            BACKWARD PASS                                 *
// ****************************************************************************

__global__ void RasterizePointsOccBackwardCudaKernel(
    const float *points,                      // (P, 3)
    const float *radii,                       // (P, 2)
    const int64_t *cloud_to_packed_first_idx, // (N)
    const int64_t *num_points_per_cloud,      // (N)
    const float radii_s,
    const int N,
    const int H, const int W,
    const float *grad_occ, // (N,H,W)
    float *grad_points     //  (P,2)
)
{
    // (N, S, S, K)
    // Simple version: One thread per output pixel
    const int num_threads = gridDim.x * blockDim.x;
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const uint num_pixels = H * W * N;
    for (int i = tid; i < num_pixels; i += num_threads)
    {
        // Convert linear index to 3D index
        const int n = i / (H * W); // Batch index
        const int pix_idx = i % (H * W);

        // Reverse ordering of the X and Y axis as the camera coordinates
        // assume that +Y is pointing up and +X is pointing left.
        const int yi = H - 1 - pix_idx / W;
        const int xi = W - 1 - pix_idx % W;

        const float grad_occ_pix = grad_occ[i];
        const float xf = PixToNdc(xi, W);
        const float yf = PixToNdc(yi, H);

        if (grad_occ_pix != 0.0f)
        {
            // Using the batch index of the thread get the start and stop
            // indices for the points.
            const int64_t point_start_idx = cloud_to_packed_first_idx[n];
            const int64_t point_stop_idx = point_start_idx + num_points_per_cloud[n];

            for (int p_idx = point_start_idx; p_idx < point_stop_idx; ++p_idx)
            {
                const float px = points[p_idx * 3 + 0];
                const float py = points[p_idx * 3 + 1];
                const float pz = points[p_idx * 3 + 2];
                // const float cutoff_thres_scaled = cutoff_thres[p_idx] * radii_s * radii_s;

                // outside renderable area
                if (pz < 0 || abs(py) > 1.0 || abs(px) > 1.0)
                    continue;

                const float dx = xf - px;
                const float dy = yf - py;

                const float radiix = radii[p_idx * 2 + 0] * radii_s;
                const float radiiy = radii[p_idx * 2 + 1] * radii_s;

                if (abs(dx) > radiix || abs(dy) > radiiy)
                    continue; // Skip if pixel out of precomputed radii range

                // const float qvalue_pix = a * dx * dx + b * dx * dy + c * dy * dy;

                // // gradient for cutoff threshold
                // const float cutoff_value = cutoff_thres[p_idx];
                // float qdiff = cutoff_value - qvalue_pix;
                // const float cutoff_grad_pix = 1e-4 * sgn(grad_occ_pix) * SigmoidDiff(-qdiff);
                // gpuAtomicAdd(grad_cutoff + p_idx, cutoff_grad_pix);

                // pixel outside the splat and is rendered yet it shouldn't be
                // such pixels don't contribute to point gradient
                // const bool pix_outside_splat = qvalue_pix > cutoff_value;
                const bool pix_outside_splat = (abs(dx) > radiix / radii_s) || (abs(dy) > radiiy / radii_s);
                if (grad_occ_pix > 0.0f && pix_outside_splat)
                    // // if (grad_occ_pix > 0.0f)
                    continue;

                // if (qvalue_pix > cutoff_thres_scaled)
                //     continue;
                // point is not on the same surface
                const float dist2 = dx * dx + dy * dy;
                const float denom = eps_denom(dist2, 1e-10f);
                const float grad_px = dx / denom * grad_occ_pix;
                const float grad_py = dy / denom * grad_occ_pix;
                gpuAtomicAdd(grad_points + p_idx * 2 + 0, grad_px);
                gpuAtomicAdd(grad_points + p_idx * 2 + 1, grad_py);
            }
        }
    }
}

at::Tensor RasterizePointsOccBackwardCuda(
    const at::Tensor &points,   // (P, 3)
    const at::Tensor &radii,    // (P, 2)
    const at::Tensor &grad_occ, //  (N, H, W)
    const at::Tensor &cloud_to_packed_first_idx,
    const at::Tensor &num_points_per_cloud,
    const float radii_s,
    const float depth_merging_thres)
{
    // Check inputs are on the same device, size checks are done in rasterize_points.h
    at::TensorArg points_t{points, "points", 1},
        radii_t{radii, "radii", 2},
        grad_occ_t{grad_occ, "grad_occ", 3},
        cloud_to_packed_first_idx_t{
            cloud_to_packed_first_idx, "cloud_to_packed_first_idx", 4},
        num_points_per_cloud_t{num_points_per_cloud, "num_points_per_cloud", 5};
    at::CheckedFrom c = "RasterizePointsOccBackwardCuda";
    at::checkAllSameGPU(
        c, {points_t, radii_t, grad_occ_t,
            cloud_to_packed_first_idx_t, num_points_per_cloud_t});

    // Set the device for the kernel launch based on the device of the input
    at::cuda::CUDAGuard device_guard(points.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int P = points.size(0);
    const int N = grad_occ.size(0);
    const int H = grad_occ.size(1);
    const int W = grad_occ.size(2);

    // TODO: fetch the new pytorch3d to allow unequal H and W
    const int S = H;
    at::Tensor grad_points = at::zeros({P, 2}, points.options());

    if (grad_points.numel() == 0)
    {
        AT_CUDA_CHECK(cudaGetLastError());
        return grad_points;
    }

    const size_t blocks = 1024;
    const size_t threads = 64;
    // initialize the look-up table x/(x^2+y^2), x\in [0, radii_s*radii]
    // linspace for x\in[0, radii_s*radii]
    // l_sq x^2 + x^2

    RasterizePointsOccBackwardCudaKernel<<<blocks, threads, 0, stream>>>(
        points.contiguous().data_ptr<float>(),
        radii.contiguous().data_ptr<float>(),
        cloud_to_packed_first_idx.contiguous().data_ptr<int64_t>(),
        num_points_per_cloud.contiguous().data_ptr<int64_t>(),
        radii_s,
        N,
        H, W,
        grad_occ.contiguous().data_ptr<float>(),
        grad_points.contiguous().data_ptr<float>());

    AT_CUDA_CHECK(cudaGetLastError());
    return grad_points;
}

__global__ void ZbufBackwardKernel(const int *idxs, const float *grad_zbuf,
                                   const int N, const int H, const int W, const int K,
                                   float *z_grad)
{
    // (N, H, W, K)
    // Simple version: One thread per output pixel
    const int num_threads = gridDim.x * blockDim.x;
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int num_pixels = N * H * W;
    for (int i = tid; i < num_pixels; i += num_threads)
    {
        for (uint k = 0; k < K; k++)
        {
            const uint ik = K*i;
            const float grad_pz = grad_zbuf[ik + k];
            if (grad_pz == 0.0) continue;
            // Accumulate grad_zbuf
            const int p_idx = idxs[ik + k];
            if (p_idx < 0)
                break;
            gpuAtomicAdd(z_grad + p_idx, grad_pz);
        }
    }
}

void RasterizeZbufBackwardCuda(const at::Tensor& idx, const at::Tensor& zbuf_grad, at::Tensor& point_z_grad)
{
    // Check inputs are on the same device, size checks are done in rasterize_points.h
    at::TensorArg idx_t{idx, "idx", 1},
        zbuf_grad_t{zbuf_grad, "zbuf_grad", 2};
    at::CheckedFrom c = "RasterizeZbufBackwardCuda";
    at::checkAllSameGPU(
        c, {idx_t, zbuf_grad_t});

    // Set the device for the kernel launch based on the device of the input
    at::cuda::CUDAGuard device_guard(idx.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int N = idx.size(0);
    const int H = idx.size(1);
    const int W = idx.size(2);
    const int K = idx.size(3);
    const int P = point_z_grad.size(0);

    if (point_z_grad.numel() == 0)
    {
        AT_CUDA_CHECK(cudaGetLastError());
    }

    const size_t blocks = 1024;
    const size_t threads = 64;

    ZbufBackwardKernel<<<blocks, threads, 0, stream>>>(
        idx.contiguous().data_ptr<int>(),
        zbuf_grad.contiguous().data_ptr<float>(),
        N,
        H, W,
        K,
        point_z_grad.contiguous().data_ptr<float>());

    AT_CUDA_CHECK(cudaGetLastError());
    return;
}