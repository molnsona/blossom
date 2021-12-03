#include <limits>
#include "embedsom_cuda.h"

#include "bitonic.cuh"

#include "cooperative_groups.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/*
 * Kernels
 */

template<typename F = float>
__inline__ __device__ F
distance(const F *__restrict__ lhs,
         const F *__restrict__ rhs,
         const std::uint32_t dim)
{
    F sum(0.0);
    for (std::uint32_t d = 0; d < dim; ++d) {
        F diff = *lhs++ - *rhs++;
        sum += diff * diff;
    }
    return sum; // squared euclidean
}

template<typename F = float>
__inline__ __device__ void
bubbleUp(knn_entry<F> *const __restrict__ topK, std::uint32_t idx)
{
    while (idx > 0 && topK[idx - 1].distance > topK[idx].distance) {
        const knn_entry<F> tmp = topK[idx];
        topK[idx] = topK[idx - 1];
        topK[idx - 1] = tmp;
        --idx;
    }
}

/**
 * Each thread iterates over whole point grid and computes kNN for a specified
 * point using insertion sort in global memory.
 */
template<typename F>
__global__ void
topkBaseKernel(const F *__restrict__ points,
               const F *const __restrict__ grid,
               knn_entry<F> *__restrict__ topKs,
               const std::uint32_t dim,
               const std::uint32_t n,
               const std::uint32_t gridSize,
               const std::uint32_t k)
{
    // assign correct point and topK pointers for a thread
    {
        const std::uint32_t pointIdx = blockIdx.x * blockDim.x + threadIdx.x;

        if (pointIdx >= n)
            return;

        topKs = topKs + pointIdx * k;
        points = points + pointIdx * dim;
    }

    // iterate over grid points
    {
        std::uint32_t gridIdx = 0;

        for (; gridIdx < k; ++gridIdx) {
            topKs[gridIdx] = { distance<F>(points, grid + gridIdx * dim, dim),
                               gridIdx };
            bubbleUp<F>(topKs, gridIdx);
        }

        for (; gridIdx < gridSize; ++gridIdx) {
            F dist = distance<F>(points, grid + gridIdx * dim, dim);

            if (topKs[k - 1].distance > dist) {
                topKs[k - 1] = { dist, gridIdx };
                bubbleUp<F>(topKs, k - 1);
            }
        }
    }
}

/**
 * Uses bitonic top-k selection (modified bitonic sort). No inputs are
 * explicitly cached, shm is used for intermediate topk results
 * @tparam K is number of top-k selected items and number of threads working
 * cooperatively (keeping 2xK intermediate result in shm) note that K is a power
 * of 2 which is the nearest greater or equal value to actualK
 */
template<typename F, int K>
__global__ void
topkBitonicOptKernel(const F *__restrict__ points,
                     const F *const __restrict__ grid,
                     knn_entry<F> *__restrict__ topKs,
                     const std::uint32_t dim,
                     const std::uint32_t n,
                     const std::uint32_t gridSize,
                     const std::uint32_t actualK)
{
    extern __shared__ char sharedMemory[];
    knn_entry<F> *const shmTopk = (knn_entry<F> *)(sharedMemory);
    knn_entry<F> *const shmTopkBlock =
      shmTopk +
      K * (threadIdx.x / (K / 2)); // topk chunk that belongs to this thread
    knn_entry<F> *const shmNewData =
      shmTopk + (blockDim.x * 2); // every thread works on two items at a time
    knn_entry<F> *const shmNewDataBlock =
      shmNewData +
      K * (threadIdx.x / (K / 2)); // newData chunk that belongs to this thread
    // assign correct point and topK pointers for a thread (K/2 threads
    // cooperate on each point)
    {
        const std::uint32_t pointIdx =
          (blockIdx.x * blockDim.x + threadIdx.x) / (K / 2);
        if (pointIdx >= n)
            return;
        points += pointIdx * dim;
        topKs += pointIdx * actualK;
    }
    // fill in initial topk intermediate result
    for (std::uint32_t i = threadIdx.x % (K / 2); i < K;
         i += (K / 2)) { // yes, this loop should go off exactly twice
        shmTopkBlock[i] = { distance<F>(points, grid + dim * i, dim), i };
    }
    // process the grid points in K-sized blocks
    const gridSizeRoundedToK = ((gridSize + K - 1) / K) * K;
    for (std::uint32_t gridOffset = K; gridOffset < gridSizeRoundedToK; gridOffset += K) {
        // compute another K new distances
        for (std::uint32_t i = threadIdx.x % (K / 2); i < K;
             i += (K / 2)) { // yes, this loop should go off exactly twice
            shmNewDataBlock[i] = (gridOFfset < gridSize) ? ({
                distance<F>(points, grid + dim * (gridOffset + i), dim),
                gridOffset + i
            }) : ({
                std::numeric_limist<F>::max(),
                gridOffset + i
            });
        }
        __syncthreads(); // actually, whole block should be synced as the
                         // bitonic update operates on the whole block
        // merge them with intermediate topk
        bitonic_topk_update_opt<knn_entry<F>, K / 2>(shmTopk, shmNewData);
        __syncthreads();
    }
    // final sorting of the topk result
    bitonic_sort<knn_entry<F>, K / 2>(shmTopk);
    __syncthreads();
    // copy topk results from shm to global memory
    for (std::uint32_t i = threadIdx.x % (K / 2); i < actualK;
         i += (K / 2)) { // note there is actual K as limit (which might not be
                         // power of 2)
        topKs[i] = shmTopkBlock[i];
    }
}

template<typename F, int K = 2>
void
runnerWrapperBitonicOpt(const F *points,
                        const F *grid,
                        knn_entry<F> *topKs,
                        std::uint32_t dim,
                        std::uint32_t pointsCount,
                        std::uint32_t gridSize,
                        std::uint32_t actualK)
{
    if constexpr (K > 256) {
        // TODO make a fallback (better run something slowly, than nothing at
        // all)
        throw std::runtime_error("Ooops, this should never happen. Bitonic "
                                 "kernel wrapper was invoked with k > 256.");
    } else if (K < 2 || K < actualK) {
        // still looking for the right K...
        runnerWrapperBitonicOpt<F, K * 2>(
          points, grid, topKs, dim, pointsCount, gridSize, actualK);
    } else {
        unsigned int blockSize = 256;

        // TODO convert to static assert
        // we found the right nearest power of two using template
        // meta-programming
        if (blockSize * 2 != (blockSize | (blockSize - 1)) + 1) {
            throw std::runtime_error("CUDA block size must be a power of two "
                                     "for bitonic topk selection.");
        }
        if (K / 2 > blockSize) {
            throw std::runtime_error("CUDA block size must be at half of k "
                                     "(rounded u to nearest power of 2).");
        }

        unsigned int blockCount =
          ((pointsCount * K / 2) + blockSize - 1) / blockSize;
        unsigned int shmSize =
          blockSize * 4 *
          sizeof(knn_entry<F>); // 2 items per thread in topk and 2 more
                                // in tmp for new data
        topkBitonicOptKernel<F, K><<<blockCount, blockSize, shmSize>>>(
          points, grid, topKs, dim, pointsCount, gridSize, actualK);
    }
}

// runner wrapped in a class
void
EmbedSOMCUDAContext::runKNNKernel(size_t d,
                                  size_t n,
                                  size_t g,
                                  size_t adjusted_k)
{
    if (adjusted_k > 256) {
        /* If k is too high, fallback to the base kernel. Fortunately no one
         * ever would like to use such a high k, right? */
        constexpr unsigned blockSize = 256;
        unsigned blockCount = (n + blockSize - 1) / blockSize;
        topkBaseKernel<float><<<blockCount, blockSize>>>(data,
                                                         lm_hi,
                                                         knns,
                                                         uint32_t(d),
                                                         uint32_t(n),
                                                         uint32_t(g),
                                                         uint32_t(adjusted_k));
    } else
        runnerWrapperBitonicOpt<float, 2>(data,
                                          lm_hi,
                                          knns,
                                          uint32_t(d),
                                          uint32_t(n),
                                          uint32_t(g),
                                          uint32_t(adjusted_k));
}
