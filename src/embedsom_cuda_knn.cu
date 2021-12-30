/*
The MIT License

Copyright (c) 2021 Adam Smelko
                   Martin Krulis
                   Mirek Kratochvil

Permission is hereby granted, free of charge,
to any person obtaining a copy of this software and
associated documentation files (the "Software"), to
deal in the Software without restriction, including
without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom
the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice
shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include "embedsom_cuda.h"
#include <limits>

#include "bitonic.cuh"

#include "cooperative_groups.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/** Compute squared euclidean distance */
template<typename F = float>
__inline__ __device__ F
sqeucl(const F *__restrict__ lhs, const F *__restrict__ rhs, const uint32_t dim)
{
    F sum(0.0);
    for (uint32_t d = 0; d < dim; ++d) {
        F diff = *lhs++ - *rhs++;
        sum += diff * diff;
    }
    return sum;
}

/** Bubble a knn-entry up a list, insertsort-style */
template<typename F = float>
__inline__ __device__ void
bubbleUp(knn_entry<F> *const __restrict__ topK, uint32_t idx)
{
    while (idx > 0 && topK[idx - 1].distance > topK[idx].distance) {
        const knn_entry<F> tmp = topK[idx];
        topK[idx] = topK[idx - 1];
        topK[idx - 1] = tmp;
        --idx;
    }
}

/** Base kernel for kNN computation.
 *
 * Each thread iterates over whole point grid and computes kNN for a specified
 * point using insertion sort in global memory.
 */
template<typename F>
__global__ void
topkBaseKernel(const F *__restrict__ points,
               const F *const __restrict__ grid,
               knn_entry<F> *__restrict__ topKs,
               const uint32_t dim,
               const uint32_t n,
               const uint32_t gridSize,
               const uint32_t k)
{
    // assign correct point and topK pointers for a thread
    {
        const uint32_t pointIdx = blockIdx.x * blockDim.x + threadIdx.x;

        if (pointIdx >= n)
            return;

        topKs = topKs + pointIdx * k;
        points = points + pointIdx * dim;
    }

    // iterate over grid points
    {
        uint32_t gridIdx = 0;

        for (; gridIdx < k; ++gridIdx) {
            topKs[gridIdx] = { sqeucl<F>(points, grid + gridIdx * dim, dim),
                               gridIdx };
            bubbleUp<F>(topKs, gridIdx);
        }

        for (; gridIdx < gridSize; ++gridIdx) {
            F dist = sqeucl<F>(points, grid + gridIdx * dim, dim);

            if (topKs[k - 1].distance > dist) {
                topKs[k - 1] = { dist, gridIdx };
                bubbleUp<F>(topKs, k - 1);
            }
        }
    }
}

/** Bitonic kNN selection.
 *
 * This uses a bitonic top-k selection (modified bitonic sort) to run the kNN
 * search. No inputs are explicitly cached, shm is used for intermediate topk
 * results @tparam K is number of top-k selected items and number of threads
 * working cooperatively (keeping 2xK intermediate result in shm) note that K
 * is a power of 2 which is the nearest greater or equal value to actualK
 */
template<typename F, int K>
__global__ void
topkBitonicOptKernel(const F *__restrict__ points,
                     const F *const __restrict__ grid,
                     knn_entry<F> *__restrict__ topKs,
                     const uint32_t dim,
                     const uint32_t n,
                     const uint32_t gridSize,
                     const uint32_t actualK)
{
    extern __shared__ char sharedMemory[];

    knn_entry<F> *const shmTopk = (knn_entry<F> *)(sharedMemory);
    // topk chunk that belongs to this thread
    knn_entry<F> *const shmTopkBlock = shmTopk + K * (threadIdx.x / (K / 2));
    // every thread works on two items at a time
    knn_entry<F> *const shmNewData = shmTopk + (blockDim.x * 2);
    knn_entry<F> *const shmNewDataBlock =
      shmNewData +
      K * (threadIdx.x / (K / 2)); // newData chunk that belongs to this thread
    // assign correct point and topK pointers for a thread (K/2 threads
    // cooperate on each point)
    {
        const uint32_t pointIdx =
          (blockIdx.x * blockDim.x + threadIdx.x) / (K / 2);
        if (pointIdx >= n)
            return;
        points += pointIdx * dim;
        topKs += pointIdx * actualK;
    }
    // fill in initial topk intermediate result
    for (uint32_t i = threadIdx.x % (K / 2); i < K;
         i += (K / 2)) { // yes, this loop should go off exactly twice
        shmTopkBlock[i] = { i < gridSize
                              ? sqeucl<F>(points, grid + dim * i, dim)
                              : valueMax<F>,
                            i };
    }
    // process the grid points in K-sized blocks
    const uint32_t gridSizeRoundedToK = ((gridSize + K - 1) / K) * K;
    for (uint32_t gridOffset = K; gridOffset < gridSizeRoundedToK;
         gridOffset += K) {
        // compute another K new distances (again the loop runs just 2 times)
        for (uint32_t i = threadIdx.x % (K / 2); i < K; i += (K / 2)) {
            shmNewDataBlock[i] = {
                (gridOffset + i) < gridSize
                  ? sqeucl<F>(points, grid + dim * (gridOffset + i), dim)
                  : valueMax<F>,
                gridOffset + i
            };
        }

        // sync the whole block (bitonic update operates on the block)
        __syncthreads();
        // merge the new K distances with the intermediate top ones
        bitonic_topk_update_opt<knn_entry<F>, K / 2>(shmTopk, shmNewData);
        // sync the block again
        __syncthreads();
    }

    // sort the top K results once more, for the last time
    bitonic_sort<knn_entry<F>, K / 2>(shmTopk);
    __syncthreads();

    /* copy the results from shm to global memory (tricky part: some of the
     * results get ignored here, because K must be a power of 2, which may be
     * bigger than the `actualK` number of results that we want. */
    for (uint32_t i = threadIdx.x % (K / 2); i < actualK; i += (K / 2))
        topKs[i] = shmTopkBlock[i];
}

/** Start the bitonic kNN with the right parameters
 *
 * This function recursively scans (in compiler!) for a sufficient K to contain
 * the actual number of neighbors required and runs the correct instantiation
 * of topkBitonicOptKernel().
 */
template<typename F, int K = 2>
void
runnerWrapperBitonicOpt(const F *points,
                        const F *grid,
                        knn_entry<F> *topKs,
                        uint32_t dim,
                        uint32_t pointsCount,
                        uint32_t gridSize,
                        uint32_t actualK)
{
    /* TODO make a fallback for this case. Better run some operation a bit
     * slower, than nothing at all
     * TODO 2: report errors to frontend instead of just exploding */
    if constexpr (K > 256)
        throw std::runtime_error("Ooops, this should never happen. Bitonic "
                                 "kernel wrapper was invoked with k > 256.");
    else if (K < 2 || K < actualK)
        runnerWrapperBitonicOpt<F, K * 2>(
          points,
          grid,
          topKs,
          dim,
          pointsCount,
          gridSize,
          actualK); // recursion step (try next power of 2)
    else {
        // here we found K such that actualK <= K <= 256
        constexpr unsigned blockSize = 256;

        // some extra safety checks (should never fire)
        if constexpr (blockSize * 2 != (blockSize | (blockSize - 1)) + 1)
            throw std::runtime_error("CUDA block size must be a power of two "
                                     "for bitonic topk selection.");
        if constexpr (K / 2 > blockSize)
            throw std::runtime_error("CUDA block size must be at half of k "
                                     "(rounded u to nearest power of 2).");

        unsigned int blockCount =
          ((pointsCount * K / 2) + blockSize - 1) / blockSize;
        /* the kernel requires 4 items per thread
         * (2 in topk, and 2 for newly incoming data) */
        unsigned int shmSize = blockSize * 4 * sizeof(knn_entry<F>);

        topkBitonicOptKernel<F, K><<<blockCount, blockSize, shmSize>>>(
          points, grid, topKs, dim, pointsCount, gridSize, actualK);
    }
}

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
