/*
The MIT License

Copyright (c) 2021 Martin Krulis
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

#ifndef ESOM_CUDA_BITONIC_CUH
#define ESOM_CUDA_BITONIC_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/** Wrapped compare and swap-to-correct-order function
 *
 * This is supposed to be used as a policy template parameter*/
template<typename T>
struct OrderWithCmpSwap
{
    static __device__ void order(T &a, T &b)
    {
        if (a <= b)
            return;
        T tmp = a;
        a = b;
        b = tmp;
    }
};

/** Min-max-into-ordering function
 *
 * This is supposed to be used as a policy template parameter*/
template<typename T>
struct OrderWithMinMax
{
    static __device__ void order(T &a, T &b)
    {
        T A = a;
        T B = b;
        a = min(A, B);
        b = max(A, B);
    }
};

/** Comparator policy that automatically chooses a good implementation */
template<typename T>
struct ComparatorPolicy : OrderWithCmpSwap<T>
{};

template<>
struct ComparatorPolicy<float> : OrderWithMinMax<float>
{};

/** A single "layer" of the parallel bitonic comparator. */
template<typename T,
         int BLOCK_SIZE,
         class CMP = ComparatorPolicy<T>,
         bool UP = false>
__device__ void
bitonic_merge_step(T *__restrict__ data)
{
    if constexpr (BLOCK_SIZE < 1)
        return; // this should never happen, but just to be safe

    /* masking lower bits from thread idx (like modulo STEP) */
    unsigned localMask = (unsigned)BLOCK_SIZE - 1;
    /* masking remaining bits from thread idx */
    unsigned blockMask = ~localMask;
    unsigned localIdx = (unsigned)threadIdx.x & localMask;
    /* offset of the block has 2x size of threads in local block
     * (each thread works on 2 values at once) */
    unsigned blockOffset = ((unsigned)threadIdx.x & blockMask) * 2;

    /* UP is the first phase of the merge, which is different (compares first
     * with last, second with last second ...).
     *
     * Remaining steps use DOWN (ie UP == false) which is regular pairwise
     * comparison within STEP-size sub-block. */
    unsigned secondIdx =
      UP ? blockOffset + ((unsigned)BLOCK_SIZE * 2 - 1) - localIdx
         : blockOffset + localIdx + (unsigned)BLOCK_SIZE;
    CMP::order(data[blockOffset + localIdx], data[secondIdx]);

    if constexpr (BLOCK_SIZE > 32)
        __syncthreads();
    else
        __syncwarp();
}

/** Parallel bitonic merge.
 *
 * This runs several layers of the parallel bitonic comparators. */
template<typename T,
         int BLOCK_SIZE,
         class CMP = ComparatorPolicy<T>,
         bool UP = true>
__device__ void
bitonic_merge(T *__restrict__ data)
{
    if constexpr (BLOCK_SIZE < 1)
        return; // recursion bottom case

    /* do one comparison step */
    bitonic_merge_step<T, BLOCK_SIZE, CMP, UP>(data);

    /* run recursion (note that in this implementation, any subsequent merge
     * steps are with UP = false, only first step is UP) */
    bitonic_merge<T, BLOCK_SIZE / 2, CMP, false>(data);
}

/**
 * Perform multiple bitonic sorts by all active threads.
 *
 * @tparam T data item type
 *
 * @tparam BLOCK_SIZE Number of threads that work cooperatively on a block of
 * items that has exactly 2x BLOCK_SIZE items. Size of thread block must be
 * multiple of BLOCK_SIZE (multiple data blocks may be sorted by one thread
 * block).
 *
 * @tparam CMP comparator policy class that implements compare and swap on type
 * T
 *
 * @param data pointer to an array of T to be sorted in-place, of size at least
 * two times the total number of threads
 */
template<typename T, int BLOCK_SIZE, class CMP = ComparatorPolicy<T>>
__device__ __forceinline__ void
bitonic_sort(T *__restrict__ data)
{
    if constexpr (BLOCK_SIZE < 1)
        return;

    if constexpr (BLOCK_SIZE > 1) {
        // recursively sort halves of the input (this will be unrolled by
        // compiler)
        bitonic_sort<T, BLOCK_SIZE / 2, CMP>(data);
        if constexpr (BLOCK_SIZE > 32) {
            __syncthreads();
        } else {
            __syncwarp();
        }
    }

    bitonic_merge<T, BLOCK_SIZE, CMP>(data);
}

/** Perform one update step of bitonic topk algorithm.
 *
 * The algorith takes two inputs: current topk sub-result and new data (e.g.,
 * newly computed distances). It sorts inputs and runs a bitonic merge on
 * these, producing a bitonic sequence with the lower half of the data in topK,
 * and (as a side product) a bitonic sequence with the upper half of the data
 * in newData.
 *
 * Note that hat way the topk part is updated, but not entirely sorted (so we
 * save some time).
 *
 * @tparam T item data type
 *
 * @tparam BLOCK_SIZE Number of threads that work cooperatively on a block of
 * items. Both topk result and newData of one block have 2x BLOCK_SIZE items.
 * Size of thread block must be multiple of BLOCK_SIZE (multiple data blocks may
 * be sorted by one thread block).
 *
 * @tparam CMP comparator policy, such as ComparatorPolicy.
 *
 * @param topK array of `T`'s with intermediate top results, containing at
 * least 2x(number of threads) items
 *
 * @param newData array of new results to be merged into topK, of the same size
 */
template<typename T, int BLOCK_SIZE, class CMP = ComparatorPolicy<T>>
__device__ __forceinline__ void
bitonic_topk_update_opt(T *__restrict__ topK, T *__restrict__ newData)
{
    /* extra safety */
    if constexpr (BLOCK_SIZE < 1)
        return;

    if constexpr (BLOCK_SIZE > 1) {
        /* recursively sort halves of the input
         * (this will be unrolled by compiler) */
        bitonic_sort<T, BLOCK_SIZE, CMP>(topK);
        bitonic_sort<T, BLOCK_SIZE, CMP>(newData);
        if constexpr (BLOCK_SIZE > 32)
            __syncthreads();
        else
            __syncwarp();
    }

    /* masking lower bits from thread idx (like modulo STEP) */
    unsigned localMask = BLOCK_SIZE - 1;
    /* masking remaining bits from thread idx */
    unsigned blockMask = ~localMask;
    unsigned localIdx = (unsigned)threadIdx.x & localMask;
    /* offset of the block has 2x size of threads in local block
     * (each thread works on 2 values at once) */
    unsigned blockOffset = ((unsigned)threadIdx.x & blockMask) * 2;

    /* compare the upper and lower half of the data, producing a bitonic lower
     * half with values smaller than the upper half */
    CMP::order(
      topK[blockOffset + localIdx],
      newData[blockOffset + ((unsigned)BLOCK_SIZE * 2 - 1) - localIdx]);
    CMP::order(topK[blockOffset + localIdx + BLOCK_SIZE],
               newData[blockOffset + ((unsigned)BLOCK_SIZE - 1) - localIdx]);

    if constexpr (BLOCK_SIZE > 32)
        __syncthreads();
    else
        __syncwarp();
}

#endif
