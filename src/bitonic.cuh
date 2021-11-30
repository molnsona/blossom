#ifndef ESOM_CUDA_BITONIC_CUH
#define ESOM_CUDA_BITONIC_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

template<typename T>
class ComparatorPolicy
{
public:
    static __device__ void compareAndSwap(T &a, T &b)
    {
        if (a > b) {
            T tmp = a;
            a = b;
            b = tmp;
        }
    }
};

template<>
__device__ void
ComparatorPolicy<float>::compareAndSwap(float &a, float &b)
{
    float A = a;
    float B = b;
    a = min(A, B);
    b = max(A, B);
};

/**
 *
 */
template<typename T,
         int BLOCK_SIZE,
         class CMP = ComparatorPolicy<T>,
         bool UP = false>
__device__ void
bitonic_merge_step(T *__restrict__ data)
{
    if (BLOCK_SIZE < 1) {
        return; // this should never happen, but just to be safe
    }

    unsigned localMask =
      (unsigned)BLOCK_SIZE -
      1; // masking lower bits from thread idx (like modulo STEP)
    unsigned blockMask = ~localMask; // masking remaining bits from thread idx
    unsigned localIdx = (unsigned)threadIdx.x & localMask;
    unsigned blockOffset =
      ((unsigned)threadIdx.x & blockMask) *
      2; // offset of the block has 2x size of threads in local block (each
         // thread works on 2 values at once)

    // UP is the first phase of the merge, which is different (compares first
    // with last, second with last second ...) remaining steps use DOWN (UP ==
    // false) which is regular pairwise comparison within STEP-size sub-block
    unsigned secondIdx =
      UP ? blockOffset + ((unsigned)BLOCK_SIZE * 2 - 1) - localIdx
         : blockOffset + localIdx + (unsigned)BLOCK_SIZE;
    CMP::compareAndSwap(data[blockOffset + localIdx], data[secondIdx]);

    if (BLOCK_SIZE > 32) {
        __syncthreads();
    } else {
        __syncwarp();
    }
}

/**
 *
 */
template<typename T,
         int BLOCK_SIZE,
         class CMP = ComparatorPolicy<T>,
         bool UP = true>
__device__ void
bitonic_merge(T *__restrict__ data)
{
    if (BLOCK_SIZE < 1) {
        return; // recursion guard
    }

    bitonic_merge_step<T, BLOCK_SIZE, CMP, UP>(data);

    // and now employ template recursion to expand all the steps
    bitonic_merge<T, BLOCK_SIZE / 2, CMP, false>(
      data); // note that any subsequent merge steps are with UP = false (only
             // first step is UP)
}

/**
 * Perform multiple bitonic sorts by all active threads.
 * @tparam T item type
 * @tparam BLOCK_SIZE Number of threads that work cooperatively on a block of
 * items that has exactly 2x BLOCK_SIZE items. Size of thread block must be
 * multiple of BLOCK_SIZE (multiple data blocks may be sorted by one thread
 * block).
 * @tparam CMP comparator policy class that implements compare and swap on T
 * type
 * @param data pointer to the memory where the data are sorted inplace
 * (containing at least 2x threads in a block of T items)
 */
template<typename T, int BLOCK_SIZE, class CMP = ComparatorPolicy<T>>
__device__ __forceinline__ void
bitonic_sort(T *__restrict__ data)
{
    if (BLOCK_SIZE < 1)
        return;

    if (BLOCK_SIZE > 1) {
        // recursively sort halves of the input (this will be unrolled by
        // compiler)
        bitonic_sort<T, BLOCK_SIZE / 2, CMP>(data);
        if (BLOCK_SIZE > 32) {
            __syncthreads();
        } else {
            __syncwarp();
        }
    }

    bitonic_merge<T, BLOCK_SIZE, CMP>(data);
}

/**
 * Perform one update step of bitonic topk algorithm.
 * The algorith takes two inputs - current topk result and new data (e.g., newly
 * computed distances). It partially sorts both inputs but only to ensure that
 * lower part of topk and upper part of new data may be directly compared and
 * swapped. That way the topk part is updated, but not entirely sorted (so we
 * save some time).
 * @tparam T item type
 * @tparam BLOCK_SIZE Number of threads that work cooperatively on a block of
 * items. Both topk result and newData of one block have 2x BLOCK_SIZE items.
 * Size of thread block must be multiple of BLOCK_SIZE (multiple data blocks may
 * be sorted by one thread block).
 * @tparam CMP comparator policy class that implements compare and swap on T
 * type
 * @param topk intermediate topk result (containing at least 2x threads in a
 * block of T items)
 * @param newData to be merged in topk (containing at least 2x threads in a
 * block of T items)
 */
template<typename T, int BLOCK_SIZE, class CMP = ComparatorPolicy<T>>
__device__ __forceinline__ void
bitonic_topk_update(T *__restrict__ topK, T *__restrict__ newData)
{
    if (BLOCK_SIZE < 1) {
        return;
    }

    if (BLOCK_SIZE > 1) {
        // recursively sort halves of the input (this will be unrolled by
        // compiler)
        bitonic_sort<T, BLOCK_SIZE / 2, CMP>(topK);
        bitonic_sort<T, BLOCK_SIZE / 2, CMP>(newData);
        if (BLOCK_SIZE > 32) {
            __syncthreads();
        } else {
            __syncwarp();
        }
    }

    // preform only the first step of the merge (the UP merge that makes sure
    // the smaller half of the data is in the first half of the array)
    bitonic_merge_step<T, BLOCK_SIZE, CMP, true>(topK);
    bitonic_merge_step<T, BLOCK_SIZE, CMP, true>(newData);

    // compare and swap upper new data with lower topk data
    unsigned localMask =
      BLOCK_SIZE - 1; // masking lower bits from thread idx (like modulo STEP)
    unsigned blockMask = ~localMask; // masking remaining bits from thread idx
    unsigned localIdx = (unsigned)threadIdx.x & localMask;
    unsigned blockOffset =
      ((unsigned)threadIdx.x & blockMask) *
      2; // offset of the block has 2x size of threads in local block (each
         // thread works on 2 values at once)

    //	CMP::compareAndSwap(topK[blockOffset + localIdx + BLOCK_SIZE],
    //newData[blockOffset + localIdx]); 	CMP::compareAndSwap(topK[blockOffset +
    //localIdx], newData[blockOffset + ((unsigned)BLOCK_SIZE * 2 - 1) -
    //localIdx]); 	CMP::compareAndSwap(topK[blockOffset + localIdx + BLOCK_SIZE],
    //newData[blockOffset + ((unsigned)BLOCK_SIZE - 1) - localIdx]);

    // overwrite lower half of topk with upper half or new data
    topK[blockOffset + localIdx + BLOCK_SIZE] = newData[blockOffset + localIdx];

    if (BLOCK_SIZE > 32) {
        __syncthreads();
    } else {
        __syncwarp();
    }

    // note that after this the topk data are not necesarily sorted!
}

/**
 * Perform one update step of bitonic topk algorithm.
 * The algorith takes two inputs - current topk result and new data (e.g., newly
 * computed distances). It partially sorts both inputs but only to ensure that
 * lower part of topk and upper part of new data may be directly compared and
 * swapped. That way the topk part is updated, but not entirely sorted (so we
 * save some time).
 * @tparam T item type
 * @tparam BLOCK_SIZE Number of threads that work cooperatively on a block of
 * items. Both topk result and newData of one block have 2x BLOCK_SIZE items.
 * Size of thread block must be multiple of BLOCK_SIZE (multiple data blocks may
 * be sorted by one thread block).
 * @tparam CMP comparator policy class that implements compare and swap on T
 * type
 * @param topk intermediate topk result (containing at least 2x threads in a
 * block of T items)
 * @param newData to be merged in topk (containing at least 2x threads in a
 * block of T items)
 */
template<typename T, int BLOCK_SIZE, class CMP = ComparatorPolicy<T>>
__device__ __forceinline__ void
bitonic_topk_update_opt(T *__restrict__ topK, T *__restrict__ newData)
{
    if (BLOCK_SIZE < 1) {
        return;
    }

    if (BLOCK_SIZE > 1) {
        // recursively sort halves of the input (this will be unrolled by
        // compiler)
        bitonic_sort<T, BLOCK_SIZE, CMP>(topK);
        bitonic_sort<T, BLOCK_SIZE, CMP>(newData);
        if (BLOCK_SIZE > 32) {
            __syncthreads();
        } else {
            __syncwarp();
        }
    }

    // compare and swap upper new data with lower topk data
    unsigned localMask =
      BLOCK_SIZE - 1; // masking lower bits from thread idx (like modulo STEP)
    unsigned blockMask = ~localMask; // masking remaining bits from thread idx
    unsigned localIdx = (unsigned)threadIdx.x & localMask;
    unsigned blockOffset =
      ((unsigned)threadIdx.x & blockMask) *
      2; // offset of the block has 2x size of threads in local block (each
         // thread works on 2 values at once)

    CMP::compareAndSwap(
      topK[blockOffset + localIdx],
      newData[blockOffset + ((unsigned)BLOCK_SIZE * 2 - 1) - localIdx]);
    CMP::compareAndSwap(
      topK[blockOffset + localIdx + BLOCK_SIZE],
      newData[blockOffset + ((unsigned)BLOCK_SIZE - 1) - localIdx]);

    if (BLOCK_SIZE > 32) {
        __syncthreads();
    } else {
        __syncwarp();
    }

    // note that after this the topk data are not necesarily sorted!
}

#endif
