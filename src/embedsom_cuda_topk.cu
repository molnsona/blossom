#include "embedsom_cuda.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cooperative_groups.h"

/*
 * Kernels
 */

template <typename F = float>
__inline__ __device__ F distance(const F* __restrict__ lhs, const F* __restrict__ rhs, const std::uint32_t dim)
{
	F sum = (F)0.0;
	for (std::uint32_t d = 0; d < dim; ++d) {
		F diff = *lhs++ - *rhs++;
		sum += diff * diff;
	}
	return sum; // squared euclidean
}
 
template <typename F = float>
__inline__ __device__ void bubbleUp(EsomCuda::TopkResult* const __restrict__ topK, std::uint32_t idx)
{
	while (idx > 0 && topK[idx - 1].distance > topK[idx].distance) {
		const typename EsomCuda::TopkResult tmp = topK[idx];
		topK[idx] = topK[idx - 1];
		topK[idx - 1] = tmp;
		--idx;
	}
}
 
 /**
 * Each thread iterates over whole point grid and computes kNN for a specified point
 * using insertion sort in global memory.
 */
template <typename F>
__global__ void topkBaseKernel(const F* __restrict__ points, const F* const __restrict__ grid, EsomCuda::TopkResult* __restrict__ topKs,
							   const std::uint32_t dim, const std::uint32_t n, const std::uint32_t gridSize, const std::uint32_t k)
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

		for (; gridIdx < k; ++gridIdx) 
		{
			topKs[gridIdx] = { distance<F>(points, grid + gridIdx * dim, dim), gridIdx };
			bubbleUp<F>(topKs, gridIdx);
		}

		for (; gridIdx < gridSize; ++gridIdx) 
		{
			F dist = distance<F>(points, grid + gridIdx * dim, dim);

			if (topKs[k - 1].distance > dist) {
				topKs[k - 1] = { dist, gridIdx };
				bubbleUp<F>(topKs, k - 1);
			}
		}
	}
}


void EsomCuda::runTopkBaseKernel()
{
	constexpr unsigned int blockSize = 256;
	unsigned int blockCount = (mPointsCount + blockSize - 1) / blockSize;
	topkBaseKernel<float><<<blockCount, blockSize>>>(
		mCuPoints, mCuLandmarksHighDim, mTopkResults, mDim, mPointsCount, mLandmarksCount, mTopK);
	CUCH(cudaGetLastError());
}
