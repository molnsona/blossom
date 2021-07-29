#include "embedsom_cuda.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cooperative_groups.h"
#include "cub/block/block_reduce.cuh"

namespace cg = cooperative_groups;

template <typename F>
using TopkResult = typename TopkProblemInstance<F>::Result;

template <typename F>
struct ArrayFloatType {};

template <>
struct ArrayFloatType<float>
{using Type2 = float2;};

template <>
struct ArrayFloatType<double>
{using Type2 = double2;};

template <typename F>
__inline__ __device__ void sortedDistsToScores(TopkResult<F>* const __restrict__ neighbors, const std::size_t adjustedK, const std::size_t k,
											   const F boost)
{
	// compute the distance distribution for the scores
	F mean = 0, sd = 0, wsum = 0;
	for (std::uint32_t i = 0; i < adjustedK; ++i) {
		const F tmp = sqrt(neighbors[i].distance);
		const F w = 1 / F(i + 1);
		mean += tmp * w;
		sd += tmp * tmp * w;
		wsum += w;
		neighbors[i].distance = tmp;
	}

	mean /= wsum;
	sd = boost / sqrt(sd / wsum - mean * mean);
	const F nmax = ProjectionProblemInstance<F>::maxAvoidance / neighbors[adjustedK - 1].distance;

	// convert the stuff to scores
	for (std::uint32_t i = 0; i < k; ++i) {
		if (k < adjustedK)
			neighbors[i].distance =
				exp((mean - neighbors[i].distance) * sd) * (1 - exp(neighbors[i].distance * nmax - ProjectionProblemInstance<F>::maxAvoidance));
		else
			neighbors[i].distance = exp((mean - neighbors[i].distance) * sd);
	}
}

/**
 * Helper structure to help transform neighbor distances to scores.
 * Shares `distance` field for storing scores and loading distances.
 */
template <typename F>
struct SharedNeighborStorage
{
	TopkResult<F>* const __restrict__ neighbors;

	__forceinline__ __device__ F getNeighborDistance(const std::uint32_t idx) const
	{
		return neighbors[idx].distance;
	}

	__forceinline__ __device__ void storeScore(const std::uint32_t idx, const F score)
	{
		neighbors[idx].distance = score;
	}
};

/**
 * Helper structure to help transform neighbor distances to scores.
 * Separate arrays for distances and scores.
 */
template <typename F>
struct NeighborScoreStorage
{
	const TopkResult<F>* const __restrict__ neighbors;
	F* const __restrict__ scores;

	__forceinline__ __device__ F getNeighborDistance(const std::uint32_t idx) const
	{
		return neighbors[idx].distance;
	}

	__forceinline__ __device__ void storeScore(const std::uint32_t idx, const F score)
	{
		scores[idx] = score;
	}
};

template <typename F, class SCORE_STORAGE>
__inline__ __device__ void sortedDistsToScoresWarp(SCORE_STORAGE storage, char* const __restrict__ sharedMemory,
												   const std::size_t adjustedK, const std::size_t k, const F boost)
{
	typedef cub::WarpReduce<F> WarpReduce;

	// each thread in warp can have at most 3 scores as k <= 64 (adjustedK <= k + 1)
	F tmpScores[3];
	F lastScore;

	// compute the distance distribution for the scores
	F mean = 0, sd = 0, wsum = 0;
	for (std::uint32_t i = threadIdx.x; i < adjustedK; i += warpSize) {
		const F tmp = sqrt(storage.getNeighborDistance(i));
		const F w = 1 / F(i + 1);
		mean += tmp * w;
		sd += tmp * tmp * w;
		wsum += w;
		tmpScores[i / warpSize] = tmp;
	}

	auto* const reduceStorage = reinterpret_cast<typename WarpReduce::TempStorage*>(sharedMemory);

	{
		mean = WarpReduce(*reduceStorage).Sum(mean);
		__syncwarp();
		sd = WarpReduce(*reduceStorage).Sum(sd);
		__syncwarp();
		wsum = WarpReduce(*reduceStorage).Sum(wsum);
	}

	{
		mean = cub::ShuffleIndex<32>(mean, 0, 0xffffffff);
		sd = cub::ShuffleIndex<32>(sd, 0, 0xffffffff);
		wsum = cub::ShuffleIndex<32>(wsum, 0, 0xffffffff);

		const auto lastScoreThreadIdx = (adjustedK - 1) % warpSize;
		const auto lastScoreIdx = (adjustedK - 1) / warpSize;
		lastScore = cub::ShuffleIndex<32>(tmpScores[lastScoreIdx], lastScoreThreadIdx, 0xffffffff);
	}

	mean /= wsum;
	sd = boost / sqrt(sd / wsum - mean * mean);
	const F nmax = ProjectionProblemInstance<F>::maxAvoidance / lastScore;

	// convert the stuff to scores
	if (k < adjustedK)
		for (std::uint32_t i = threadIdx.x; i < k; i += warpSize) {
			const auto scoreIdx = i / warpSize;
			const F score = exp((mean - tmpScores[scoreIdx]) * sd) * (1 - exp(tmpScores[scoreIdx] * nmax - ProjectionProblemInstance<F>::maxAvoidance));
			storage.storeScore(i, score);
		}
	else
		for (std::uint32_t i = threadIdx.x; i < k; i += warpSize)
			storage.storeScore(i, exp((mean - tmpScores[i / warpSize]) * sd));
}

template <typename F>
__inline__ __device__ void addGravity(const F score, const F* const __restrict__ grid2DPoint, F* const __restrict__ mtx)
{
	const F gs = score * ProjectionProblemInstance<F>::gridGravity;

	mtx[0] += gs;
	mtx[3] += gs;
	mtx[4] += gs * grid2DPoint[0];
	mtx[5] += gs * grid2DPoint[1];
}

template <typename F>
__inline__ __device__ typename ArrayFloatType<F>::Type2 euclideanProjection(const F* const __restrict__ point, const F* const __restrict__ gridPointI,
																			const F* const __restrict__ gridPointJ, const std::uint32_t dim)
{
	typename ArrayFloatType<F>::Type2 result { 0.0, 0.0 };
	for (std::uint32_t k = 0; k < dim; ++k) {
		const F tmp = gridPointJ[k] - gridPointI[k];
		result.y += tmp * tmp;
		result.x += tmp * (point[k] - gridPointI[k]);
	}
	return result;
}

template <typename F>
__inline__ __device__ void addApproximation(const F scoreI, const F scoreJ, const F* const __restrict__ grid2DPointI,
											const F* const __restrict__ grid2DPointJ, const F adjust, const F scalarProjection,
											F* const __restrict__ mtx)
{
	F h[2], hp = 0;
	#pragma unroll
	for (std::uint32_t i = 0; i < 2; ++i) {
		h[i] = grid2DPointJ[i] - grid2DPointI[i];
		hp += h[i] * h[i];
	}

	if (hp < ProjectionProblemInstance<F>::zeroAvoidance)
		return;

	const F exponentPart = scalarProjection - .5;
	const F s = scoreI * scoreJ * pow(1 + hp, adjust) * exp(- exponentPart * exponentPart);
	const F sihp = s / hp;
	const F rhsc = s * (scalarProjection + (h[0] * grid2DPointI[0] + h[1] * grid2DPointI[1]) / hp);

	mtx[0] += h[0] * h[0] * sihp;
	mtx[1] += h[0] * h[1] * sihp;
	mtx[2] += h[1] * h[0] * sihp;
	mtx[3] += h[1] * h[1] * sihp;
	mtx[4] += h[0] * rhsc;
	mtx[5] += h[1] * rhsc;
}

/**
 * One thread computes embedding for one point.
 */
template <typename F>
__global__ void projectionBaseKernel(const F* __restrict__ points, const F* const __restrict__ grid, const F* const __restrict__ grid2d,
									 TopkResult<F>* __restrict__ neighbors, F* __restrict__ projections, const std::uint32_t dim,
									 const std::uint32_t n, const std::uint32_t gridSize, const std::uint32_t k, const F adjust, const F boost)
{
	// assign defaults and generate scores
	{
		const std::uint32_t adjustedK = k < gridSize ? k + 1 : k;
		const std::uint32_t pointIdx = blockIdx.x * blockDim.x + threadIdx.x;

		if (pointIdx >= n)
			return;

		points = points + pointIdx * dim;
		neighbors = neighbors + pointIdx * adjustedK;
		projections = projections + pointIdx * 2;

		sortedDistsToScores<F>(neighbors, adjustedK, k, boost);
	}

	F mtx[6];
	memset(mtx, 0, 6 * sizeof(F));

	for (std::uint32_t i = 0; i < k; ++i) 
	{
		const std::uint32_t idxI = neighbors[i].index;
		const F scoreI = neighbors[i].distance;

		addGravity(scoreI, grid2d + idxI * 2, mtx);

		for (std::uint32_t j = i + 1; j < k; ++j) 
		{
			const std::uint32_t idxJ = neighbors[j].index;
			const F scoreJ = neighbors[j].distance;

			const auto result = euclideanProjection<F>(points, grid + idxI * dim, grid + idxJ * dim, dim);
			F scalarProjection = result.x;
			const F squaredGridPointsDistance = result.y;

			if (squaredGridPointsDistance == F(0))
				continue;

			scalarProjection /= squaredGridPointsDistance;

			addApproximation(scoreI, scoreJ, grid2d + idxI * 2, grid2d + idxJ * 2, adjust, scalarProjection, mtx);
		}
	}

	// solve linear equation
	const F det = mtx[0] * mtx[3] - mtx[1] * mtx[2];
	projections[0] = (mtx[4] * mtx[3] - mtx[5] * mtx[2]) / det;
	projections[1] = (mtx[0] * mtx[5] - mtx[1] * mtx[4]) / det;
}

// runner wrapped in a class
template <typename F>
void ProjectionBaseKernel<F>::run(const ProjectionProblemInstance<F>& in, CudaExecParameters& exec)
{
	unsigned int blockCount = (in.n + exec.blockSize - 1) / exec.blockSize;
	projectionBaseKernel<F><<<blockCount, exec.blockSize>>>(in.points, in.grid, in.grid2d, in.neighbors, in.projections, in.dim, in.n, in.gridSize,
															in.k, in.adjust, in.boost);
}

template <typename INDEXER>
__inline__ __device__ uint2 getIndices(std::uint32_t plainIndex, std::uint32_t k) {}

/**
 * Assigns consecutive indices to consecutive threads.
 * However, branch divergence occurs.
 * Thread index assignments:
 * k|	0	1	2	3	4
 * -+--------------------
 * 0|		0	1	2	3
 * 1|			4	5	6
 * 2|				7	8
 * 3|					9
 * 4|
 */
struct BaseIndexer {};
template <>
__inline__ __device__ uint2 getIndices<BaseIndexer>(std::uint32_t plainIndex, std::uint32_t k)
{
	--k;
	uint2 indices { 0, 0 };
	while (plainIndex >= k) {
		++indices.x;
		plainIndex -= k--;
	}
	indices.y = plainIndex + 1 + indices.x;
	return indices;
}

/**
 * "Concatenates" 2 columns into one (1. and k-1., 2. and k-2., ...) and 
 * creates indexing on top of k * k/2 rectangle.
 * No branch divergence.
 * Thread index assignments:
 * k|	0	1	2	3	4
 * -+--------------------
 * 0|		0	1	2	3
 * 1|			5	6	7
 * 2|				8	9
 * 3|					4
 * 4|
 */
struct RectangleIndexer {};
template <>
__inline__ __device__ uint2 getIndices<RectangleIndexer>(std::uint32_t plainIndex, std::uint32_t k)
{
	uint2 indices;
	const std::uint32_t tempI = plainIndex / k;
	const std::uint32_t tempJ = plainIndex % k;
	const auto invertedI = k - 1 - tempI;
	indices.x = tempJ < invertedI ? tempI : invertedI - 1;
	indices.y = (tempJ < invertedI ? tempJ : tempJ - invertedI) + indices.x + 1;

	return indices;
}

/**
 * One block computes embedding for one point, using CUB block reduce for matrix reduction.
 */
template <typename F, typename INDEXER>
__global__ void projectionBlockKernel(const F* __restrict__ points, const F* const __restrict__ grid, const F* const __restrict__ grid2d,
									  TopkResult<F>* __restrict__ neighbors, F* __restrict__ projections, const std::uint32_t dim,
									  const std::uint32_t n, const std::uint32_t gridSize, const std::uint32_t k, const F adjust, const F boost,
									  const std::uint32_t itemsPerThread)
{
	typedef cub::BlockReduce<F, 1024> BlockReduce;
	__shared__ typename BlockReduce::TempStorage temp_storage;

	// assign defaults and generate scores
	{
		const std::uint32_t adjustedK = k < gridSize ? k + 1 : k;

		points = points + blockIdx.x * dim;
		neighbors = neighbors + blockIdx.x * adjustedK;
		projections = projections + blockIdx.x * 2;

		if (threadIdx.x < 32)
			// silently assuming that CUB storage needed for warp reduction is smaller that the one needed for block reduction
			sortedDistsToScoresWarp<F>(SharedNeighborStorage<F> { neighbors }, reinterpret_cast<char*>(&temp_storage), adjustedK, k, boost);

		__syncthreads();
	}

	F mtx[6];
	memset(mtx, 0, 6 * sizeof(F));

	for (std::uint32_t i = threadIdx.x; i < k; i += blockDim.x) {
		const auto neighbor = neighbors[i];
		addGravity(neighbor.distance, grid2d + neighbor.index * 2, mtx);
	}

	const std::uint32_t neighborPairs = (k * (k - 1)) / 2;
	for (std::uint32_t i = threadIdx.x * itemsPerThread; i < neighborPairs; i += blockDim.x * itemsPerThread) 
	{
		for (size_t j = 0; j < itemsPerThread; ++j) 
		{
			const auto threadIndex = i + j;
			if (threadIndex >= neighborPairs)
				continue;

			const auto indices = getIndices<INDEXER>(threadIndex, k);

			const std::uint32_t idxI = neighbors[indices.x].index;
			const std::uint32_t idxJ = neighbors[indices.y].index;
			const F scoreI = neighbors[indices.x].distance;
			const F scoreJ = neighbors[indices.y].distance;

			const auto result = euclideanProjection<F>(points, grid + idxI * dim, grid + idxJ * dim, dim);
			F scalarProjection = result.x;
			const F squaredGridPointsDistance = result.y;

			if (squaredGridPointsDistance == F(0))
				continue;

			scalarProjection /= squaredGridPointsDistance;

			addApproximation(scoreI, scoreJ, grid2d + idxI * 2, grid2d + idxJ * 2, adjust, scalarProjection, mtx);
		}
	}

	#pragma unroll
	for (size_t i = 0; i < 6; ++i) {
		mtx[i] = BlockReduce(temp_storage).Sum(mtx[i], blockDim.x);
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		const F det = mtx[0] * mtx[3] - mtx[1] * mtx[2];
		projections[0] = (mtx[4] * mtx[3] - mtx[5] * mtx[2]) / det;
		projections[1] = (mtx[0] * mtx[5] - mtx[1] * mtx[4]) / det;
	}
}

// runner wrapped in a class
template <typename F>
void ProjectionBlockKernel<F>::run(const ProjectionProblemInstance<F>& in, CudaExecParameters& exec)
{
	unsigned int blockCount = in.n;
	projectionBlockKernel<F, BaseIndexer><<<blockCount, exec.blockSize>>>(in.points, in.grid, in.grid2d, in.neighbors, in.projections, in.dim, in.n,
																		  in.gridSize, in.k, in.adjust, in.boost, exec.itemsPerThread);
}

// runner wrapped in a class
template <typename F>
void ProjectionBlockRectangleIndexKernel<F>::run(const ProjectionProblemInstance<F>& in, CudaExecParameters& exec)
{
	unsigned int blockCount = in.n;
	projectionBlockKernel<F, RectangleIndexer><<<blockCount, exec.blockSize>>>(in.points, in.grid, in.grid2d, in.neighbors, in.projections, in.dim,
																			   in.n, in.gridSize, in.k, in.adjust, in.boost, exec.itemsPerThread);
}

/**
 * One block computes embedding for one point, using CUB block reduce for matrix reduction.
 * All data that is used in the computation is copied to shared memory.
 */
template <typename F>
__global__ void projectionBlockSharedKernel(const F* const __restrict__ points, const F* const __restrict__ grid, const F* const __restrict__ grid2d,
											TopkResult<F>* __restrict__ neighbors, F* __restrict__ projections, const std::uint32_t dim,
											const std::uint32_t n, const std::uint32_t gridSize, const std::uint32_t k, const F adjust, const F boost)
{
	extern __shared__ char sharedMemory[];
	typedef cub::BlockReduce<F, 1024> BlockReduce;
	auto* const __restrict__ reduceStorage = reinterpret_cast<typename BlockReduce::TempStorage*>(sharedMemory);
	F* const __restrict__ pointCache = reinterpret_cast<F*>(sharedMemory + sizeof(typename BlockReduce::TempStorage));
	F* const __restrict__ scoresCache = pointCache + dim;
	F* const __restrict__ grid2dCache = scoresCache + k;
	F* const __restrict__ gridCache = grid2dCache + 2 * k;

	// assign defaults and generate scores
	{
		const std::uint32_t adjustedK = k < gridSize ? k + 1 : k;

		neighbors = neighbors + blockIdx.x * adjustedK;
		projections = projections + blockIdx.x * 2;

		int copyIdx = threadIdx.x;

		if (threadIdx.x < warpSize) {
			// silently assuming that CUB storage needed for warp reduction is smaller that the one needed for block reduction
			sortedDistsToScoresWarp<F>(NeighborScoreStorage<F> { neighbors, scoresCache }, sharedMemory, adjustedK, k, boost);
			copyIdx += blockDim.x;
		}

		//ugly copying to shared
		{
			for (; copyIdx < warpSize + dim; copyIdx += blockDim.x) {
				auto cacheIdx = copyIdx - warpSize;
				pointCache[cacheIdx] = points[blockIdx.x * dim + cacheIdx];
			}

			for (; copyIdx < warpSize + dim + k * 2; copyIdx += blockDim.x) {
				auto cacheIdx = copyIdx - warpSize - dim;
				grid2dCache[cacheIdx] = grid2d[neighbors[cacheIdx / 2].index * 2 + (cacheIdx % 2)];
			}

			for (; copyIdx < warpSize + dim + k * 2 + k * dim; copyIdx += blockDim.x) {
				auto cacheIdx = copyIdx - k * 2 - warpSize - dim;
				auto globIdx = cacheIdx / k;
				auto globOff = cacheIdx % k;
				gridCache[cacheIdx] = grid[neighbors[globIdx].index * dim + globOff];
			}
		}
		
		__syncthreads();
	}

	F mtx[6];
	memset(mtx, 0, 6 * sizeof(F));

	for (std::uint32_t i = threadIdx.x; i < k; i += blockDim.x) {
		addGravity(scoresCache[i], grid2dCache + i * 2, mtx);
	}

	const std::uint32_t neighborPairs = (k * (k - 1)) / 2;
	for (std::uint32_t i = threadIdx.x; i < neighborPairs; i += blockDim.x) 
	{
		const auto indices = getIndices<RectangleIndexer>(i, k);

		const auto I = indices.x;
		const auto J = indices.y;
		const F scoreI = scoresCache[I];
		const F scoreJ = scoresCache[J];

		// addGravity(scoreI, grid2dCache + I * 2, mtx); POSSIBLY BUG SOMEWHERE

		const auto result = euclideanProjection<F>(pointCache, gridCache + I * dim, gridCache + J * dim, dim);
		F scalarProjection = result.x;
		const F squaredGridPointsDistance = result.y;

		if (squaredGridPointsDistance == F(0))
			continue;

		scalarProjection /= squaredGridPointsDistance;

		addApproximation(scoreI, scoreJ, grid2dCache + I * 2, grid2dCache + J * 2, adjust, scalarProjection, mtx);
	}

	#pragma unroll
	for (size_t i = 0; i < 6; ++i) {
		mtx[i] = BlockReduce(*reduceStorage).Sum(mtx[i], blockDim.x);
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		const F det = mtx[0] * mtx[3] - mtx[1] * mtx[2];
		projections[0] = (mtx[4] * mtx[3] - mtx[5] * mtx[2]) / det;
		projections[1] = (mtx[0] * mtx[5] - mtx[1] * mtx[4]) / det;
	}
}

// runner wrapped in a class
template <typename F>
void ProjectionBlockSharedKernel<F>::run(const ProjectionProblemInstance<F>& in, CudaExecParameters& exec)
{
	unsigned int blockCount = in.n;
	exec.sharedMemorySize = sizeof(typename cub::BlockReduce<F, 1024>::TempStorage) + // for reduction
							in.dim * sizeof(F) +									  // for point
							in.k * sizeof(F) +										  // for scores
							in.k * 2 * sizeof(F) +									  // for grid2d points
							in.k * in.dim * sizeof(F);								  // for grid points
	projectionBlockSharedKernel<F><<<blockCount, exec.blockSize, exec.sharedMemorySize>>>(in.points, in.grid, in.grid2d, in.neighbors, in.projections,
																						  in.dim, in.n, in.gridSize, in.k, in.adjust, in.boost);
}

/*
 * Explicit template instantiation.
 */
template <typename F>
void instantiateKernelRunnerTemplates()
{
	ProjectionProblemInstance<F> instance(nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, 0, 0, F(0), F(0));
	CudaExecParameters exec;

	ProjectionBaseKernel<F>::run(instance, exec);
	ProjectionBlockKernel<F>::run(instance, exec);
	ProjectionBlockRectangleIndexKernel<F>::run(instance, exec);
	ProjectionBlockSharedKernel<F>::run(instance, exec);
}

template void instantiateKernelRunnerTemplates<float>();
#ifndef NO_DOUBLES
template void instantiateKernelRunnerTemplates<double>();
#endif
