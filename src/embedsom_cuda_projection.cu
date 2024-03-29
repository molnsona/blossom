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

#define _CG_ABI_EXPERIMENTAL

#include "cooperative_groups.h"
#include "cooperative_groups/reduce.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace cg = cooperative_groups;

template<typename F = float>
struct EmbedSOMConstants
{
public:
    static constexpr F minBoost = F(1e-5);
    static constexpr F maxAvoidance = F(10);
    static constexpr F zeroAvoidance = F(1e-10);
    static constexpr F gridGravity = F(1e-5);
};

/** Helper structure to transform neighbor distances to scores.
 *
 * Shares `distance` field for storing scores and loading distances.
 */
template<typename F>
struct SharedNeighborStorage
{
    knn_entry<F> *const __restrict__ neighbors;
    __forceinline__ __device__ F getNeighborDistance(const uint32_t idx) const
    {
        return neighbors[idx].distance;
    }
    __forceinline__ __device__ void storeScore(const uint32_t idx,
                                               const F score)
    {
        neighbors[idx].distance = score;
    }
};

/** Helper structure to help transform neighbor distances to scores.
 *
 * Separate arrays for distances and scores.
 */
template<typename F>
struct NeighborScoreStorage
{
    const knn_entry<F> *const __restrict__ neighbors;
    F *const __restrict__ scores;

    __forceinline__ __device__ F getNeighborDistance(const uint32_t idx) const
    {
        return neighbors[idx].distance;
    }

    __forceinline__ __device__ void storeScore(const uint32_t idx,
                                               const F score)
    {
        scores[idx] = score;
    }
};

/** Sum a value across the warp */
template<typename F>
__inline__ __device__ F
warpReduceSum(F val)
{
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

/** Read aligned data entries for EmbedSOM projection */
template<typename F, typename ArrayF>
__inline__ __device__ void
readAligned(F *const __restrict__ dst,
            const F *const __restrict__ src,
            const knn_entry<F> *const __restrict__ neighbors,
            const uint32_t n,
            const uint32_t dim,
            const uint32_t groupRank,
            const uint32_t groupSize)
{
    constexpr auto size = sizeof(ArrayF) / sizeof(F);

    const uint32_t loadsCount = dim / size;
    const uint32_t dimX = dim / size;

    ArrayF *const dstX = reinterpret_cast<ArrayF *>(dst);
    const ArrayF *const srcX = reinterpret_cast<const ArrayF *>(src);

    for (size_t i = groupRank; i < n * loadsCount; i += groupSize) {
        const auto idx = i / loadsCount;
        const auto off = i % loadsCount;

        dstX[idx * dimX + off] = srcX[neighbors[idx].index * dimX + off];
    }
}

/** Version of readAligned() for the 2D grid */
template<typename F>
__inline__ __device__ void
readAlignedGrid2D(F *const __restrict__ dst,
                  const F *const __restrict__ src,
                  const knn_entry<F> *const __restrict__ neighbors,
                  const uint32_t n,
                  const uint32_t groupRank,
                  const uint32_t groupSize)
{
    using ArrayT = typename Vec<2, F>::Type;

    ArrayT *const dstX = reinterpret_cast<ArrayT *>(dst);
    const ArrayT *const srcX = reinterpret_cast<const ArrayT *>(src);

    for (size_t i = groupRank; i < n; i += groupSize)
        dstX[i] = srcX[neighbors[i].index];
}

/** Load data into shared memory cache */
template<typename F>
__inline__ __device__ void
storeToCache(const uint32_t groupRank,
             const uint32_t groupSize,
             const F *const __restrict__ points,
             F *const __restrict__ pointsCache,
             const F *const __restrict__ grid,
             F *const __restrict__ gridCache,
             const F *const __restrict__ grid2d,
             F *const __restrict__ grid2dCache,
             const knn_entry<F> *const __restrict__ neighbors,
             const uint32_t dim,
             const uint32_t gridCacheLeadingDim,
             const uint32_t k)
{
    auto copyIdx = groupRank;
    for (; copyIdx < dim; copyIdx += groupSize)
        pointsCache[copyIdx] = points[copyIdx];

    copyIdx -= dim;

    for (; copyIdx < k * dim; copyIdx += groupSize) {
        auto globIdx = copyIdx / dim;
        auto globOff = copyIdx % dim;
        gridCache[globIdx * gridCacheLeadingDim + globOff] =
          grid[neighbors[globIdx].index * dim + globOff];
    }

    readAlignedGrid2D<F>(
      grid2dCache, grid2d, neighbors, k, groupRank, groupSize);
}

/** Compute scores for a vector of sorted distances */
template<typename F>
__inline__ __device__ void
sortedDistsToScores(knn_entry<F> *const __restrict__ neighbors,
                    const std::size_t adjustedK,
                    const std::size_t k,
                    const F boost)
{
    // compute the distance distribution for the scores
    F mean = 0, sd = 0, wsum = 0;
    for (uint32_t i = 0; i < adjustedK; ++i) {
        const F tmp = sqrt(neighbors[i].distance);
        const F w = 1 / F(i + 1);
        mean += tmp * w;
        sd += tmp * tmp * w;
        wsum += w;
        neighbors[i].distance = tmp;
    }

    mean /= wsum;
    sd = boost / sqrt(sd / wsum - mean * mean);
    const F nmax =
      EmbedSOMConstants<F>::maxAvoidance / neighbors[adjustedK - 1].distance;

    // convert the stuff to scores
    for (uint32_t i = 0; i < k; ++i) {
        if (k < adjustedK)
            neighbors[i].distance =
              exp((mean - neighbors[i].distance) * sd) *
              (1 - exp(neighbors[i].distance * nmax -
                       EmbedSOMConstants<F>::maxAvoidance));
        else
            neighbors[i].distance = exp((mean - neighbors[i].distance) * sd);
    }
}

/** Convert the distances to scores using thread_block_tile and its built in
 * functions for reduction and shuffle.
 */
template<typename F, class SCORE_STORAGE, class TILE>
__inline__ __device__ void
sortedDistsToScoresGroup(const TILE &tile,
                         SCORE_STORAGE storage,
                         const std::size_t adjustedK,
                         const std::size_t k,
                         const F boost)
{
    // TRICKY: if k is big enough and tile is small enough, this array may
    // overflow. Should be MAX_K / tile.size()
    F tmpScores[10];
    F lastScore;

    // compute the distance distribution for the scores
    F mean = 0, sd = 0, wsum = 0;
    for (uint32_t i = tile.thread_rank(); i < adjustedK; i += tile.size()) {
        const F tmp = sqrt(storage.getNeighborDistance(i));
        const F w = 1 / F(i + 1);
        mean += tmp * w;
        wsum += w;
        tmpScores[i / tile.size()] = tmp;
    }

    {
        mean = cg::reduce(tile, mean, cg::plus<F>());
        wsum = cg::reduce(tile, wsum, cg::plus<F>());
    }

    mean /= wsum;

    for (uint32_t i = tile.thread_rank(); i < adjustedK; i += tile.size()) {
        const F tmp = tmpScores[i / tile.size()] - mean;
        const F w = 1 / F(i + 1);
        sd += tmp * tmp * w;
    }

    {
        sd = cg::reduce(tile, sd, cg::plus<F>());

        const auto lastScoreThreadIdx = (adjustedK - 1) % tile.size();
        const auto lastScoreIdx = (adjustedK - 1) / tile.size();
        lastScore = tile.shfl(tmpScores[lastScoreIdx], lastScoreThreadIdx);
    }

    sd = boost / sqrt(sd / wsum);
    const F nmax = EmbedSOMConstants<F>::maxAvoidance / lastScore;

    // convert the stuff to scores
    if (k < adjustedK)
        for (uint32_t i = tile.thread_rank(); i < k; i += tile.size()) {
            const auto scoreIdx = i / tile.size();
            const F score = exp((mean - tmpScores[scoreIdx]) * sd) *
                            (1 - exp(tmpScores[scoreIdx] * nmax -
                                     EmbedSOMConstants<F>::maxAvoidance));
            storage.storeScore(i, score);
        }
    else
        for (uint32_t i = tile.thread_rank(); i < k; i += tile.size())
            storage.storeScore(i,
                               exp((mean - tmpScores[i / tile.size()]) * sd));
}

/** Add single-point approximation to the matrix */
template<typename F>
__inline__ __device__ void
addGravity(const F score,
           const F *const __restrict__ grid2DPoint,
           F *const __restrict__ mtx)
{
    const F gs = score * EmbedSOMConstants<F>::gridGravity;

    mtx[0] += gs;
    mtx[3] += gs;
    mtx[4] += gs * grid2DPoint[0];
    mtx[5] += gs * grid2DPoint[1];
}

/** Add single-point approximation to the matrix, on 2-vectors */
template<typename F>
__inline__ __device__ void
addGravity2Wise(const F score,
                const F *const __restrict__ grid2DPoint,
                F *const __restrict__ mtx)
{
    const F gs = score * EmbedSOMConstants<F>::gridGravity;

    const typename Vec<2, F>::Type tmpGrid2d =
      reinterpret_cast<const typename Vec<2, F>::Type *>(grid2DPoint)[0];

    mtx[0] += gs;
    mtx[3] += gs;
    mtx[4] += gs * tmpGrid2d.x;
    mtx[5] += gs * tmpGrid2d.y;
}

/** Compute a projection of the point to a line defined by points I and J */
template<typename F>
__inline__ __device__ typename Vec<2, F>::Type
euclideanProjection(const F *const __restrict__ point,
                    const F *const __restrict__ gridPointI,
                    const F *const __restrict__ gridPointJ,
                    const uint32_t dim)
{
    typename Vec<2, F>::Type result{ 0.0, 0.0 };
    for (uint32_t k = 0; k < dim; ++k) {
        const F tmp = gridPointJ[k] - gridPointI[k];
        result.y += tmp * tmp;
        result.x += tmp * (point[k] - gridPointI[k]);
    }
    return result;
}

/** Run the projection on 4-vectors */
template<typename F>
__inline__ __device__ typename Vec<2, F>::Type
euclideanProjection4Wise(const F *const __restrict__ point,
                         const F *const __restrict__ gridPointI,
                         const F *const __restrict__ gridPointJ,
                         const uint32_t dim)
{
    const auto *const __restrict__ gridPointI4 =
      reinterpret_cast<const typename Vec<4, F>::Type *>(gridPointI);
    const auto *const __restrict__ gridPointJ4 =
      reinterpret_cast<const typename Vec<4, F>::Type *>(gridPointJ);
    const auto *const __restrict__ point4 =
      reinterpret_cast<const typename Vec<4, F>::Type *>(point);

    typename Vec<2, F>::Type result{ 0.0, 0.0 };

#define DOIT(X)                                                                \
    tmp = tmpGridJ.X - tmpGridI.X;                                             \
    result.y += tmp * tmp;                                                     \
    result.x += tmp * (tmpPoint.X - tmpGridI.X)

    for (uint32_t k = 0; k < dim / 4; ++k) {
        const auto tmpGridI = gridPointI4[k];
        const auto tmpGridJ = gridPointJ4[k];
        const auto tmpPoint = point4[k];

        F tmp;
        DOIT(x);
        DOIT(y);
        DOIT(z);
        DOIT(w);
    }
#undef DOIT

    for (uint32_t k = dim - (dim % 4); k < dim; ++k) {
        const F tmp = gridPointJ[k] - gridPointI[k];
        result.y += tmp * tmp;
        result.x += tmp * (point[k] - gridPointI[k]);
    }

    return result;
}

/** Add the result of projections to the approximation matrix */
template<typename F>
__inline__ __device__ void
addApproximation(const F scoreI,
                 const F scoreJ,
                 const F *const __restrict__ grid2DPointI,
                 const F *const __restrict__ grid2DPointJ,
                 const F adjust,
                 const F scalarProjection,
                 F *const __restrict__ mtx)
{
    F h[2], hp = 0;
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        h[i] = grid2DPointJ[i] - grid2DPointI[i];
        hp += h[i] * h[i];
    }

    if (hp < EmbedSOMConstants<F>::zeroAvoidance)
        return;

    const F exponent = scalarProjection - .5;
    const F s =
      scoreI * scoreJ * pow(1 + hp, adjust) * exp(-exponent * exponent);
    const F sihp = s / hp;
    const F rhsc = s * (scalarProjection +
                        (h[0] * grid2DPointI[0] + h[1] * grid2DPointI[1]) / hp);

    mtx[0] += h[0] * h[0] * sihp;
    mtx[1] += h[0] * h[1] * sihp;
    mtx[2] += h[1] * h[0] * sihp;
    mtx[3] += h[1] * h[1] * sihp;
    mtx[4] += h[0] * rhsc;
    mtx[5] += h[1] * rhsc;
}

/** Same as addApproximation(), on 2-vectors. */
template<typename F>
__inline__ __device__ void
addApproximation2Wise(const F scoreI,
                      const F scoreJ,
                      const F *const __restrict__ grid2DPointI,
                      const F *const __restrict__ grid2DPointJ,
                      const F adjust,
                      const F scalarProjection,
                      F *const __restrict__ mtx)
{
    const typename Vec<2, F>::Type tmpGrid2dI =
      reinterpret_cast<const typename Vec<2, F>::Type *>(grid2DPointI)[0];
    const typename Vec<2, F>::Type tmpGrid2dJ =
      reinterpret_cast<const typename Vec<2, F>::Type *>(grid2DPointJ)[0];

    const F h[2]{ tmpGrid2dJ.x - tmpGrid2dI.x, tmpGrid2dJ.y - tmpGrid2dI.y };
    const F hp = h[0] * h[0] + h[1] * h[1];

    if (hp < EmbedSOMConstants<F>::zeroAvoidance)
        return;

    const F exponent = scalarProjection - .5;
    const F s =
      scoreI * scoreJ * pow(1 + hp, adjust) * exp(-exponent * exponent);
    const F sihp = s / hp;
    const F rhsc =
      s * (scalarProjection + (h[0] * tmpGrid2dI.x + h[1] * tmpGrid2dI.y) / hp);

    mtx[0] += h[0] * h[0] * sihp;
    mtx[1] += h[0] * h[1] * sihp;
    mtx[2] += h[1] * h[0] * sihp;
    mtx[3] += h[1] * h[1] * sihp;
    mtx[4] += h[0] * rhsc;
    mtx[5] += h[1] * rhsc;
}

/** Type tag for indexing by small rectangles.
 *
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
struct RectangleIndexer
{};

/** Function template for getting indexes */
template<typename INDEXER>
__inline__ __device__ uint2
getIndices(uint32_t plainIndex, uint32_t k)
{
}

/** Specialization of getIndices for the rectangle indexing. */
template<>
__inline__ __device__ uint2
getIndices<RectangleIndexer>(uint32_t plainIndex, uint32_t k)
{
    uint2 indices;
    const uint32_t tempI = plainIndex / k;
    const uint32_t tempJ = plainIndex % k;
    const auto invertedI = k - 1 - tempI;
    indices.x = tempJ < invertedI ? tempI : invertedI - 1;
    indices.y = (tempJ < invertedI ? tempJ : tempJ - invertedI) + indices.x + 1;
    return indices;
}

/** Base projection kernel (each thread does everything for a single point). */
template<typename F>
__global__ void
projectionBaseKernel(const F *__restrict__ points,
                     const F *const __restrict__ grid,
                     const F *const __restrict__ grid2d,
                     knn_entry<F> *__restrict__ neighbors,
                     F *__restrict__ projections,
                     const uint32_t dim,
                     const uint32_t n,
                     const uint32_t gridSize,
                     const uint32_t k,
                     const F adjust,
                     const F boost)
{
    // assign defaults and generate scores
    {
        const uint32_t adjustedK = k < gridSize ? k + 1 : k;
        const uint32_t pointIdx = blockIdx.x * blockDim.x + threadIdx.x;
        if (pointIdx >= n)
            return;
        points = points + pointIdx * dim;
        neighbors = neighbors + pointIdx * adjustedK;
        projections = projections + pointIdx * 2;
        sortedDistsToScores<F>(neighbors, adjustedK, k, boost);
    }

    F mtx[6];
    memset(mtx, 0, 6 * sizeof(F));
    for (uint32_t i = 0; i < k; ++i) {
        const uint32_t idxI = neighbors[i].index;
        const F scoreI = neighbors[i].distance;
        addGravity(scoreI, grid2d + idxI * 2, mtx);
        for (uint32_t j = i + 1; j < k; ++j) {
            const uint32_t idxJ = neighbors[j].index;
            const F scoreJ = neighbors[j].distance;
            const auto result = euclideanProjection<F>(
              points, grid + idxI * dim, grid + idxJ * dim, dim);
            F scalarProjection = result.x;
            const F squaredGridPointsDistance = result.y;
            if (squaredGridPointsDistance == F(0))
                continue;
            scalarProjection /= squaredGridPointsDistance;
            addApproximation(scoreI,
                             scoreJ,
                             grid2d + idxI * 2,
                             grid2d + idxJ * 2,
                             adjust,
                             scalarProjection,
                             mtx);
        }
    }
    // solve linear equation
    const F det = mtx[0] * mtx[3] - mtx[1] * mtx[2];
    projections[0] = (mtx[4] * mtx[3] - mtx[5] * mtx[2]) / det;
    projections[1] = (mtx[0] * mtx[5] - mtx[1] * mtx[4]) / det;
}

/** Aligned-shared-memory projection kernel.
 *
 * One block computes embedding for one point, using CUB block reduce for
 * matrix reduction.
 */
template<typename F, typename INDEXER, size_t tileSize>
__global__ void
projectionAlignedShMemoryKernel(const F *__restrict__ points,
                                const F *const __restrict__ grid,
                                const F *const __restrict__ grid2d,
                                knn_entry<F> *__restrict__ neighbors,
                                F *__restrict__ projections,
                                const uint32_t dim,
                                const uint32_t n,
                                const uint32_t gridSize,
                                const uint32_t k,
                                const F adjust,
                                const F boost,
                                const uint32_t groupSize,
                                const uint32_t cacheLeadingDim)
{
    extern __shared__ char sharedMemory[];

    const uint32_t groupRank = threadIdx.x % groupSize;
    const uint32_t groupIdx = threadIdx.x / groupSize;
    const uint32_t groupsCount = blockDim.x / groupSize;

    const auto grid2dPadding =
      (k * 3) % cacheLeadingDim == 0
        ? 0
        : cacheLeadingDim - ((k * 3) % cacheLeadingDim);
    auto sharedMemoryoff =
      reinterpret_cast<F *>(sharedMemory) +
      ((k + 1) * cacheLeadingDim + k * 3 + grid2dPadding) * groupIdx;

    F *const __restrict__ pointCache = sharedMemoryoff;
    F *const __restrict__ gridCache = sharedMemoryoff + cacheLeadingDim;
    F *const __restrict__ grid2dCache =
      sharedMemoryoff + (k + 1) * cacheLeadingDim;
    F *const __restrict__ scoreCache = grid2dCache + k * 2;

    F *const __restrict__ reduceFinishStorage =
      reinterpret_cast<F *>(sharedMemory) +
      ((k + 1) * cacheLeadingDim + k * 3 + grid2dPadding) * groupsCount;

    auto tile = cg::tiled_partition<tileSize>(cg::this_thread_block());

    // assign defaults and generate scores
    {
        const uint32_t adjustedK = k < gridSize ? k + 1 : k;

        const auto workIdx = blockIdx.x * groupsCount + groupIdx;

        if (workIdx >= n)
            return;

        points = points + workIdx * dim;
        neighbors = neighbors + workIdx * adjustedK;
        projections = projections + workIdx * 2;

        if (groupRank < tile.size())
            sortedDistsToScoresGroup<F>(
              tile,
              NeighborScoreStorage<F>{ neighbors, scoreCache },
              adjustedK,
              k,
              boost);
        else
            storeToCache(groupRank - tile.size(),
                         groupSize - tile.size(),
                         points,
                         pointCache,
                         grid,
                         gridCache,
                         grid2d,
                         grid2dCache,
                         neighbors,
                         dim,
                         cacheLeadingDim,
                         k);

        if (groupSize == tile.size())
            storeToCache(groupRank,
                         groupSize,
                         points,
                         pointCache,
                         grid,
                         gridCache,
                         grid2d,
                         grid2dCache,
                         neighbors,
                         dim,
                         cacheLeadingDim,
                         k);

        __syncthreads();
    }

    F mtx[6];
    memset(mtx, 0, 6 * sizeof(F));

    for (uint32_t i = groupRank; i < k; i += groupSize)
        addGravity2Wise(scoreCache[i], grid2dCache + i * 2, mtx);

    const uint32_t neighborPairs = (k * (k - 1)) / 2;
    for (uint32_t i = groupRank; i < neighborPairs; i += groupSize) {
        const auto indices = getIndices<INDEXER>(i, k);

        const auto I = indices.x;
        const auto J = indices.y;

        const auto result =
          euclideanProjection4Wise<F>(pointCache,
                                      gridCache + I * cacheLeadingDim,
                                      gridCache + J * cacheLeadingDim,
                                      dim);
        F scalarProjection = result.x;
        const F squaredGridPointsDistance = result.y;

        if (squaredGridPointsDistance == F(0))
            continue;

        scalarProjection /= squaredGridPointsDistance;

        addApproximation2Wise(scoreCache[I],
                              scoreCache[J],
                              grid2dCache + I * 2,
                              grid2dCache + J * 2,
                              adjust,
                              scalarProjection,
                              mtx);
    }

#pragma unroll
    for (size_t i = 0; i < 6; ++i) {
        mtx[i] = cg::reduce(tile, mtx[i], cg::plus<F>());

        const auto warpId = threadIdx.x / warpSize;

        if (threadIdx.x % warpSize == 0 && groupRank != 0)
            reduceFinishStorage[warpId] = mtx[i];

        __syncthreads();

        if (groupRank == 0) {
            for (uint32_t j = 1; j < groupSize / warpSize; ++j) {
                mtx[i] += reduceFinishStorage[warpId + j];
            }
        }
    }

    if (groupRank == 0) {
        const F det = mtx[0] * mtx[3] - mtx[1] * mtx[2];
        projections[0] = (mtx[4] * mtx[3] - mtx[5] * mtx[2]) / det;
        projections[1] = (mtx[0] * mtx[5] - mtx[1] * mtx[4]) / det;
    }
}

// TODO this might be used as a backup solution for extreme params
#if 0
void
EsomCuda::runProjectionBaseKernel(float boost, float adjust)
{
    unsigned int blockSize = 256;
    unsigned int blockCount = (mPointsCount + blockSize - 1) / blockSize;
    projectionBaseKernel<float> << <blockCount, blockSize >> > (
        mCuPoints,
        mCuLandmarksHighDim,
        mCuLandmarksLowDim,
        reinterpret_cast<::knn_entry<float> *>(mCuknn_entry),
        mCuEmbedding,
        mDim,
        mPointsCount,
        mLandmarksCount,
        mTopK,
        adjust,
        boost);

    CUCH(cudaGetLastError());
}
#endif

void
EmbedSOMCUDAContext::runProjectionKernel(size_t d,
                                         size_t n,
                                         size_t g,
                                         size_t k,
                                         float boost,
                                         float adjust)
{
    constexpr size_t alignment = 16;
    auto pointBytes = d * sizeof(float);
    auto dimCacheResidual = pointBytes % alignment;
    auto gridCacheLeadingDim =
      pointBytes + (dimCacheResidual == 0 ? 0 : alignment - dimCacheResidual);
    gridCacheLeadingDim /= sizeof(float);

    auto blockSize = 128;
    auto groupSize = 32;
    auto groupsPerBlock = blockSize / groupSize;
    unsigned int blockCount = (n + groupsPerBlock - 1) / groupsPerBlock;
    auto warpCount = blockSize / 32;
    auto grid2dPadding =
      (k * 3) % gridCacheLeadingDim == 0
        ? 0
        : gridCacheLeadingDim - ((k * 3) % gridCacheLeadingDim);
    auto sharedMem =
      sizeof(float) * warpCount +
      sizeof(float) * (k + 1) * gridCacheLeadingDim * groupsPerBlock +
      sizeof(float) * (k * 3 + grid2dPadding) * groupsPerBlock;

    projectionAlignedShMemoryKernel<float, RectangleIndexer, 32>
      <<<blockCount, blockSize, sharedMem>>>(data,
                                             lm_hi,
                                             lm_lo,
                                             knns,
                                             points,
                                             d,
                                             n,
                                             g,
                                             k,
                                             adjust,
                                             boost,
                                             groupSize,
                                             gridCacheLeadingDim);

    CUCH(cudaGetLastError());
}
