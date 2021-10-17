
#ifndef EMBEDSOM_CUDA_H
#define EMBEDSOM_CUDA_H
#ifndef NO_CUDA

#include "cuda_runtime.h"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <exception>
#include <sstream>
#include <string>

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

/**
 * Wrapper for embedsom computed on CUDA. It holds intermediate context, such as
 * preallocated memory buffers.
 */
class EsomCuda
{
public:
    /**
     * Internal structure used for intermediate topk results
     */
    struct TopkResult
    {
        float distance;
        std::uint32_t index;

        CUDA_CALLABLE_MEMBER bool operator<(const TopkResult &rhs) const
        {
            return this->distance < rhs.distance ||
                   (this->distance == rhs.distance && this->index < rhs.index);
        }

        CUDA_CALLABLE_MEMBER bool operator>(const TopkResult &rhs) const
        {
            return rhs < *this;
        }

        CUDA_CALLABLE_MEMBER bool operator<=(const TopkResult &rhs) const
        {
            return !(*this > rhs);
        }

        CUDA_CALLABLE_MEMBER bool operator>=(const TopkResult &rhs) const
        {
            return !(*this < rhs);
        }
    };

private:
    // configuration
    std::size_t mPointsCount;
    std::size_t mLandmarksCount;
    std::size_t mDim;
    std::size_t mTopK;
    std::size_t mAdjustedTopK; // topk + 1 (if topk < gridsize)
    std::size_t mAllocatedTopk;
    std::size_t mAllocatedEmbedding;

    // buffers
    float *mCuPoints;
    float *mCuLandmarksHighDim;
    float *mCuLandmarksLowDim;
    float *mCuEmbedding;
    TopkResult *mCuTopkResult;

    // flags that indicate whether buffers hold up-to-date data
    bool mPointsValid;
    bool mLandmarksHighDimValid;
    bool mLandmarksLowDimValid;

    // some stats
    const std::size_t timeMeasurements = 10;
    std::deque<float> mPointsUploadTimes;
    std::deque<float> mLandmarksUploadTimes;
    std::deque<float> mProcessingTimes;

    void dimCheck();
    void preflightCheck();

    // kernel runners (implemented separately in .cu files)
    void runTopkBaseKernel();
    void runTopkBitonicOptKernel();
    void runProjectionBaseKernel(float boost, float adjust);
    void runProjectionKernel(float boost, float adjust);

public:
    EsomCuda();

    /**
     * Change the dimensionality of the dataset.
     * Invalidates all internal buffers except for low dimensional points (which
     * are always in 2D).
     */
    void setDim(std::size_t dim);

    /**
     * Change the number of nearest landmarks used for each data point (i.e.,
     * for internal top-k search). Invalidates internal topk result buffer
     * (needs to be reallocated in next embedsom call).
     */
    void setK(std::size_t k);

    /**
     * Sets the points dataset.
     * @param pointsCount nummber of points in the dataset (n)
     * @param pointsData If not null, the buffer must contain dim * n floats in
     * AoS format. If null, it preallocates necessary buffers but does not fill
     * them with data.
     */
    void setPoints(std::size_t pointsCount, const float *pointsData = nullptr);

    /**
     * Sets the landmark datasets.
     * @param landmarksCount number of landmarks in the grid
     * @param highDim high dimensional coordinates of the landmarks (dim *
     * landmarksCount in AoS)
     * @param lowDim low dimensional (2D) coordinates of the landmarks (2 *
     * landmarksCount in AoS) If any of the pointers are null, no data are
     * copied (but the buffers are preallocated).
     */
    void setLandmarks(std::size_t landmarksCount,
                      const float *highDim = nullptr,
                      const float *lowDim = nullptr);

    /**
     * Compute the embedding. All input buffers (points, landmarks) must be
     * filled before this method is called.
     * @param boost
     * @param adjust
     * @param embedding a buffer for the result
     */
    void embedsom(float boost, float adjust, float *embedding);

    // Constant accessors
    float getAvgPointsUploadTime() const;
    float getAvgLandmarksUploadTime() const;
    float getAvgProcessingTime() const;
};

/**
 * Exception thrown if something goes wrong with CUDA.
 */
class CudaError : public std::exception
{
protected:
    std::string mMessage; ///< Internal buffer where the message is kept.
    cudaError_t mStatus;

public:
    CudaError(cudaError_t status = cudaSuccess)
      : std::exception()
      , mMessage()
      , mStatus(status)
    {}
    CudaError(const char *msg, cudaError_t status = cudaSuccess)
      : std::exception()
      , mMessage(msg)
      , mStatus(status)
    {}
    CudaError(const std::string &msg, cudaError_t status = cudaSuccess)
      : std::exception()
      , mMessage(msg)
      , mStatus(status)
    {}
    virtual ~CudaError() noexcept {}

    virtual const char *what() const throw() { return mMessage.c_str(); }

    // Overloading << operator that uses stringstream to append data to
    // mMessage.
    template<typename T>
    CudaError &operator<<(const T &data)
    {
        std::stringstream stream;
        stream << mMessage << data;
        mMessage = stream.str();
        return *this;
    }
};

/**
 * CUDA error code check. This is internal function used by CUCH macro.
 */
inline void
_cuda_check(cudaError_t status,
            int line,
            const char *srcFile,
            const char *errMsg = nullptr)
{
    if (status != cudaSuccess) {
        throw(CudaError(status)
              << "CUDA Error (" << status << "): " << cudaGetErrorString(status)
              << "\n"
              << "at " << srcFile << "[" << line << "]: " << errMsg);
    }
}

/**
 * Macro wrapper for CUDA calls checking.
 */
#define CUCH(status) _cuda_check(status, __LINE__, __FILE__, #status)

#endif
#endif