#ifndef NO_CUDA

#include "embedsom_cuda.h"

#include "cuda_runtime.h"

#include <chrono>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

/*
 * Private
 */

void
EsomCuda::dimCheck()
{
    if (mDim < 2 || mDim > 256) {
        throw std::runtime_error(
          "Dimension sanity check failed. CUDA embedsom can work with dataset "
          "with dimension in 2-256 range.");
    }
}

void
EsomCuda::preflightCheck()
{
    dimCheck();

    // validate inputs
    if (!mPointsValid) {
        throw std::runtime_error("Dataset points must be loaded before "
                                 "embedding computation can be started.");
    }
    if (!mLandmarksHighDimValid) {
        throw std::runtime_error(
          "Landmark coordinates in dateset space must be set before embedding "
          "computation can be started.");
    }
    if (!mLandmarksLowDimValid) {
        throw std::runtime_error(
          "Landmark coordinates in target embedding space must be set before "
          "embedding computation can be started.");
    }

    if (mTopK < 1 && mTopK > std::min<std::size_t>(256, mLandmarksCount)) {
        throw std::runtime_error(
          "The k parameter (for top-k selection) is out of range.");
    }

    mAdjustedTopK = mTopK < mLandmarksCount ? mTopK + 1 : mTopK;

    // allocate missing buffers if necessary
    std::size_t topkResultSize =
      mAdjustedTopK * mPointsCount * sizeof(EsomCuda::TopkResult);
    if (mCuTopkResult == nullptr || mAllocatedTopk != topkResultSize) {
        if (mCuTopkResult != nullptr) {
            CUCH(cudaFree(mCuTopkResult));
            mCuTopkResult = nullptr;
        }

        CUCH(cudaMalloc(&mCuTopkResult, topkResultSize));
        mAllocatedTopk = topkResultSize;
    }

    std::size_t embeddingSize = 2 * mPointsCount * sizeof(float);
    if (mCuEmbedding == nullptr || mAllocatedEmbedding != embeddingSize) {
        if (mCuEmbedding != nullptr) {
            CUCH(cudaFree(mCuEmbedding));
            mCuEmbedding = nullptr;
        }

        CUCH(cudaMalloc(&mCuEmbedding, embeddingSize));
        mAllocatedEmbedding = embeddingSize;
    }
}

/*
 * Public
 */

EsomCuda::EsomCuda()
  : mPointsCount(0)
  , mLandmarksCount(0)
  , mDim(0)
  , mTopK(0)
  , mAllocatedTopk(0)
  , mAllocatedEmbedding(0)
  , mCuPoints(nullptr)
  , mCuLandmarksHighDim(nullptr)
  , mCuLandmarksLowDim(nullptr)
  , mCuEmbedding(nullptr)
  , mCuTopkResult(nullptr)
  , mPointsValid(false)
  , mLandmarksHighDimValid(false)
  , mLandmarksLowDimValid(false)
{
    CUCH(cudaSetDevice(0));
}

void
EsomCuda::setDim(std::size_t dim)
{
    if (mDim == dim)
        return;
    mDim = dim;

    if (mCuPoints != nullptr) {
        CUCH(cudaFree(mCuPoints));
        mCuPoints = nullptr;
        mPointsValid = false;
    }
    if (mCuLandmarksHighDim != nullptr) {
        CUCH(cudaFree(mCuLandmarksHighDim));
        mCuLandmarksHighDim = nullptr;
        mLandmarksHighDimValid = false;
    }
}

void
EsomCuda::setK(std::size_t k)
{
    mTopK = k;
}

void
EsomCuda::setPoints(std::size_t pointsCount, const float *pointsData)
{
    CUCH(cudaSetDevice(0));
    dimCheck();

    // realocate buffers if necessary
    if (pointsCount != mPointsCount || mCuPoints == nullptr) {
        mPointsValid = false;
        if (mCuPoints != nullptr) {
            CUCH(cudaFree(mCuPoints));
            mCuPoints = nullptr;
        }
        if (pointsCount > 0) {
            CUCH(cudaMalloc(&mCuPoints, pointsCount * mDim * sizeof(float)));
        }
    }

    mPointsCount = pointsCount;

    // copy data to GPU if possible
    if (mPointsCount > 0 && pointsData != nullptr) {
        auto startTs = std::chrono::high_resolution_clock::now();

        CUCH(cudaMemcpy(mCuPoints,
                        pointsData,
                        mPointsCount * mDim * sizeof(float),
                        cudaMemcpyHostToDevice));
        mPointsValid = true;

        auto endTs = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> duration = endTs - startTs;
        while (mPointsUploadTimes.size() >= timeMeasurements)
            mPointsUploadTimes.pop_front();
        mPointsUploadTimes.push_back(duration.count());
    }
}

void
EsomCuda::setLandmarks(std::size_t landmarksCount,
                       const float *highDim,
                       const float *lowDim)
{
    CUCH(cudaSetDevice(0));
    dimCheck();

    // realocate buffers if necessary
    if (landmarksCount != mLandmarksCount || mCuLandmarksHighDim == nullptr) {
        mLandmarksHighDimValid = false;
        if (mCuLandmarksHighDim != nullptr) {
            CUCH(cudaFree(mCuLandmarksHighDim));
            mCuLandmarksHighDim = nullptr;
        }
        if (landmarksCount > 0) {
            CUCH(cudaMalloc(&mCuLandmarksHighDim,
                            landmarksCount * mDim * sizeof(float)));
        }
    }

    if (landmarksCount != mLandmarksCount || mCuLandmarksLowDim == nullptr) {
        mLandmarksLowDimValid = false;
        if (mCuLandmarksLowDim != nullptr) {
            CUCH(cudaFree(mCuLandmarksLowDim));
            mCuLandmarksLowDim = nullptr;
        }
        if (landmarksCount > 0) {
            CUCH(cudaMalloc(&mCuLandmarksLowDim,
                            landmarksCount * 2 * sizeof(float)));
        }
    }

    mLandmarksCount = landmarksCount;
    if (mLandmarksCount == 0)
        return;

    // copy data to GPU if possible
    if (highDim != nullptr) {
        auto startTs = std::chrono::high_resolution_clock::now();

        CUCH(cudaMemcpy(mCuLandmarksHighDim,
                        highDim,
                        mLandmarksCount * mDim * sizeof(float),
                        cudaMemcpyHostToDevice));
        mLandmarksHighDimValid = true;

        auto endTs = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> duration = endTs - startTs;
        while (mLandmarksUploadTimes.size() >= timeMeasurements)
            mLandmarksUploadTimes.pop_front();
        mLandmarksUploadTimes.push_back(duration.count());
    }

    if (lowDim != nullptr) {
        CUCH(cudaMemcpy(mCuLandmarksLowDim,
                        lowDim,
                        mLandmarksCount * 2 * sizeof(float),
                        cudaMemcpyHostToDevice));
        mLandmarksLowDimValid = true;
    }
}

void
EsomCuda::embedsom(float boost, float adjust, float *embedding)
{
    CUCH(cudaSetDevice(0));

    preflightCheck();

    auto startTs = std::chrono::high_resolution_clock::now();

    runTopkBitonicOptKernel();
    runProjectionKernel(boost, adjust);

    // this is blocking operation so it also provides host sync with GPU
    CUCH(cudaMemcpy(embedding,
                    mCuEmbedding,
                    mPointsCount * 2 * sizeof(float),
                    cudaMemcpyDeviceToHost));

    auto endTs = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = endTs - startTs;
    while (mProcessingTimes.size() >= timeMeasurements)
        mProcessingTimes.pop_front();
    mProcessingTimes.push_back(duration.count());
}

float
EsomCuda::getAvgPointsUploadTime() const
{
    float sum = 0.0f;
    for (auto &t : mPointsUploadTimes)
        sum += t;
    if (!mPointsUploadTimes.empty())
        sum /= (float)mPointsUploadTimes.size();
    return sum;
}

float
EsomCuda::getAvgLandmarksUploadTime() const
{
    float sum = 0.0f;
    for (auto &t : mLandmarksUploadTimes)
        sum += t;
    if (!mLandmarksUploadTimes.empty())
        sum /= (float)mLandmarksUploadTimes.size();
    return sum;
}

float
EsomCuda::getAvgProcessingTime() const
{
    float sum = 0.0f;
    for (auto &t : mProcessingTimes)
        sum += t;
    if (!mProcessingTimes.empty())
        sum /= (float)mProcessingTimes.size();
    return sum;
}

#endif
