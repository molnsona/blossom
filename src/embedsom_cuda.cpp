#include "embedsom_cuda.h"

#include "cuda_runtime.h"

#include <exception>
#include <string>
#include <sstream>


/*
 * Private
 */

void EsomCuda::dimCheck()
{
	if (mDim < 2 || mDim > 256) {
		throw std::runtime_error("Dimension sanity check failed. CUDA embedsom can work with dataset with dimension in 2-256 range.");
	}
}

void EsomCuda::preflightCheck()
{
	dimCheck();

	// validate inputs
	if (!mPointsValid) {
		throw std::runtime_error("Dataset points must be loaded before embedding computation can be started.");
	}
	if (!mLandmarksHighDimValid) {
		throw std::runtime_error("Landmark coordinates in dateset space must be set before embedding computation can be started.");
	}
	if (!mLandmarksLowDimValid) {
		throw std::runtime_error("Landmark coordinates in target embedding space must be set before embedding computation can be started.");
	}

	if (mTopK < 1 && mTopK > std::min<std::size_t>(256, mLandmarksCount)) {
		throw std::runtime_error("The k parameter (for top-k selection) is out of range.");
	}

	// allocate missing buffers if necessary
	if (mTopkResults == nullptr) {
		CUCH(cudaMalloc(&mTopkResults, mTopK * mPointsCount * sizeof(EsomCuda::TopkResult)));
	}

	if (mCuEmbedding == nullptr) {
		CUCH(cudaMalloc(&mCuEmbedding, 2 * mPointsCount * sizeof(float)));
	}
}


/*
 * Public
 */

EsomCuda::EsomCuda()
	: mPointsCount(0), mLandmarksCount(0), mDim(0), mTopK(0),
	mCuPoints(nullptr), mCuLandmarksHighDim(nullptr), mCuLandmarksLowDim(nullptr), mCuEmbedding(nullptr), mTopkResults(nullptr),
	mPointsValid(false), mLandmarksHighDimValid(false), mLandmarksLowDimValid(false)
{}

void EsomCuda::setDim(std::size_t dim)
{
	if (mDim == dim) return;
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

void EsomCuda::setK(std::size_t k)
{
	if (mTopK == k) return;
	mTopK = k;

	if (mTopkResults != nullptr) {
		CUCH(cudaFree(mTopkResults));
		mTopkResults = nullptr;
	}
}

void EsomCuda::setPoints(std::size_t pointsCount, const float *pointsData)
{
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

	if (pointsCount != mPointsCount) {
		if (mCuEmbedding != nullptr) {
			CUCH(cudaFree(mCuEmbedding));
			mCuEmbedding = nullptr;
		}
		if (mTopkResults != nullptr) {
			CUCH(cudaFree(mTopkResults));
			mTopkResults = nullptr;
		}
	}

	mPointsCount = pointsCount;

	// copy data to GPU if possible
	if (mPointsCount > 0 && pointsData != nullptr) {
		CUCH(cudaMemcpy(mCuPoints, pointsData, mPointsCount * mDim * sizeof(float), cudaMemcpyHostToDevice));
		mPointsValid = true;
	}
}

void EsomCuda::setLandmarks(std::size_t landmarksCount, const float *highDim, const float *lowDim)
{
	dimCheck();

	// realocate buffers if necessary
	if (landmarksCount != mLandmarksCount || mCuLandmarksHighDim == nullptr) {
		mLandmarksHighDimValid = false;
		if (mCuLandmarksHighDim != nullptr) {
			CUCH(cudaFree(mCuLandmarksHighDim));
			mCuLandmarksHighDim = nullptr;
		}
		if (landmarksCount > 0) {
			CUCH(cudaMalloc(&mCuLandmarksHighDim, landmarksCount * mDim * sizeof(float)));
		}
	}

	if (landmarksCount != mLandmarksCount || mCuLandmarksLowDim == nullptr) {
		mLandmarksLowDimValid = false;
		if (mCuLandmarksLowDim != nullptr) {
			CUCH(cudaFree(mCuLandmarksLowDim));
			mCuLandmarksLowDim = nullptr;
		}
		if (landmarksCount > 0) {
			CUCH(cudaMalloc(&mCuLandmarksLowDim, landmarksCount * 2 * sizeof(float)));
		}
	}

	mLandmarksCount = landmarksCount;
	if (mLandmarksCount == 0) return;

	// copy data to GPU if possible
	if (highDim != nullptr) {
		CUCH(cudaMemcpy(mCuLandmarksHighDim, highDim, mLandmarksCount * mDim * sizeof(float), cudaMemcpyHostToDevice));
		mLandmarksHighDimValid = true;
	}

	if (lowDim != nullptr) {
		CUCH(cudaMemcpy(mCuLandmarksLowDim, lowDim, mLandmarksCount * 2 * sizeof(float), cudaMemcpyHostToDevice));
		mLandmarksLowDimValid = true;
	}
}

void EsomCuda::embedsom(float boost, float adjust, float *embedding)
{
	preflightCheck();

	CUCH(cudaSetDevice(0));
	runTopkBaseKernel();

	// TODO run projection kernel

	// this is blocking operation so it also provides host sync with GPU
	CUCH(cudaMemcpy(embedding, mCuEmbedding, mPointsCount * 2 * sizeof(float), cudaMemcpyDeviceToHost));
}
