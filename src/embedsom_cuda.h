/*
The MIT License

Copyright (c) 2021 Martin Krulis
                   Mirek Kratochvil
                   Sona Molnarova

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

#ifndef EMBEDSOM_CUDA_H
#define EMBEDSOM_CUDA_H

#include "cuda_runtime.h"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <exception>
#include <sstream>
#include <string>

#include "cuda_structs.cuh"

/** Data context wrapper for CUDA EmbedSOM
 *
 * A compound "context" object for the EmbedSOM computation in CUDA, mainly
 * holding some required preallocated memory buffers.
 */
struct EmbedSOMCUDAContext
{
    size_t ndata, nlm_hi, nlm_lo, npoints, nknns;

    float *data, *lm_hi, *lm_lo, *points;
    knn_entry<float> *knns;

    EmbedSOMCUDAContext()
      : ndata(0)
      , nlm_hi(0)
      , nlm_lo(0)
      , npoints(0)
      , nknns(0)
      , data(nullptr)
      , lm_hi(nullptr)
      , lm_lo(nullptr)
      , points(nullptr)
      , knns(nullptr)
    {}

    ~EmbedSOMCUDAContext();

    /** Run EmbedSOM on the given data. */
    void run(size_t n,
             size_t g,
             size_t d,
             float boost,
             size_t k,
             float adjust,
             const float *hidim_points,
             const float *hidim_landmarks,
             const float *lodim_landmarks,
             float *lodim_points);

private:
    /** Execute the CUDA kNN within the context */
    void runKNNKernel(size_t d, size_t n, size_t g, size_t adjusted_k);

    /** Execute the CUDA projection kernel atop the context */
    void runProjectionKernel(size_t d,
                             size_t n,
                             size_t g,
                             size_t k,
                             float boost,
                             float adjust);
};

/** Helper exception for throwing sensible CUDA errors. */
struct CudaError : std::exception
{
    std::string msg;    /// message
    cudaError_t status; /// reported cuda status, may be examined separately

    CudaError(cudaError_t status = cudaSuccess)
      : status(status)
    {}
    CudaError(const char *msg, cudaError_t status = cudaSuccess)
      : msg(msg)
      , status(status)
    {}
    CudaError(const std::string &msg, cudaError_t status = cudaSuccess)
      : msg(msg)
      , status(status)
    {}
    virtual ~CudaError() noexcept {}

    virtual const char *what() const throw() { return msg.c_str(); }
};

/** CUDA error code check. This is internal function used by CUCH macro. */
inline void
_cuda_check(cudaError_t status,
            int line,
            const char *srcFile,
            const char *errMsg = nullptr)
{
    if (status != cudaSuccess) {
        std::stringstream str_s;
        str_s << "CUDA Error (" << status << "): " << cudaGetErrorString(status)
              << "\n"
              << "at " << srcFile << "[" << line << "]: " << errMsg;

        throw(CudaError(str_s.str(), status));
    }
}

/** Macro wrapper for CUDA calls checking. */
#define CUCH(status) _cuda_check(status, __LINE__, __FILE__, #status)

#endif // EMBEDSOM_CUDA_H
