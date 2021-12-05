
#include "embedsom_cuda.h"

#include "cuda_runtime.h"

void
EmbedSOMCUDAContext::run(size_t n,
                         size_t g,
                         size_t d,
                         float boost,
                         size_t k,
                         float adjust,
                         const float *hidim_points,
                         const float *hidim_landmarks,
                         const float *lodim_landmarks,
                         float *lodim_points)
{

    /* first make sure all sizes are acceptable and there's actual sensible
     * work that can be done */
    if (!n)
        return;
    if (d < 2 || d > 256)
        return; // TODO try d=1; d>256 should warn
    if (k > g)
        k = g;
    if (k < 3)
        return;

    /* make sure the buffers are sufficiently big. The allocations are kept as
     * such: |data| = d*n |points| = 2*n |lm_hi| = d*g |lm_lo| = 2*g |topk| =
     * adjusted_k()*n
     *
     * Because `n` and `g` is highly variable in this usecase, we _only_
     * _increase_ the buffer sizes. Reallocation in case the sizes grow totally
     * out of limits would help too (feel free to add into the conditions).
     */

    CUCH(cudaSetDevice(0));

#define size_check(tgt, cur, buffer, type)                                     \
    if (tgt > cur) {                                                           \
        if (buffer)                                                            \
            cudaFree(buffer);                                                  \
        CUCH(cudaMalloc(&buffer, tgt * sizeof(type)));                         \
        cur = tgt;                                                             \
    }

    size_check(d * n, ndata, data, float);
    size_check(2 * n, npoints, points, float);
    size_check(d * g, nlm_hi, lm_hi, float);
    size_check(2 * g, nlm_lo, lm_lo, float);

    size_t adjusted_k = std::min(g, k + 1);
    size_check(adjusted_k * n, nknns, knns, knn_entry<float>);
    CUCH(cudaGetLastError());
#undef size_check

    /* all is safely allocated, transfer the data! */
    CUCH(cudaMemcpy(
      data, hidim_points, d * n * sizeof(float), cudaMemcpyHostToDevice));
    CUCH(cudaMemcpy(
      lm_hi, hidim_landmarks, d * g * sizeof(float), cudaMemcpyHostToDevice));
    CUCH(cudaMemcpy(
      lm_lo, lodim_landmarks, 2 * g * sizeof(float), cudaMemcpyHostToDevice));
    CUCH(cudaGetLastError());

    /* run the main embedding */
    runKNNKernel(d, n, g, adjusted_k);
    runProjectionKernel(d, n, g, k, boost, adjust);

    /* get the results */
    CUCH(cudaMemcpy(
      lodim_points, points, 2 * n * sizeof(float), cudaMemcpyDeviceToHost));
    CUCH(cudaGetLastError());
}

EmbedSOMCUDAContext::~EmbedSOMCUDAContext()
{
    if (data)
        cudaFree(data);
    if (points)
        cudaFree(points);
    if (lm_hi)
        cudaFree(lm_hi);
    if (lm_lo)
        cudaFree(lm_lo);
    if (knns)
        cudaFree(knns);
}
