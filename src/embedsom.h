
#ifndef EMBEDSOM_H
#define EMBEDSOM_H

#include <cstddef>

void
embedsom(const size_t n,
         const size_t n_landmarks,
         const size_t dim,
         const float boost,
         const size_t topn,
         const float adjust,
         const float *points,
         const float *hidim_lm,
         const float *lodim_lm,
         float *embedding);

#endif
