
#ifndef KMEANS_LANDMARK_H
#define KMEANS_LANDMARK_H

#include <Magnum/Magnum.h>

#include <random>
#include <vector>

#include "data_model.h"

struct KMeansData
{
    // kept allocated for efficiency
    std::vector<size_t> assignments;
    std::default_random_engine gen;
};

void
kmeans_landmark_step(KMeansData &data,
                     const DataModel &model,
                     size_t n_means,
                     size_t d,
                     size_t iters,
                     float alpha,
                     float neighbor_alpha,
                     const std::vector<std::pair<size_t, size_t>> &neighbors,
                     std::vector<float> &means);

#endif
