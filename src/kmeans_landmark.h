
#ifndef KMEANS_LANDMARK_H
#define KMEANS_LANDMARK_H

#include <Magnum/Magnum.h>
#include <Magnum/Math/Vector2.h>

#include <random>
#include <vector>

#include "trans_data.h"

struct KMeansData
{
    std::default_random_engine gen;
};

void
kmeans_landmark_step(KMeansData &data,
                     const TransData &model,
                     size_t n_means,
                     size_t d,
                     size_t iters,
                     float alpha,
                     float neighbor_alpha,
                     const std::vector<std::pair<size_t, size_t>> &neighbors,
                     std::vector<float> &means);

void
som_landmark_step(KMeansData &data,
                  const TransData &model,
                  size_t n_means,
                  size_t d,
                  size_t iters,
                  float alpha,
                  float sigma,
                  std::vector<float> &neurons,
                  const std::vector<Magnum::Vector2> &map);

#endif
