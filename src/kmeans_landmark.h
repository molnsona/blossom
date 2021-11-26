
#ifndef KMEANS_LANDMARK_H
#define KMEANS_LANDMARK_H

#include <Magnum/Magnum.h>
#include <Magnum/Math/Vector2.h>

#include <random>
#include <vector>

#include "landmark_model.h"
#include "scaled_data.h"

struct KMeansData
{
    std::default_random_engine gen;
};

void
kmeans_landmark_step(KMeansData &data,
                     const ScaledData &model,
                     size_t iters,
                     float alpha,
                     float neighbor_alpha,
                     LandmarkModel &lm);

void
som_landmark_step(KMeansData &data,
                  const ScaledData &model,
                  size_t iters,
                  float alpha,
                  float sigma,
                  LandmarkModel &lm);

#endif
