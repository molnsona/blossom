/* This file is part of BlosSOM.
 *
 * Copyright (C) 2021 Mirek Kratochvil
 *
 * BlosSOM is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option)
 * any later version.
 *
 * BlosSOM is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * BlosSOM. If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef KMEANS_LANDMARK_H
#define KMEANS_LANDMARK_H

#include <Magnum/Magnum.h>
#include <Magnum/Math/Vector2.h>

#include <random>
#include <vector>

#include "landmark_model.h"
#include "scaled_data.h"

/** Structure for storing the kmeans-style data */
struct KMeansData
{
    /** Random engine for picking the points for training. */
    std::default_random_engine gen;
};

/** Run a k-means-like optimization of high-dimensional landmark positions */
void
kmeans_landmark_step(KMeansData &data,
                     const ScaledData &model,
                     size_t iters,
                     float alpha,
                     float neighbor_alpha,
                     LandmarkModel &lm);

/** Run a SOM to optimize high-dimensional landmark positions. */
void
som_landmark_step(KMeansData &data,
                  const ScaledData &model,
                  size_t iters,
                  float alpha,
                  float sigma,
                  LandmarkModel &lm);

#endif
