/* This file is part of BlosSOM.
 *
 * Copyright (C) 2021 Sona Molnarova
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

#include "batch_size_gen.h"

#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <tuple>

BatchSizeGen::BatchSizeGen()
{
    reset();
}

void
BatchSizeGen::reset()
{
    estimator.reset();
    N = 100;
    prevT = 0.0f;
}

size_t
BatchSizeGen::next(float T, float t)
{
    // If the algorithm should last 0ms, just return
    // and reset values.
    if (t == 0.0f) {
        reset();
        return N;
    }
    
    // Prevent increase of batch size to inifinity
    // when SOM or kmeans is turned off or when no
    // data set is loaded.
    if (std::abs(prevT - T) < 0.0001f) {
        reset();
        return N;
    }

    estimator.process_measurement(N, T);
    auto [const_time, time_per_point] = estimator.get_estimate();

    // Compute estimated number of points.
    float n = (t - const_time) / time_per_point;
    N = n <= 0 ? 100 : n;
    prevT = T;

    // Subtract random value from N.
    std::random_device rd{};
    std::mt19937 gen{ rd() };
    std::normal_distribution<> d{ 100, 50 };
    float rv = std::round(std::abs(d(gen)));

    N = rv < N ? N - rv : N;

    return N;
}
