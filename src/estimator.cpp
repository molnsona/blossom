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

#include "estimator.h"

#include <cmath>

Estimator::Estimator()
{
    reset();
}

void
Estimator::reset()
{
    a = 0.00001;
    b = 0.00001;
    c = 0.00001;
    d = 0.00001;
    e = 0.00001;
    f = 0.00001;
    alpha = 0.05;
    coalpha = 1 - alpha;
}

void
Estimator::process_measurement(size_t N, float T)
{
    // Prevent the division with zero.
    if(N == 0) N = 100;
    if(T == 0) T = 0.00001;

    // Computation time of one point.
    float TN = T / N;
    // Normalized normal line to the line with slope (-T, T/N).
    // The normal line before normalization is (T, T/N).
    float n1 = TN * (1 / (std::sqrt(TN * TN + T * T)));
    float n2 = T * (1 / (std::sqrt(TN * TN + T * T)));
    // Distance of the line from origin [0,0].
    float n3 = T * n1;

    a = a * coalpha + n1 * n1 * alpha;
    b = b * coalpha + n2 * n2 * alpha;
    c = c * coalpha + (2 * n1 * n2) * alpha;
    d = d * coalpha + (-2 * n1 * n3) * alpha;
    e = e * coalpha + (-2 * n2 * n3) * alpha;
    f = f * coalpha + n3 * n3 * alpha;
}

std::tuple<float, float>
Estimator::get_estimate()
{
    float x = (c * e - 2 * b * d) / (4 * a * b - c * c);
    float y = (c * d - 2 * a * e) / (4 * a * b - c * c);
    return { x, y };
}
