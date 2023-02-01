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

#include <cmath>

//#define DEBUG
#ifdef DEBUG
#include <iostream>
#endif

BatchSizeGen::BatchSizeGen()
  : a(0.00001)
  , b(0.00001)
  , c(0.00001)
  , d(0.00001)
  , e(0.00001)
  , f(0.00001)
  , alpha(0.05)
  , coalpha(1 - alpha)
  , N(100)
  , prevT(0.0f)
{
}

size_t
BatchSizeGen::next(float T)
{
    // Prevent increase of batch size to inifinity
    // when SOM or kmeans is turned off or when no
    // data set is loaded.
    if (std::abs(prevT - T) < 0.0001)
        return N;

    // Computation time of one point.
    float TN = T / N;
    // Normalized normal line to the line with slope (-T, T/N).
    // The normal line before normalization is (T, T/N).
    float n1 = TN * (1 / (std::sqrt(TN * TN + T * T)));
    float n2 = T * (1 / (std::sqrt(TN * TN + T * T)));
    // Distance of the line from origin [0,0].
    float n3 = T * n1;

#ifdef DEBUG
    std::cout << "N: " << N << std::endl;
    std::cout << "T: " << T << std::endl;
    std::cout << "TN: " << TN << std::endl;
    std::cout << "n1: " << n1 << std::endl;
    std::cout << "n2: " << n2 << std::endl;
    std::cout << "n3: " << n3 << std::endl;
#endif

    a = a * coalpha + n1 * n1 * alpha;
    b = b * coalpha + n2 * n2 * alpha;
    c = c * coalpha + (2 * n1 * n2) * alpha;
    d = d * coalpha + (-2 * n1 * n3) * alpha;
    e = e * coalpha + (-2 * n2 * n3) * alpha;
    f = f * coalpha + n3 * n3 * alpha;
#ifdef DEBUG
    // std::cout << "a: " << a << std::endl;
    // std::cout << "b: " << b << std::endl;
    // std::cout << "c: " << c << std::endl;
    std::cout << "d: " << d << std::endl;
    std::cout << "e: " << e << std::endl;
    // std::cout << "f: " << f << std::endl;
#endif

    float x = (c * e - 2 * b * d) / (4 * a * b - c * c);
    float y = (c * d - 2 * a * e) / (4 * a * b - c * c);

#ifdef DEBUG
    std::cout << "x: " << x << std::endl;
    std::cout << "y: " << y << std::endl;
#endif

    // We want the algorithm to last 5ms.
#ifndef ENABLE_CUDA
    float t = 5;
#else
    float t = 5;
#endif
    float n = (t - x) / y;
    N = n < 0 ? 100 : n;
    prevT = T;
    return N;
}
