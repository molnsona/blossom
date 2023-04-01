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

//#define DEBUG
#ifdef DEBUG
#include <iostream>
#endif

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
#ifdef DEBUG
    std::cout << "N: " << N << std::endl;
    std::cout << "T: " << T << std::endl;
    std::cout << "TN: " << TN << std::endl;
    std::cout << "n1: " << n1 << std::endl;
    std::cout << "n2: " << n2 << std::endl;
    std::cout << "n3: " << n3 << std::endl;

    std::cout << "a: " << a << std::endl;
    std::cout << "b: " << b << std::endl;
    std::cout << "c: " << c << std::endl;
    std::cout << "d: " << d << std::endl;
    std::cout << "e: " << e << std::endl;
    std::cout << "f: " << f << std::endl;
#endif
}

std::tuple<float, float>
Estimator::get_estimate()
{
    float x = (c * e - 2 * b * d) / (4 * a * b - c * c);
    float y = (c * d - 2 * a * e) / (4 * a * b - c * c);
#ifdef DEBUG
    std::cout << "x: " << x << std::endl;
    std::cout << "y: " << y << std::endl;
#endif
    return { x, y };
}

float
Estimator::get_z(float x, float y)
{
    return a * pow(x, 2) + b * pow(y, 2) + c * x * y + d * x + e * y + f;
}

float
Estimator::get_var()
{
    return a + b;
}

mat2x2
Estimator::get_cov_matrix()
{
    return { a, c / 2, c / 2, b };
}
