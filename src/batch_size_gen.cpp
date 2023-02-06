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
#include <random>

//#define DEBUG
#ifdef DEBUG
#include <iostream>
#endif

BatchSizeGen::BatchSizeGen()  
{
    reset();
}

void BatchSizeGen::reset() 
{
    a = 0.00001;
    b = 0.00001;
    c = 0.00001;
    d = 0.00001;
    e = 0.00001;
    f = 0.00001;
    alpha = 0.05;
    coalpha = 1 - alpha;
    N = 100;
    prevT = 0.0f;
}

size_t
BatchSizeGen::next(float T, float t)
{
    // If the algorithm should last 0ms, just return
    // and reset values.
    if(t == 0.0f) {
        reset();
        return N;
    }
    // Prevent increase of batch size to inifinity
    // when SOM or kmeans is turned off or when no
    // data set is loaded.
    if (std::abs(prevT - T) < 0.0001f){
        reset();
        return N;}

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

    float x = (c * e - 2 * b * d) / (4 * a * b - c * c);
    float y = (c * d - 2 * a * e) / (4 * a * b - c * c);

    // variance and standard deviation
    float z = a*x*x + b*y*y + c*x*y + d*x + e*y + f;
    float sd = std::sqrt(z);

#ifdef DEBUG
    // std::cout << "N: " << N << std::endl;
    // std::cout << "T: " << T << std::endl;
    // std::cout << "TN: " << TN << std::endl;
    // std::cout << "n1: " << n1 << std::endl;
    // std::cout << "n2: " << n2 << std::endl;
    // std::cout << "n3: " << n3 << std::endl;

    // std::cout << "a: " << a << std::endl;
    // std::cout << "b: " << b << std::endl;
    // std::cout << "c: " << c << std::endl;
    // std::cout << "d: " << d << std::endl;
    // std::cout << "e: " << e << std::endl;
    // std::cout << "f: " << f << std::endl;

    // std::cout << "x: " << x << std::endl;
    // std::cout << "y: " << y << std::endl;
    std::cout << "sd: " << sd << std::endl;
#endif

    float n = (t - x) / y;
    N = n <= 0 ? 100 : n;
    prevT = T;

    // Subtract random value from N.
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<> d{100, 50/*sd * 100000*/};
    float rv = std::round(std::abs(d(gen)));
#ifdef DEBUG
    std::cout << "rv: " << rv << std::endl;
#endif
    N = rv < N ? N - rv : N;    
    return N;
}
