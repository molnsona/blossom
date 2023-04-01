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

    // TODO add variance

    // Compute estimated number of points.
    // float n = (t - const_time) / time_per_point;
    // N = n <= 0 ? 100 : n;
    // prevT = T;

    // // Subtract random value from N.
    // std::random_device rd{};
    // std::mt19937 gen{ rd() };
    // std::normal_distribution<> d{ 100, 50 };
    // float rv = std::round(std::abs(d(gen)));

    // N = rv < N ? N - rv : N;

    compute_n(t, const_time, time_per_point);
    return N;
}

void
BatchSizeGen::compute_n(float t, float x, float y)
{
    std::array<float, 2> mu{ y, x };
    auto cov_m = estimator.get_cov_matrix();
    auto [U, SIGMA, V] = get_svd(cov_m);

    // Prepare scale factor.
    auto z = estimator.get_z(x, y);
    auto var = estimator.get_var();
    auto scale_x = sqrt(z / SIGMA[0] / var);
    auto scale_y = sqrt(z / SIGMA[1] / var);

    // Get axes vectors at [0,0] from magnitudes of the axes in SIGMA
    // and directions of the axes in V.
    mat2x2 axes{
        V[0] * scale_x, V[1] * scale_x, V[2] * scale_y, V[3] * scale_y
    };

    // Move axes to the center of the ellipse
    mat2x1 aaxis{ mu[0] - axes[0], mu[0] - axes[1] };
    mat2x1 baxis{ mu[1] - axes[2], mu[1] - axes[3] };

    // Get coefficients of the quadratic equation
    float a1 = aaxis[0];
    float a2 = aaxis[1];
    float b1 = baxis[0];
    float b2 = baxis[1];
    float c1 = mu[0];
    float c2 = mu[1];
    float w = 1.64;
    float qa = pow(t - c2, 2) - (pow(a2, 2) + pow(b2, 2)) * pow(w, 2);
    float qb = -(2 * c1 * (t - c2) + 2 * (a1 * a2 + b1 * b2) * pow(w, 2));
    float qc = pow(c1, 2) - (pow(a1, 2) + pow(b1, 2)) * pow(w, 2);

    // Find N
    constexpr size_t max = std::numeric_limits<size_t>::max();
    float D = pow(qb, 2) - 4 * qa * qc;
    float N1 = (-qb + sqrt(D)) / 2 * qa;
    float N2 = (-qb - sqrt(D)) / 2 * qa;
    if (N1 > 0.6 && N1 < max)
        N = round(N1);
    else if (N2 > 0.6 && N2 < max)
        N = round(N2);
    else
        N = 100;
}

std::tuple<mat2x2, mat2x1, mat2x2>
BatchSizeGen::get_svd(mat2x2 A)
{
    auto a = A[0];
    auto b = A[1];
    auto c = A[2];
    auto d = A[3];

    auto trace = a + d;
    auto det = a * d - b * c;

    // Eigenvalues
    float lambda1 = (trace + sqrt(pow(trace, 2) - 4 * det)) / 2;
    float lambda2 = (trace - sqrt(pow(trace, 2) - 4 * det)) / 2;

    // Eigenvectors
    mat2x1 u1;
    mat2x1 u2;
    if (b != 0) {
        u1 = { b, lambda1 - a };
        u2 = { b, lambda2 - a };
    } else if (c != 0) {
        u1 = { lambda1 - d, c };
        u2 = { lambda2 - d, c };
    } else if (b == 0 && c == 0) {
        u1 = { 1, 0 };
        u2 = { 0, 1 };
    }

    // Normalize eigenvectors
    float u1_length = sqrt(pow(u1[0], 2) + pow(u1[1], 2));
    float u2_length = sqrt(pow(u2[0], 2) + pow(u2[1], 2));
    u1[0] = u1[0] / u1_length;
    u1[1] = u1[1] / u1_length;
    u2[0] = u2[0] / u2_length;
    u2[1] = u2[1] / u2_length;

    // WLOG Flip signs of the eigenvectors
    u1[0] *= -1;
    u1[1] *= -1;
    u2[0] *= -1;
    u2[1] *= -1;

    // Create U matrix
    mat2x2 U{ u1[0], u2[0], u1[1], u2[1] };

    // Singular values
    float sigma1 = abs(lambda1);
    float sigma2 = abs(lambda2);

    // Singular values must be in decreasing order
    if (sigma1 < sigma2) {
        float tmp = sigma1;
        sigma1 = sigma2;
        sigma2 = tmp;
    }

    // Create Sigma
    mat2x1 SIGMA{ sigma1, sigma2 };

    // Create V = U^T
    mat2x2 V{ u1[0], u1[1], u2[0], u2[1] };

    return { U, SIGMA, V };
}
