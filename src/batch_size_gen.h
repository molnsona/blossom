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

#ifndef BATCH_SIZE_GEN_H
#define BATCH_SIZE_GEN_H

#include <cstddef>

#include "estimator.h"

/**
 * @brief Generator of the size of the next point batch. It implements MLEM
 * algorithm described in the thesis text.
 *
 */
class BatchSizeGen
{
public:
    BatchSizeGen();

    void reset();

    /**
     * @brief Computes size of the next batch.
     *
     * @param T How long the computation lasted in the previous frame.
     * @param t How long the computation should run in the current frame.
     * @return size_t
     */
    size_t next(float T, float t);

private:
    Estimator estimator;
    size_t N;
    float prevT;

    /**
     * @brief Computes the size of the next batch.
     *
     * @param t How long the computation should run.
     * @param x Constant time of the computation.
     * @param y Time spent per one point.
     */
    void compute_n(float t, float x, float y);

    /**
     * @brief Compute SVD from a 2x2 symmetric matrix.
     *
     * @param A 2x2 symmetric matrix
     * @return std::tuple<mat2x2, mat2x1, mat2x2> SVD of matrix A = U SIGMA V.
     */
    std::tuple<mat2x2, mat2x1, mat2x2> get_svd(mat2x2 A);
};

#endif // #ifndef BATCH_SIZE_GEN_H
