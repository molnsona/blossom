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
};

#endif // #ifndef BATCH_SIZE_GEN_H
