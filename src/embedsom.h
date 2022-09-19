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

#ifndef EMBEDSOM_H
#define EMBEDSOM_H

#include <cstddef>

void
embedsom(const size_t n,
         const size_t n_landmarks,
         const size_t dim,
         const float boost,
         const size_t topn,
         const float adjust,
         const float *points,
         const float *hidim_lm,
         const float *lodim_lm,
         float *embedding);

#endif
