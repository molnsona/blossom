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

#ifndef TSNE_LAYOUT_H
#define TSNE_LAYOUT_H

#include <Magnum/Magnum.h>
#include <Magnum/Math/Vector2.h>

#include <vector>

#include "landmark_model.h"
#include "mouse_data.h"

/** A context structure for tSNE computation.
 *
 * This mostly holds pre-allocated memory so that the vectors don't need to get
 * recreated every frame.
 */
struct TSNELayoutData
{
    std::vector<float> pji;
    std::vector<size_t> heap;
    std::vector<Vector2> updates;
};

/** Optimize the positions of low-dimensional landmarks using the t-SNE
 * algorithm. */
void
tsne_layout_step(TSNELayoutData &data,
                 const MouseData &mouse,
                 LandmarkModel &lm,
                 float time);
#endif
