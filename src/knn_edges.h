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

#ifndef KNN_EDGES_H
#define KNN_EDGES_H

#include "landmark_model.h"

struct KnnEdgesData
{
    // actually not used now
    size_t last_point;

    KnnEdgesData()
      : last_point(0)
    {}
};

void
make_knn_edges(KnnEdgesData &data, LandmarkModel &landmarks, size_t kns);

#endif
