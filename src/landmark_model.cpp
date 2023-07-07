/* This file is part of BlosSOM.
 *
 * Copyright (C) 2021 Mirek Kratochvil
 *                    Sona Molnarova
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

#include "landmark_model.h"

#include <limits>
#include <random>

LandmarkModel::LandmarkModel()
  : d(0)
{
    init_grid(0);
}

void
LandmarkModel::init_grid(size_t n)
{
    if (!d) {
        hidim_vertices.clear();
        lodim_vertices.clear();
        edges.clear();
        return;
    }

    lodim_vertices.resize(n * n);
    hidim_vertices.resize(n * n * d);
    edges.clear();

    std::default_random_engine gen;
    std::uniform_real_distribution<float> dist(-1, 1);

    for (size_t i = 0; i < n * n; ++i) {
        auto x = i % n;
        auto y = i / n;
        lodim_vertices[i] = glm::vec2(x, y);

        for (size_t di = 0; di < d; ++di)
            hidim_vertices[i * d + di] = (di & 1) ? x : y;
    }

    touch();
}

void
LandmarkModel::update_dim(size_t dim)
{
    if (dim == d)
        return;
    d = dim;
    init_grid(4);
}

void
LandmarkModel::move(size_t ind, const glm::vec2 &mouse_pos)
{
    lodim_vertices[ind] = mouse_pos;
    touch();
}

void
LandmarkModel::duplicate(size_t ind)
{
    // Add new line to hidim
    size_t line_idx = d * ind;
    for (size_t i = 0; i < d; ++i) {
        hidim_vertices.emplace_back(hidim_vertices[line_idx + i]);
    }

    // Add new vertex to lodim
    lodim_vertices.emplace_back(
      glm::vec2(lodim_vertices[ind].x + 0.3, lodim_vertices[ind].y));

    touch();
}

void
LandmarkModel::add(const glm::vec2 &mouse_pos)
{
    size_t vert_ind = closest_landmark(mouse_pos);

    // Add new vertex to lodim
    lodim_vertices.emplace_back(mouse_pos);

    // Add new line to hidim
    size_t line_idx = d * vert_ind;
    for (size_t i = 0; i < d; ++i) {
        hidim_vertices.emplace_back(hidim_vertices[line_idx + i]);
    }

    touch();
}

void
LandmarkModel::remove(size_t ind)
{
    lodim_vertices.erase(lodim_vertices.begin() + ind);
    size_t line_idx = d * ind;
    hidim_vertices.erase(hidim_vertices.begin() + line_idx,
                         hidim_vertices.begin() + line_idx + d);

    // Remove edges.
    std::vector<size_t> edge_idxs;
    size_t edge_ind = 0;
    for (auto i = edges.begin(); i != edges.end();) {
        if (i->first == ind || i->second == ind) {
            i = edges.erase(i);
            edge_lengths.erase(edge_lengths.begin() + edge_ind);
            continue;
        }
        ++edge_ind;
        ++i;
    }

    // Update indices of vertices that are after the removing vertex
    // so the edges have proper vertex indices
    for (auto i = edges.begin(); i != edges.end(); ++i) {
        if (i->first >= ind) {
            --i->first;
        }
        if (i->second >= ind) {
            --i->second;
        }
    }

    touch();
}

static float
distance(const glm::vec2 &x, const glm::vec2 &y)
{
    auto a = powf(x.x - y.x, 2);
    auto b = powf(x.y - y.y, 2);
    return sqrtf(a + b);
}

size_t
LandmarkModel::closest_landmark(const glm::vec2 &mouse_pos) const
{
    auto min_dist = std::numeric_limits<float>::max();
    size_t vert_ind = 0;
    for (size_t i = 0; i < lodim_vertices.size(); ++i) {
        auto dist = distance(lodim_vertices[i], mouse_pos);
        if (dist < min_dist) {
            min_dist = dist;
            vert_ind = i;
        }
    }

    return vert_ind;
}
