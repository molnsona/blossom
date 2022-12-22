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

#include "graph_layout.h"

// TODO: parametrize the constants

void
graph_layout_step(GraphLayoutData &data,
                  bool vert_pressed,
                  int vert_ind,
                  LandmarkModel &lm,
                  float time)
{
    auto &vertices = lm.lodim_vertices;
    const auto &edges = lm.edges;
    const auto &edge_lengths = lm.edge_lengths;

    if (time > 0.05)
        time = 0.05;
    // check if data is m'kay
    if (data.velocities.size() != vertices.size())
        data.velocities.resize(vertices.size(), glm::vec2(0, 0));
    if (data.forces.size() != vertices.size())
        data.forces.resize(vertices.size());

    // clean up the forces
    for (auto &v : data.forces)
        v = glm::vec2(0, 0);

    // add repulsive forces
    for (size_t i = 1; i < vertices.size(); ++i)
        for (size_t j = 0; j < i; ++j) {
            auto d = vertices[j] - vertices[i];
            float q = exp(-glm::length(d) * 3) * 1000 * time;
            data.forces[i] += -q * d;
            data.forces[j] += q * d;
        }

    // add compulsive forces
    for (size_t i = 0; i < edges.size(); ++i) {
        auto [p1, p2] = edges[i];
        auto d = vertices[p2] - vertices[p1];
        auto q = (edge_lengths[i] - glm::length(d)) * 1000 * time;
        data.forces[p1] += -q * d;
        data.forces[p2] += q * d;
    }

    // update velocities and positions
    auto slowdown = pow(0.1f, time);
    for (size_t i = 0; i < vertices.size(); ++i) {
        if (vert_pressed && vert_ind == i)
            continue;

        data.velocities[i] += data.forces[i] * time;
        vertices[i] += data.velocities[i] * time;
        data.velocities[i] *= slowdown;
    }

    lm.touch();
}
