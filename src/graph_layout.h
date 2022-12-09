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

#ifndef LAYOUT_H
#define LAYOUT_H

#include <glm/glm.hpp>

#include <vector>

#include "landmark_model.h"
#include "mouse_data.h"

/**
 * @brief Data for landmark graph layouting algorithm using forces.
 *
 */
struct GraphLayoutData
{
    /** Velocities of 2D landmarks. */
    std::vector<glm::vec2> velocities;
    /** Forces of 2D landmarks. */
    std::vector<glm::vec2> forces; // kept allocated for efficiency
};

/**
 * @brief One iteration step of the landmark layouting algorithm.
 *
 * @param data Data of the layouting algorithm.
 * @param mouse Coordinations of the mouse.
 * @param lm Landmark model.
 * @param time Time duration of the last frame.
 */
void
graph_layout_step(GraphLayoutData &data,
                  const MouseData &mouse,
                  LandmarkModel &lm,
                  float time);

#endif
