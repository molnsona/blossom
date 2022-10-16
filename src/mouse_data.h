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

#ifndef MOUSE_DATA_H
#define MOUSE_DATA_H

#include <glm/glm.hpp>

/**
 * @brief Mouse events data storage.
 *
 */
struct MouseData
{
    /** Raw coordinates on the screen (upper left [0,0]). */
    glm::vec2 mouse_pos;

    /** Flag indicating if left mouse button was pressed. */
    bool left_pressed;
    /** Flag indicating if right mouse button was pressed. */
    bool right_pressed;
    /** Flag indicating if a vertex was pressed. */
    bool vert_pressed;
    /** Index of the pressed vertex. If the vertex was not pressed, it is UB. */
    size_t vert_ind;

    MouseData()
      : left_pressed(false)
      , right_pressed(false)
      , vert_pressed(false)
      , vert_ind(0)
    {}
};

#endif // #ifndef MOUSE_DATA_H
