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
    /** Left, right or middle button.*/
    int button;
    /** Pressed, released or held button.*/
    int action;

    /** Offset of the mouse wheel along x-axis.*/
    double xoffset;
    /** Offset of the mouse wheel along y-axis.*/
    double yoffset;

    /** Raw mouse cursor coordinates on the screen
     *  ([0,0] in the upper left corner).
     *  Have to convert it to coordinates with [0,0] in the middle
     *  of the screen.
     */
    glm::vec2 pos;

    /** Flag indicating if a vertex was pressed. */
    bool vert_pressed;
    /** Index of the pressed vertex. If the vertex was not pressed, it is UB. */
    size_t vert_ind;

    MouseData()
      : vert_pressed(false)
      , vert_ind(0)
    {
        reset();
    }

    void reset()
    {
        button = -1;
        xoffset = 0;
        yoffset = 0;
    }
};

#endif // #ifndef MOUSE_DATA_H
