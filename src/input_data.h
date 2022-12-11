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

#ifndef INPUT_DATA_H
#define INPUT_DATA_H

#include "keyboard_data.h"
#include "mouse_data.h"

/**
 * @brief Input events data storage.
 *
 */
struct InputData {
    MouseData mouse;
    KeyboardData keyboard;

    InputData() { reset(); }

    int fb_width = 800;
    int fb_height = 600;

    int key;
    int key_action;

    double xoffset;
    double yoffset;

    // Raw mouse cursor position([0,0] in the upper left corner).
    // Have to convert it to coordinates with [0,0] in
    // the middle of the screen.
    double xpos;
    double ypos;
    int button;
    int mouse_action;
    bool left_click = false;

    void reset()
    {
        key = 0;
        xoffset = 0;
        yoffset = 0;
        button = -1;
    }    
};

#endif // #ifndef INPUT_DATA_H
