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
struct InputData
{
    MouseData mouse;
    KeyboardData keyboard;

    InputData() { reset(); }

    int fb_width = 800;
    int fb_height = 600;

    void reset()
    {
        keyboard.reset();
        mouse.reset();
    }
};

#endif // #ifndef INPUT_DATA_H
