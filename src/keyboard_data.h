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

#ifndef KEYBOARD_DATA_H
#define KEYBOARD_DATA_H

/**
 * @brief Keyboard events data storage.
 *
 */
struct KeyboardData
{
    /** Code of the key of the recent event.*/
    int key;

    /** Key action, whether it was pressed, released or held.*/
    int action;

    /** Flag indicating if CTRL was pressed. */
    bool ctrl_pressed;

    /** Flag indicating if SHIFT was pressed. */
    bool shift_pressed;

    KeyboardData()
      : ctrl_pressed(false)
      , shift_pressed(false)
    {
        reset();
    }

    void reset() { key = 0; }
};

#endif // #ifndef KEYBOARD_DATA_H
