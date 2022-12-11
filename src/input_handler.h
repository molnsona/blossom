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

#ifndef INPUT_HANDLER_H
#define INPUT_HANDLER_H

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glad/glad.h>

#include "input_data.h"
#include "view.h"

/**
 * @brief Handler of input events.
 * 
 */
class InputHandler {
public:
    InputData input;

    void update(View& view);
    void reset();
private:
    /**
     * @brief Process keyboard input.
     *
     * @param key Name of the used key.
     * @param action Pressed, released or hold key.
     */

    /**
     * @brief Process keyboard input and let other parts(listed in arguments)
     * know about new input.
     * 
     * @param view 
     */
    void process_keyboard(View& view);
};

#endif // #ifndef INPUT_HANDLER_H
