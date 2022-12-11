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

#include "input_handler.h"

void InputHandler::update(View& view) {
    process_keyboard(view);
}

void InputHandler::reset() {
    input.reset();
}

void InputHandler::process_keyboard(View& view)
{
    if (input.key == GLFW_KEY_W &&
        (input.key_action == GLFW_PRESS || input.key_action == GLFW_REPEAT))
        view.move_y(1);
        //target_pos.y += velocity;
    if (input.key == GLFW_KEY_S &&
        (input.key_action == GLFW_PRESS || input.key_action == GLFW_REPEAT))
        view.move_y(-1);
        //target_pos.y -= velocity;
    if (input.key == GLFW_KEY_A &&
        (input.key_action == GLFW_PRESS || input.key_action == GLFW_REPEAT))
        view.move_x(-1);
        //target_pos.x -= velocity;
    if (input.key == GLFW_KEY_D &&
        (input.key_action == GLFW_PRESS || input.key_action == GLFW_REPEAT))
        view.move_x(1);
        //target_pos.x += velocity;
}
