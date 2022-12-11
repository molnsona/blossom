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

void
InputHandler::update(View &view, Renderer &renderer, State &state)
{
    process_keyboard(view);
    process_mouse_button(view, renderer, state);
    process_mouse_scroll(view);
}

void
InputHandler::reset()
{
    input.reset();
}

void
InputHandler::process_keyboard(View &view)
{
    int key = input.keyboard.key;
    int action = input.keyboard.action;
    if (key == GLFW_KEY_W && (action == GLFW_PRESS || action == GLFW_REPEAT))
        view.move_y(1);
    if (key == GLFW_KEY_S && (action == GLFW_PRESS || action == GLFW_REPEAT))
        view.move_y(-1);
    if (key == GLFW_KEY_A && (action == GLFW_PRESS || action == GLFW_REPEAT))
        view.move_x(-1);
    if (key == GLFW_KEY_D && (action == GLFW_PRESS || action == GLFW_REPEAT))
        view.move_x(1);

    if (key == GLFW_KEY_LEFT_CONTROL &&
        (action == GLFW_PRESS || action == GLFW_REPEAT)) {
        input.keyboard.ctrl_pressed = true;
    } else if (key == GLFW_KEY_LEFT_CONTROL && action == GLFW_RELEASE) {
        input.keyboard.ctrl_pressed = false;
    }
}

void
InputHandler::process_mouse_button(View &view, Renderer &renderer, State &state)
{
    int action = input.mouse.action;
    auto pos = input.mouse.pos;
    switch (input.mouse.button) {
        case GLFW_MOUSE_BUTTON_LEFT:
            switch (action) {
                case GLFW_PRESS:
                    renderer.check_pressed_vertex(view, pos);

                    if (input.keyboard.ctrl_pressed)
                        renderer.add_vert(state, view, pos);
                    break;
                case GLFW_RELEASE:
                    renderer.reset_pressed_vert();
                    break;
                default:
                    break;
            }
            break;
        case GLFW_MOUSE_BUTTON_RIGHT:
            switch (action) {
                case GLFW_PRESS:
                    renderer.check_pressed_vertex(view, pos);

                    if (input.keyboard.ctrl_pressed)
                        renderer.remove_vert(state);
                    break;
                case GLFW_RELEASE:
                    renderer.reset_pressed_vert();
                    break;
                default:
                    break;
            }
            break;
        case GLFW_MOUSE_BUTTON_MIDDLE:
            switch (action) {
                case GLFW_PRESS:
                    view.look_at(pos);
                    break;
                default:
                    break;
            }
            break;
        default:
            break;
    }

    if (renderer.get_vert_pressed()) {
        if (!input.keyboard.ctrl_pressed) {
            renderer.move_vert(state, view, input.mouse.pos);
        }
    }
}

void
InputHandler::process_mouse_scroll(View &view)
{
    view.zoom(input.mouse.yoffset, input.mouse.pos);
}
