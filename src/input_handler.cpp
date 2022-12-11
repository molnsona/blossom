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
    if (input.key == GLFW_KEY_W &&
        (input.key_action == GLFW_PRESS || input.key_action == GLFW_REPEAT))
        view.move_y(1);
    if (input.key == GLFW_KEY_S &&
        (input.key_action == GLFW_PRESS || input.key_action == GLFW_REPEAT))
        view.move_y(-1);
    if (input.key == GLFW_KEY_A &&
        (input.key_action == GLFW_PRESS || input.key_action == GLFW_REPEAT))
        view.move_x(-1);
    if (input.key == GLFW_KEY_D &&
        (input.key_action == GLFW_PRESS || input.key_action == GLFW_REPEAT))
        view.move_x(1);

    if (input.key == GLFW_KEY_LEFT_CONTROL &&
        (input.key_action == GLFW_PRESS || input.key_action == GLFW_REPEAT)) {
        input.keyboard.ctrl_pressed = true;
    } else if (input.key == GLFW_KEY_LEFT_CONTROL &&
               input.key_action == GLFW_RELEASE) {
        input.keyboard.ctrl_pressed = false;
    }
}

void
InputHandler::process_mouse_button(View &view, Renderer &renderer, State &state)
{
    switch (input.button) {
        case GLFW_MOUSE_BUTTON_LEFT:
            switch (input.mouse_action) {
                case GLFW_PRESS:
                    if (!input.mouse.vert_pressed) {
                        glm::vec2 screen_mouse = view.screen_mouse_coords(
                          glm::vec2(input.xpos, input.ypos));
                        if (renderer.is_vert_pressed(
                              view, screen_mouse, input.mouse.vert_ind)) {
                            input.mouse.vert_pressed = true;
                        }
                    }

                    if (input.keyboard.ctrl_pressed)
                        // Copy pressed landmark
                        if (input.mouse.vert_pressed)
                            state.landmarks.duplicate(input.mouse.vert_ind);
                        // Add landmark
                        else
                            state.landmarks.add(view.model_mouse_coords(
                              glm::vec2(input.xpos, input.ypos)));
                    break;
                case GLFW_RELEASE:
                    input.mouse.vert_pressed = false;
                    break;
                default:
                    break;
            }
            break;
        case GLFW_MOUSE_BUTTON_RIGHT:
            switch (input.mouse_action) {
                case GLFW_PRESS:
                    if (!input.mouse.vert_pressed) {
                        glm::vec2 screen_mouse = view.screen_mouse_coords(
                          glm::vec2(input.xpos, input.ypos));
                        if (renderer.is_vert_pressed(
                              view, screen_mouse, input.mouse.vert_ind)) {
                            input.mouse.vert_pressed = true;
                        }
                    }
                    // Remove landmark
                    if (input.keyboard.ctrl_pressed && input.mouse.vert_pressed)
                        state.landmarks.remove(input.mouse.vert_ind);
                    break;
                case GLFW_RELEASE:
                    input.mouse.vert_pressed = false;
                    break;
                default:
                    break;
            }
            break;
        case GLFW_MOUSE_BUTTON_MIDDLE:
            switch (input.mouse_action) {
                case GLFW_PRESS:
                    view.look_at(glm::vec2(input.xpos, input.ypos));
                    break;
                default:
                    break;
            }
            break;
        default:
            break;
    }

    if (input.mouse.vert_pressed) {
        if (!input.keyboard.ctrl_pressed) { // Move landmark
            glm::vec2 model_mouse =
              view.model_mouse_coords(glm::vec2(input.xpos, input.ypos));

            state.landmarks.move(input.mouse.vert_ind, model_mouse);
        }
    }
}

void
InputHandler::process_mouse_scroll(View &view)
{
    view.zoom(input.yoffset, glm::vec2(input.xpos, input.ypos));
}
