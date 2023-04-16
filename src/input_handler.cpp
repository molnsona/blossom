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
    process_keyboard(view, renderer);
    process_mouse_button(view, renderer, state);
    process_mouse_scroll(view);
}

void
InputHandler::reset()
{
    input.reset();
}

void
InputHandler::process_keyboard(View &view, Renderer &renderer)
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

    if (key == GLFW_KEY_LEFT_SHIFT &&
        (action == GLFW_PRESS || action == GLFW_REPEAT))
        input.keyboard.shift_pressed = true;
    else if (key == GLFW_KEY_LEFT_SHIFT && action == GLFW_RELEASE)
        input.keyboard.shift_pressed = false;
}

void
InputHandler::process_mouse_button(View &view, Renderer &renderer, State &state)
{
    int action = input.mouse.action;
    auto pos = input.mouse.pos;
    auto model_mouse_pos = view.model_mouse_coords(input.mouse.pos);
    switch (input.mouse.button) {
        case GLFW_MOUSE_BUTTON_LEFT:
            switch (action) {
                case GLFW_PRESS:
                    if(state.colors.clustering.active_cluster != -1)
                    {
                        renderer.check_pressed_vertex(view, pos);
                        if(renderer.get_vert_pressed()) {
                            auto vert_ind = renderer.get_vert_ind();
                            state.colors.color_landmark(vert_ind);
                        }
                        break;
                    }

                    if (renderer.is_passive_multiselect()) {
                        renderer.check_pressed_rect(model_mouse_pos);
                        break;
                    }

                    if (input.keyboard.shift_pressed) {
                        renderer.start_multiselect(model_mouse_pos);
                        break;
                    }

                    renderer.check_pressed_vertex(view, pos);

                    if (input.keyboard.ctrl_pressed) {
                        renderer.add_vert(state, view, pos);
                        break;
                    }

                    break;
                case GLFW_RELEASE:
                    renderer.reset_pressed_vert();
                    renderer.reset_multiselect();
                    break;
                default:
                    break;
            }
            break;
        case GLFW_MOUSE_BUTTON_RIGHT:
            switch (action) {
                case GLFW_PRESS:
                    if (renderer.is_passive_multiselect()) {
                        renderer.stop_multiselect();
                        break;
                    }

                    if (input.keyboard.ctrl_pressed) {
                        renderer.check_pressed_vertex(view, pos);
                        renderer.remove_vert(state);
                        break;
                    }
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

    if(state.colors.clustering.active_cluster != -1) return;

    if (renderer.is_active_multiselect() && input.keyboard.shift_pressed) {
        renderer.update_multiselect(model_mouse_pos);
    } else if (renderer.get_rect_pressed()) {
        renderer.move_selection(model_mouse_pos, state.landmarks);
    } else if (renderer.get_vert_pressed() &&
               !renderer.is_passive_multiselect()) {
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
