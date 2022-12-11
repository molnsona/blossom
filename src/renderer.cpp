#include "renderer.h"

#include <iostream>

#include "glm/gtc/matrix_transform.hpp"
#include "shaders.h"

Renderer::Renderer() {}

bool
Renderer::init()
{
    glBlendEquation(GL_FUNC_ADD);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    scatter_renderer.init();
    graph_renderer.init();

    return true;
}

void
Renderer::update(State &state, View &view, InputData &input)
{
    process_keyboard(state, view, input);
    process_mouse(state, view, input);

    render(state, view);
}

void
Renderer::render(State &state, View &view)
{
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    scatter_renderer.draw(view, state.scatter, state.colors);
    graph_renderer.draw(view, state.landmarks);
}

void
Renderer::process_mouse(State &state,
                        const View &view,
                        InputData &input)
{
    switch (input.button) {
        case GLFW_MOUSE_BUTTON_LEFT:
            switch (input.mouse_action) {
                case GLFW_PRESS:
                    if (!input.mouse.vert_pressed) {
                        glm::vec2 screen_mouse =
                          view.screen_mouse_coords(glm::vec2(input.xpos, input.ypos));
                        if (graph_renderer.is_vert_pressed(
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
                        glm::vec2 screen_mouse =
                          view.screen_mouse_coords(glm::vec2(input.xpos, input.ypos));
                        if (graph_renderer.is_vert_pressed(
                              view, screen_mouse, input.mouse.vert_ind)) {
                            input.mouse.vert_pressed = true;
                        }
                    }
                    // Remove landmark
                    if (input.keyboard.ctrl_pressed && input.mouse.vert_pressed)
                        state.landmarks.remove(input.mouse.vert_ind);
                case GLFW_RELEASE:
                    input.mouse.vert_pressed = false;
                    break;
                default:
                    break;
            }

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
Renderer::process_keyboard(State &state,
                           const View &view,
                           InputData &input)
{
    if (input.key == GLFW_KEY_LEFT_CONTROL &&
        (input.key_action == GLFW_PRESS || input.key_action == GLFW_REPEAT)) {
        input.keyboard.ctrl_pressed = true;
    } else if (input.key == GLFW_KEY_LEFT_CONTROL &&
               input.key_action == GLFW_RELEASE) {
        input.keyboard.ctrl_pressed = false;
    }
}
