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
Renderer::render(const State &state, const View &view)
{
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    scatter_renderer.draw(view, state.scatter, state.colors);
    graph_renderer.draw(view, state.landmarks);
}

void
Renderer::check_pressed_vertex(const View &view, glm::vec2 mouse_pos)
{
    if (!graph_renderer.vert_pressed) {
        glm::vec2 screen_mouse = view.screen_mouse_coords(mouse_pos);
        if (graph_renderer.is_vert_pressed(view, screen_mouse)) {
            graph_renderer.vert_pressed = true;
        }
    }
}

void
Renderer::reset_pressed_vert()
{
    graph_renderer.vert_pressed = false;
}

bool
Renderer::get_vert_pressed()
{
    return graph_renderer.vert_pressed;
}

int
Renderer::get_vert_ind()
{
    return graph_renderer.vert_ind;
}

void
Renderer::add_vert(State &state, View &view, glm::vec2 mouse_pos)
{
    // Copy pressed landmark
    if (graph_renderer.vert_pressed)
        state.landmarks.duplicate(graph_renderer.vert_ind);
    // Add new landmark to cursor position
    else
        state.landmarks.add(view.model_mouse_coords(mouse_pos));
}

void
Renderer::remove_vert(State &state)
{
    if (graph_renderer.vert_pressed)
        // Remove landmark
        state.landmarks.remove(graph_renderer.vert_ind);
}

void
Renderer::move_vert(State &state, View &view, glm::vec2 mouse_pos)
{
    // Move landmark
    glm::vec2 model_mouse = view.model_mouse_coords(mouse_pos);

    state.landmarks.move(graph_renderer.vert_ind, model_mouse);
}
