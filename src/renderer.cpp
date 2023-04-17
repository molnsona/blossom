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
    ui_renderer.init();

    return true;
}

void
Renderer::render(const State &state, const View &view)
{
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    scatter_renderer.draw(view, state.scatter, state.colors);
    graph_renderer.draw(view, state.landmarks, state.colors);
    ui_renderer.draw(view);
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

size_t
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

void
Renderer::start_multiselect(glm::vec2 mouse_pos)
{
    ui_renderer.set_rect_start_point(mouse_pos);
}

bool
Renderer::is_active_multiselect()
{
    return ui_renderer.update_rect_pos;
}

bool
Renderer::is_passive_multiselect()
{
    return ui_renderer.draw_rect && !ui_renderer.update_rect_pos;
}

void
Renderer::update_multiselect(glm::vec2 mouse_pos, const LandmarkModel &model)
{
    ui_renderer.set_rect_end_point(mouse_pos, model);
}

void
Renderer::reset_multiselect()
{
    ui_renderer.update_rect_pos = false;
    ui_renderer.rect_pressed = false;
}

void
Renderer::stop_multiselect()
{
    ui_renderer.draw_rect = ui_renderer.update_rect_pos = false;
}

bool
Renderer::check_pressed_rect(glm::vec2 mouse_pos)
{
    return ui_renderer.is_rect_pressed(mouse_pos);
}

void
Renderer::move_selection(glm::vec2 mouse_pos, LandmarkModel &landmarks)
{
    ui_renderer.move_selection(mouse_pos, landmarks);
}

void
Renderer::draw_cursor_radius(const View &v, glm::vec2 mouse_pos, float r)
{
    ui_renderer.should_draw_circle(v, mouse_pos, r);
}

std::vector<size_t>
Renderer::get_landmarks_within_circle(const View &view,
                                      const glm::vec2 &pos,
                                      float radius,
                                      const LandmarkModel &landmarks)
{
    std::vector<size_t> ids;
    for (size_t i = 0; i < landmarks.n_landmarks(); ++i) {
        glm::vec2 screen_mouse = view.screen_mouse_coords(pos);
        glm::vec2 vert = view.screen_coords(landmarks.lodim_vertices[i]);
        if (ui_renderer.is_within_circle(vert, screen_mouse, radius))
            ids.emplace_back(i);
    }
    return ids;
}
