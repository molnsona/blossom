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

#ifndef RENDERER_H
#define RENDERER_H

#include "graph_renderer.h"
#include "scatter_renderer.h"
#include "state.h"
#include "ui_renderer.h"
#include "view.h"

/**
 * @brief Handles rendering of the graph and scatter plot and handles IO.
 *
 */
class Renderer
{
public:
    Renderer();
    bool init();

    /**
     * @brief Render graph and scatterplot.
     *
     * @param state
     * @param view
     */
    void render(const State &state, const View &view);

    /**
     * @brief Check whether the vertex was pressed and set flags.
     *
     * @param view
     * @param mouse
     * @param vert_ind
     * @return true
     * @return false
     */
    void check_pressed_vertex(const View &view, glm::vec2 mouse_pos);

    void reset_pressed_vert();

    bool get_vert_pressed();
    size_t get_vert_ind();

    void add_vert(State &state, View &view, glm::vec2 mouse_pos);
    void remove_vert(State &state);
    void move_vert(State &state, View &view, glm::vec2 mouse_pos);

    void start_multiselect(glm::vec2 mouse_pos);

    bool is_active_multiselect();
    bool is_passive_multiselect();

    void update_multiselect(glm::vec2 mouse_pos, const LandmarkModel &model);

    void reset_multiselect();
    void stop_multiselect();

    bool check_pressed_rect(glm::vec2 mouse_pos);
    bool get_rect_pressed() { return ui_renderer.rect_pressed; }

    void move_selection(glm::vec2 mouse_pos, LandmarkModel &landmarks);

    void start_brushing() { ui_renderer.is_brushing_active = true; }
    bool is_brushing_active() { return ui_renderer.is_brushing_active; }
    void stop_brushing() { ui_renderer.is_brushing_active = false; }

    void draw_cursor_radius(const View &v, glm::vec2 mouse_pos, float r);
    void stop_cursor_radius() { ui_renderer.draw_circle = false; }

    std::vector<size_t> get_landmarks_within_circle(
      const View &view,
      const glm::vec2 &pos,
      float radius,
      const LandmarkModel &landmarks);

private:
    ScatterRenderer scatter_renderer;
    GraphRenderer graph_renderer;
    UiRenderer ui_renderer;
};

#endif // RENDERER_H
