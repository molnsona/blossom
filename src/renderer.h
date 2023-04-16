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
    void check_pressed_vertex(const View &view, glm::vec2 mouse_pos, float r = radius);

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

    void draw_cursor_radius(glm::vec2 mouse_pos, float r);

private:
    /** Radius around mouse cursor for checking pressed vertex.    
     */
    static constexpr float radius = 5.0f;


    ScatterRenderer scatter_renderer;
    GraphRenderer graph_renderer;
    UiRenderer ui_renderer;
};

#endif // RENDERER_H
