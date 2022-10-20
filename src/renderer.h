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
     * @brief Process input events and render graph and scatterplot with updated
     * values.
     *
     * @param state
     * @param view
     * @param callbacks
     */
    void update(State &state, View &view, const CallbackValues &callbacks);

private:
    ScatterRenderer scatter_renderer;
    GraphRenderer graph_renderer;

    /**
     * @brief Render graph and scatterplot.
     *
     * @param state
     * @param view
     */
    void render(State &state, View &view);

    /**
     * @brief Process mouse input and moves the landmarks.
     *
     * @param state
     * @param view
     * @param callbacks
     */
    void process_mouse(State &state,
                       const View &view,
                       const CallbackValues &callbacks);
    /**
     * @brief Process keyboard input.
     *
     * @param state
     * @param view
     * @param callbacks
     */
    void process_keyboard(State &state,
                          const View &view,
                          const CallbackValues &callbacks);
};

#endif // RENDERER_H
