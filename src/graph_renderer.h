/* This file is part of BlosSOM.
 *
 * Copyright (C) 2021 Mirek Kratochvil
 *                    Sona Molnarova
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

#ifndef GRAPH_RENDERER_H
#define GRAPH_RENDERER_H

#include <vector>

#include "color_data.h"
#include "landmark_model.h"
#include "shader.h"
#include "view.h"

#include <array>
#include <vector>

/**
 * @brief Renderer of the 2D landmark graph.
 *
 */
struct GraphRenderer
{
    /** Flag indicating if a vertex was pressed. */
    bool vert_pressed;
    /** Index of the pressed vertex. If the vertex was not pressed, it is UB. */
    size_t vert_ind;

    GraphRenderer();

    void init();

    // TODO: this should not know about actual Landmarks, we should pass actual
    // vertex + edge positions as with the layouter.
    /**
     * @brief Draw event of the 2D landmark graph.
     *
     * Renders vertices and edges at current positions.
     *
     * @param v
     * @param m
     *
     * \todo TODO: this should not know about actual Landmarks, we should pass
     * actual vertex + edge positions as with the layouter.
     */
    void draw(const View &v, const LandmarkModel &m, const ColorData &colors);

    /**
     * @brief Checks if some vertex was pressed.
     *
     * @param[in] mouse Mouse screen coordinates.
     * @param[out] vert_ind If the vertex was pressed it returns the index of
     * the vertex, otherwise it is UB.
     * @return true If a vertex was pressed.
     * @return false If no vertex was pressed.
     */
    bool is_vert_pressed(const View &view, glm::vec2 mouse);

private:
    /** Radius of the vertex used for comparing, if
     * the landmark was pressed.
     */
    static constexpr float vertex_size = 5.0f;

    /** Cached screen coordinates of the vertices. */
    std::vector<glm::vec2> vertices;

    /** Number of all vertices for rendering circles(graph vertices).*/
    int num_all_vtxs;

    Shader shader_v;
    unsigned int VAO_v;
    unsigned int VBO_v_pos;
    unsigned int VBO_v_col;

    int num_all_vtxs_outlines;
    Shader shader_v_outline;
    unsigned int VAO_v_outline;
    unsigned int VBO_v_pos_outline;

    Shader shader_e;
    unsigned int VAO_e;
    unsigned int VBO_e;

    /**
     * @brief Prepare data to render vertices and edges.
     *
     * Fill VBOs and VAOs.
     *
     * @param current_zoom Current zoom of the "camera".
     * @param model Data source
     */
    void prepare_data(float current_zoom,
                      const LandmarkModel &model,
                      const ColorData &colors);
    /**
     * @brief Prepare graph vertices that are rendered as circles.
     *
     * @param current_zoom Current zoom of the "camera".
     * @param model Data source
     */
    void prepare_vertices(float current_zoom,
                          const LandmarkModel &model,
                          const ColorData &colors);
    /**
     * @brief Prepare graph edges that are rendered as lines.
     *
     * @param model Data source
     */
    void prepare_edges(const LandmarkModel &model);

    /**
     * @brief Add vertices for TRIANGLE_FAN that creates circle
     * at given position.
     *
     * @param middle_x x position of the middle of the circle.
     * @param middle_y y position of the middle of the circle.
     * @param zoom Current zoom level used to adjust the size of the circle.
     * @param all_vtxs Storage of the vertices that will be filled.
     */
    void add_circle(float middle_x,
                    float middle_y,
                    float zoom,
                    std::vector<float> &all_vtxs,
                    std::vector<float> &vtxs_outlines,
                    std::vector<glm::vec3> &all_colors,
                    const glm::vec3 &color);
};

#endif // #ifndef GRAPH_RENDERER_H
