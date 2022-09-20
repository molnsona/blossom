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

#include "landmark_model.h"
#include "shader.h"
//#include "view.h"

/**
 * @brief Renderer of the 2D landmark graph.
 *
 */
struct GraphRenderer
{
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
    void draw(/*const View &v, */const LandmarkModel &m);

    /**
     * @brief Checks if some vertex was pressed.
     *
     * @param[in] mouse Mouse screen coordinates.
     * @param[out] vert_ind If the vertex was pressed it returns the index of
     * the vertex, otherwise it is UB.
     * @return true If a vertex was pressed.
     * @return false If no vertex was pressed.
     */
    //bool is_vert_pressed(glm::vec2 mouse, size_t &vert_ind) const;

private:
    /** Radius of the vertex for rendering.
     * \todo TODO: Make dynamic according to the depth of the zoom.
     */
    static constexpr float vertex_size = 5.0f;

    // /** Mesh of the edge used to draw all edges. */
    // Magnum::GL::Mesh line_mesh;
    // /** Mesh of the vertex used to draw all vertices. */
    // Magnum::GL::Mesh circle_mesh;
    // /** Shader used for rendering both - edges and vertices. */
    // Magnum::Shaders::FlatGL2D flat_shader;

    /** Cached screen coordinates of the vertices. */
    std::vector<glm::vec2> vertices;

    Shader shader;
    unsigned int VAO;
    unsigned int VBO;

    void prepare_data(const LandmarkModel &model);
};

#endif // #ifndef GRAPH_RENDERER_H
