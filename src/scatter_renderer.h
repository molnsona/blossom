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

#ifndef SCATTER_RENDERER_H
#define SCATTER_RENDERER_H

#include "color_data.h"
#include "scatter_model.h"
#include "view.h"

#include "shader.h"

/**
 * @brief Renderer of the 2D data points.
 *
 */
struct ScatterRenderer
{
    ScatterRenderer();

    void init();

    /**
     * @brief Draw event of the 2D data points.
     *
     * Renders data points at given positions.
     *
     * @param v View of the whole window.
     * @param m Model that contains 2D coordinates.
     * @param colors Data that contains colors of the points.
     */
    void draw(const View &v, const ScatterModel &m, const ColorData &colors);

private:
    Shader shader;
    unsigned int VAO;
    unsigned int VBO_pos;
    unsigned int VBO_col;

    void prepare_data(const ScatterModel &model, const ColorData &colors);
};

#endif
