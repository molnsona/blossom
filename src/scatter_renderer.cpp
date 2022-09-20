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

#include "scatter_renderer.h"

#include "glm/gtc/matrix_transform.hpp"

#include <iostream>

#include "shaders.h"

ScatterRenderer::ScatterRenderer() {}

void
ScatterRenderer::init()
{
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO_pos);
    glGenBuffers(1, &VBO_col);

    shader.build(scatter_vs, scatter_fs);
}

void
ScatterRenderer::draw(const View &view,
                      const ScatterModel &model,
                      const ColorData &colors)
{
    glEnable(GL_BLEND);
    size_t n =
      std::min(model.points.size(),
               colors.data.size()); // misalignment aborts it, be careful

    prepare_data(model, colors);

    glm::mat4 view_matrix = glm::mat4(1.0f);
    view_matrix = glm::translate(view_matrix, glm::vec3(1.0f, 1.0f, 0.0f));

    glm::mat4 proj = glm::mat4(1.0f);
    proj = glm::ortho(0.0f, 800.0f, 0.0f, 600.0f, 0.1f, 100.0f);

    shader.use();
    shader.setMat4("view", view_matrix);
    shader.setMat4("proj", proj);
    //shader.setMat4("proj", view.projection_matrix());

    glBindVertexArray(VAO);
    glDrawArrays(GL_POINTS, 0, n);

    glDisable(GL_BLEND);
}

void
ScatterRenderer::prepare_data(const ScatterModel &model,
                              const ColorData &colors)
{
    glBindVertexArray(VAO);

    // for (size_t i = 0; i < model.points.size(); ++i) {
    //     std::cout << model.points[i].x << ", " << model.points[i].y << std::endl;
    // }

    glBindBuffer(GL_ARRAY_BUFFER, VBO_pos);
    glBufferData(GL_ARRAY_BUFFER,
                 model.points.size() * sizeof(glm::vec2),
                 &model.points[0],
                 GL_DYNAMIC_DRAW);
    glVertexAttribPointer(
      0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void *)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, VBO_col);
    glBufferData(GL_ARRAY_BUFFER,
                 colors.data.size() * sizeof(glm::vec4),
                 &colors.data[0],
                 GL_DYNAMIC_DRAW);
    glVertexAttribPointer(
      1, 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), (void *)0);
    glEnableVertexAttribArray(1);
}