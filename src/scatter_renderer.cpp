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

    texture_renderer.init();
}

void
ScatterRenderer::draw(const glm::vec2 &fb_size,
                      const View &view,
                      const ScatterModel &model,
                      const ColorData &colors)
{
    texture_renderer.bind_fb(fb_size);

    size_t n =
      std::min(model.points.size(),
               colors.data.size()); // misalignment aborts it, be careful

    prepare_data(model, colors);

    shader.use();
    shader.set_mat4("model", glm::mat4(1.0f));
    shader.set_mat4("view", view.get_view_matrix());
    shader.set_mat4("proj", view.get_proj_matrix());

  
    glBindVertexArray(VAO);
    glEnable(GL_BLEND);
    glDrawArrays(GL_POINTS, 0, n/2);
    
    glDisable(GL_BLEND);
  
    texture_renderer.bind_fb(fb_size);
    glBindVertexArray(VAO);
    glEnable(GL_BLEND);
    glDrawArrays(GL_POINTS, n/2, n/2);

    glDisable(GL_BLEND);

    texture_renderer.bind_default_fb();

    texture_renderer.render();
}

void
ScatterRenderer::prepare_data(const ScatterModel &model,
                              const ColorData &colors)
{
    glBindVertexArray(VAO);

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
