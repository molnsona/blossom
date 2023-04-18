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

    glGenFramebuffers(1, &fb);
    glGenTextures(1, &texture);

    shader_tex.build(tex_vs, tex_fs);

    glBindFramebuffer(GL_FRAMEBUFFER, fb);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0,GL_RGB, 800, 600, 0,GL_RGB, GL_UNSIGNED_BYTE, NULL);
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); 
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);
   
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

}

void
ScatterRenderer::draw(const View &view,
                      const ScatterModel &model,
                      const ColorData &colors)
{
    GLenum DrawBuffers[1] = {GL_COLOR_ATTACHMENT0};
    glDrawBuffers(1, DrawBuffers);

    glBindFramebuffer(GL_FRAMEBUFFER, fb);
		glClear(GL_COLOR_BUFFER_BIT);
		//glClearColor(0.8, 0.8, 0.8, 0.0);

    size_t n =
      std::min(model.points.size(),
               colors.data.size()); // misalignment aborts it, be careful

    prepare_data(model, colors);

    shader.use();
    shader.set_mat4("model", glm::mat4(1.0f));
    shader.set_mat4("view", view.get_view_matrix());
    shader.set_mat4("proj", view.get_proj_matrix());

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);
     
    glBindVertexArray(VAO);
    glEnable(GL_BLEND);
    glDrawArrays(GL_POINTS, 0, n);

    glDisable(GL_BLEND);

    glBindTexture(GL_TEXTURE_2D, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0,0,800,600);

    //glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    GLuint quad_VertexArrayID;
    glGenVertexArrays(1, &quad_VertexArrayID);
    glBindVertexArray(quad_VertexArrayID);

    static const GLfloat g_quad_vertex_buffer_data[] = {
        -10.0f, -10.0f, 0.0f,
        10.0f, -10.0f, 0.0f,
        -10.0f,  10.0f, 0.0f,
        -10.0f,  10.0f, 0.0f,
        10.0f, -10.0f, 0.0f,
        10.0f,  10.0f, 0.0f,
    };

    GLuint quad_vertexbuffer;
    glGenBuffers(1, &quad_vertexbuffer);
    glBindBuffer(GL_ARRAY_BUFFER, quad_vertexbuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(g_quad_vertex_buffer_data), g_quad_vertex_buffer_data, GL_STATIC_DRAW);

    GLuint texID = glGetUniformLocation(shader_tex.ID, "renderedTexture");
    GLuint timeID = glGetUniformLocation(shader_tex.ID, "time");

    shader_tex.use();
    shader_tex.set_mat4("model", glm::mat4(1.0f));
    shader_tex.set_mat4("view", view.get_view_matrix());
    shader_tex.set_mat4("proj", view.get_proj_matrix());

    glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, texture);
		// Set our "renderedTexture" sampler to use Texture Unit 0
		glUniform1i(texID, 0);

		glUniform1f(timeID, (float)(10.0f) );

		// 1rst attribute buffer : vertices
		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, quad_vertexbuffer);
		glVertexAttribPointer(
			0,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
			3,                  // size
			GL_FLOAT,           // type
			GL_FALSE,           // normalized?
			0,                  // stride
			(void*)0            // array buffer offset
		);

		// Draw the triangles !
		glDrawArrays(GL_TRIANGLES, 0, 6); // 2*3 indices starting at 0 -> 2 triangles

		glDisableVertexAttribArray(0);

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
