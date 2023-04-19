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

#include "texture_renderer.h"

#include "shaders.h"

TextureRenderer::TextureRenderer()
  : screen_quad_data({ -1.0f,
                       -1.0f,
                       1.0f,
                       -1.0f,
                       -1.0f,
                       1.0f,
                       -1.0f,
                       1.0f,
                       1.0f,
                       -1.0f,
                       1.0f,
                       1.0f })
{
}

void
TextureRenderer::init()
{
    glGenVertexArrays(1, &VAO_quad);
    glGenBuffers(1, &VBO_quad);

    glGenFramebuffers(1, &fb);
    glGenTextures(1, &texture);

    shader_tex.build(tex_vs, tex_fs);

    glBindFramebuffer(GL_FRAMEBUFFER, fb);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(
      GL_TEXTURE_2D, 0, GL_RGB, 800, 600, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);

    glFramebufferTexture2D(
      GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void
TextureRenderer::activate()
{
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);
}

void
TextureRenderer::deactivate()
{
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void
TextureRenderer::render()
{
    prepare_screen_quad_data();

    shader_tex.use();
    shader_tex.set_int("renderedTexture", 0);
    
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);

    glBindVertexArray(VAO_quad);
    glEnable(GL_BLEND);
    glDrawArrays(
      GL_TRIANGLES, 0, 6); // 2*3 indices starting at 0 -> 2 triangles
    glDisable(GL_BLEND);

    glDisableVertexAttribArray(0);
}

void
TextureRenderer::resize_fb(const glm::vec2 &fb_size)
{
    glBindFramebuffer(GL_FRAMEBUFFER, fb);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 GL_RGB,
                 fb_size.x,
                 fb_size.y,
                 0,
                 GL_RGB,
                 GL_UNSIGNED_BYTE,
                 NULL);
    glFramebufferTexture2D(
      GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void
TextureRenderer::bind_fb(const glm::vec2 &fb_size)
{
    resize_fb(fb_size);

    GLenum DrawBuffers[1] = { GL_COLOR_ATTACHMENT0 };
    glDrawBuffers(1, DrawBuffers);

    glBindFramebuffer(GL_FRAMEBUFFER, fb);
    glClear(GL_COLOR_BUFFER_BIT);
}

void
TextureRenderer::prepare_screen_quad_data()
{
    glBindVertexArray(VAO_quad);

    glBindBuffer(GL_ARRAY_BUFFER, VBO_quad);
    glBufferData(GL_ARRAY_BUFFER,
                 screen_quad_data.size() * sizeof(float),
                 &screen_quad_data[0],
                 GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void *)0);
    glEnableVertexAttribArray(0);
}
