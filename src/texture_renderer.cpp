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
  , current_fb(num_of_textures - 1)
  , fb_size({ 800.0f, 600.0f })
{
}

void
TextureRenderer::init()
{
    glGenVertexArrays(1, &VAO_quad);
    glGenBuffers(1, &VBO_quad);

    shader_tex.build(tex_vs, tex_fs);

    GLenum DrawBuffers[1] = { GL_COLOR_ATTACHMENT0 };
    glDrawBuffers(1, DrawBuffers);

    gen_fbs();
    gen_textures();

    bind_fbs_and_textures();

    prepare_screen_quad_data();
}

void
TextureRenderer::bind_default_fb()
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void
TextureRenderer::render()
{
    shader_tex.use();
    shader_tex.set_int("renderedTexture", 0);

    glBindVertexArray(VAO_quad);

    glDisable(GL_BLEND);

    for (size_t i = 0; i < num_of_textures; ++i) {
        glBindTexture(GL_TEXTURE_2D, textures[i]);
        glDrawArrays(GL_TRIANGLES, 0, 6);
    }
}

void
TextureRenderer::bind_fb(const glm::vec2 &s)
{
    current_fb = (current_fb + 1) % num_of_textures;

    if (fb_size != s)
        resize_fbs(s);

    glBindFramebuffer(GL_FRAMEBUFFER, fbs[current_fb]);
    glClearColor(1.0f, 1.0f, 1.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);
}

void
TextureRenderer::gen_fbs()
{
    for (size_t i = 0; i < fbs.size(); ++i) {
        glGenFramebuffers(1, &fbs[i]);
    }
}

void
TextureRenderer::gen_textures()
{
    for (size_t i = 0; i < textures.size(); ++i) {
        glGenTextures(1, &textures[i]);
    }
}

void
TextureRenderer::bind_fbs_and_textures()
{
    for (size_t i = 0; i < fbs.size(); ++i) {
        glBindFramebuffer(GL_FRAMEBUFFER, fbs[i]);
        glBindTexture(GL_TEXTURE_2D, textures[i]);
        glTexImage2D(GL_TEXTURE_2D,
                     0,
                     GL_RGBA,
                     800,
                     600,
                     0,
                     GL_RGBA,
                     GL_UNSIGNED_BYTE,
                     NULL);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glBindTexture(GL_TEXTURE_2D, 0);

        glFramebufferTexture2D(
          GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textures[i], 0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
}

void
TextureRenderer::resize_fbs(const glm::vec2 &s)
{
    fb_size = s;
    for (size_t i = 0; i < num_of_textures; ++i) {
        glBindFramebuffer(GL_FRAMEBUFFER, fbs[i]);
        glBindTexture(GL_TEXTURE_2D, textures[i]);
        glTexImage2D(GL_TEXTURE_2D,
                     0,
                     GL_RGBA,
                     fb_size.x,
                     fb_size.y,
                     0,
                     GL_RGBA,
                     GL_UNSIGNED_BYTE,
                     NULL);
        glFramebufferTexture2D(
          GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textures[i], 0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
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
