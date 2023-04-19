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
{
}

void TextureRenderer::init()
{
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

   // prepare_screen_quad_data();
}

void TextureRenderer::activate(const glm::vec2 &fb_size)
{
    // resize_fb(fb_size);

    // // This is used in the fragment shader of the scatter
    // // to output the color at the location 0.
    // GLenum DrawBuffers[1] = {GL_COLOR_ATTACHMENT0};
    // glDrawBuffers(1, DrawBuffers);

    // glBindFramebuffer(GL_FRAMEBUFFER, fb);
    // glClear(GL_COLOR_BUFFER_BIT);
    // //glClearColor(0.8, 0.8, 0.8, 0.0);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);
}

void TextureRenderer::deactivate()
{
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    // glViewport(0,0,800,600);

    //glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void TextureRenderer::render(const View &view)
{
    shader_tex.use();
    // shader_tex.set_mat4("model", glm::mat4(1.0f));
    // shader_tex.set_mat4("view", view.get_view_matrix());
    // shader_tex.set_mat4("proj", view.get_proj_matrix());

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);
    // Set our "renderedTexture" sampler to use Texture Unit 0
    glUniform1i(texID, 0);

    glUniform1f(timeID, (float)(10.0f) );

    // 1rst attribute buffer : vertices
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, VBO_quad);
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

void TextureRenderer::resize_fb(const glm::vec2 &fb_size)
{

    // glBindTexture(GL_TEXTURE_2D, texture);
    // glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, fb_size.x, fb_size.y, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);

    glBindFramebuffer(GL_FRAMEBUFFER, fb);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0,GL_RGB, fb_size.x, fb_size.y, 0,GL_RGB, GL_UNSIGNED_BYTE, NULL);  
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);


}

void TextureRenderer::bind_fb()
{
    GLenum DrawBuffers[1] = {GL_COLOR_ATTACHMENT0};
    glDrawBuffers(1, DrawBuffers);

    glBindFramebuffer(GL_FRAMEBUFFER, fb);
		glClear(GL_COLOR_BUFFER_BIT);
		//glClearColor(0.8, 0.8, 0.8, 0.0);
}

void TextureRenderer::prepare_screen_quad_data()
{    
    glGenVertexArrays(1, &VAO_quad);
    glBindVertexArray(VAO_quad);

    // TODO move to constructor and to method
    // attribute.
    static const float screen_quad_data[] = {
        -1.0f, -1.0f, 0.0f,
        1.0f, -1.0f, 0.0f,
        -1.0f,  1.0f, 0.0f,
        -1.0f,  1.0f, 0.0f,
        1.0f, -1.0f, 0.0f,
        1.0f,  1.0f, 0.0f,
    };
    
    glGenBuffers(1, &VBO_quad);
    glBindBuffer(GL_ARRAY_BUFFER, VBO_quad);
    glBufferData(GL_ARRAY_BUFFER, sizeof(screen_quad_data), screen_quad_data, GL_STATIC_DRAW);

    // glVertexAttribPointer(
    //     0,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
    //     3,                  // size
    //     GL_FLOAT,           // type
    //     GL_FALSE,           // normalized?
    //     0,                  // stride
    //     (void*)0            // array buffer offset
    // );
    // glEnableVertexAttribArray(0);

    texID = glGetUniformLocation(shader_tex.ID, "renderedTexture");
    timeID = glGetUniformLocation(shader_tex.ID, "time");
}
