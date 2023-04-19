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

#ifndef TEXTURE_RENDERER_H
#define TEXTURE_RENDERER_H

#include "shader.h"
#include "view.h"

/**
 * @brief Takes care of the rendering to the texture
 * and then rendering the texture to the screen.
 * 
 */
struct TextureRenderer
{
    TextureRenderer();

    void init();

    void activate(const glm::vec2 &fb_size);
    void deactivate();
    void render(const View &view);
    Shader shader_tex;
    unsigned int VAO_quad;
    unsigned int VBO_quad;

    unsigned int fb;
    unsigned int texture;
    void resize_fb(const glm::vec2 &fb_size);
    void bind_fb();

    GLuint texID;
    GLuint timeID;

    void prepare_screen_quad_data();
private:
};

#endif // #ifndef TEXTURE_RENDERER_H
