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

#include "graph_renderer.h"

#include "glm/gtc/matrix_transform.hpp"

#include <cmath>
#include <iostream>

#include "shaders.h"

GraphRenderer::GraphRenderer()
{
    // line_mesh.setPrimitive(MeshPrimitive::Lines);
    // circle_mesh = MeshTools::compile(Primitives::circle2DSolid(36));

    // // Setup proper blending function.
    // GL::Renderer::setBlendFunction(
    //   GL::Renderer::BlendFunction::SourceAlpha,
    //   GL::Renderer::BlendFunction::OneMinusSourceAlpha);

}

void GraphRenderer::init()
{
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    
    shader.build(graph_vs, graph_fs);
}

void
GraphRenderer::draw(const View &view, const LandmarkModel &model)
{
    glm::vec3 cameraPos   = glm::vec3(0.0f, 0.0f,  10.0f);
    glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 cameraUp    = glm::vec3(0.0f, 1.0f,  0.0f);
    
    glEnable(GL_BLEND);
    
    prepare_data(view, model);
    
    // glm::mat4 view_matrix = glm::mat4(1.0f);
    // view_matrix = glm::translate(view_matrix, glm::vec3(0.5f, 0.5f, 0.0f));

    glm::mat4 proj = glm::perspective(glm::radians(45.0f), (float)800 / (float)600, 0.1f, 100.0f);
    ////proj = glm::ortho(0.0f, 800.0f, 0.0f, 600.0f, 0.1f, 100.0f);
    glm::mat4 view_mat = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);

    glm::mat4 model_mat = glm::mat4(1.0f);

    shader.use();
    shader.setMat4("model", model_mat);
    shader.setMat4("view", view.GetViewMatrix());//view_mat);//view.GetViewMatrix());
    shader.setMat4("proj", view.GetProjMatrix());//proj);
    //shader.setMat4("proj", view.screen_projection_matrix());

    glBindVertexArray(VAO);
    
    glEnable(GL_PROGRAM_POINT_SIZE);
    glDrawArrays(GL_POINTS, 0, 4); //model.lodim_vertices.size());
    glDisable(GL_PROGRAM_POINT_SIZE);

    glDisable(GL_BLEND);

    // // TODO cache these allocations in GraphRenderer object
    // // std::vector<Vector2> vertices(model.lodim_vertices.size());
    // if (vertices.size() != model.lodim_vertices.size()) {
    //     vertices.clear();
    //     vertices.resize(model.lodim_vertices.size());
    // }

    // for (size_t i = 0; i < vertices.size(); ++i) {
    //     vertices[i] = view.screen_coords(model.lodim_vertices[i]);
    // }

    // std::vector<Vector2> edge_lines(2 * model.edges.size());
    // for (size_t i = 0; i < model.edges.size(); ++i) {
    //     edge_lines[2 * i + 0] = vertices[model.edges[i].first];
    //     edge_lines[2 * i + 1] = vertices[model.edges[i].second];
    // }

    // GL::Buffer buffer;
    // buffer.setData(
    //   Corrade::Containers::ArrayView(edge_lines.data(), edge_lines.size()));

    // GL::Renderer::enable(GL::Renderer::Feature::Blending);

    // line_mesh.setCount(edge_lines.size())
    //   .addVertexBuffer(std::move(buffer), 0, decltype(flat_shader)::Position{});

    // auto screen_proj = view.screen_projection_matrix();

    // flat_shader.setTransformationProjectionMatrix(screen_proj)
    //   .setColor(0xc01010_rgbf)
    //   .draw(line_mesh);

    // flat_shader.setColor(0x66666666_rgbaf);
    // for (auto &&v : vertices) {
    //     flat_shader
    //       .setTransformationProjectionMatrix(
    //         screen_proj * Matrix3::translation(v) *
    //         Matrix3::scaling(Vector2(vertex_size)))
    //       .draw(circle_mesh);
    // }

    // GL::Renderer::disable(GL::Renderer::Feature::Blending);
}

void
GraphRenderer::prepare_data(const View &view, const LandmarkModel &model)
{
    float point[] = {0.0f, 0.0f,
                    1.0f, 0.0f,
                    0.0f, 1.0f,
                    1.0f, 1.0f};

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(point), point, GL_DYNAMIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // float landmarks[] = {0.0f, 0.0f,
    //                 1.0f, 0.0f,
    //                 0.0f, 1.0f,
    //                 1.0f, 1.0f,};

    // glBindVertexArray(VAO);

    // glBindBuffer(GL_ARRAY_BUFFER, VBO);
    // glBufferData(GL_ARRAY_BUFFER,
    //              sizeof(landmarks),
    //              landmarks,
    //              GL_STATIC_DRAW);//GL_DYNAMIC_DRAW);
    // glVertexAttribPointer(
    //   0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void *)0);
    // glEnableVertexAttribArray(0);




    // if (vertices.size() != model.lodim_vertices.size()) {
    //     vertices.clear();
    //     vertices.resize(model.lodim_vertices.size());
    // }

    // for (size_t i = 0; i < vertices.size(); ++i) {
    //     vertices[i] = /*view.screen_coords(*/model.lodim_vertices[i]/*)*/;
    //     //std::cout << vertices[i].x << vertices[i].y << std::endl;
    // }

    // // std::vector<glm::vec2> edge_lines(2 * model.edges.size());
    // // for (size_t i = 0; i < model.edges.size(); ++i) {
    // //     edge_lines[2 * i + 0] = vertices[model.edges[i].first];
    // //     edge_lines[2 * i + 1] = vertices[model.edges[i].second];
    // // }

    // glBindVertexArray(VAO);

    // glBufferData(GL_ARRAY_BUFFER,
    //           vertices.size() * 2 * sizeof(float),//sizeof(glm::vec2),
    //           &vertices[0],
    //           GL_DYNAMIC_DRAW);
    // glVertexAttribPointer(
    //   0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float)/*sizeof(glm::vec2)*/, (void *)0);
    // glEnableVertexAttribArray(0);
}

// bool
// GraphRenderer::is_vert_pressed(Magnum::Vector2 mouse, size_t &vert_ind) const
// {
//     float radius = vertex_size;

//     for (size_t i = 0; i < vertices.size(); ++i) {
//         auto vert = vertices[i];
//         if ((mouse.x() >= roundf(vert.x()) - radius) &&
//             (mouse.x() <= roundf(vert.x()) + radius) &&
//             (mouse.y() >= roundf(vert.y()) - radius) &&
//             (mouse.y() <= roundf(vert.y()) + radius)) {
//             vert_ind = i;
//             return true;
//         }
//     }

//     return false;
// }
