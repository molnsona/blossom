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
  : vert_pressed(false)
  , vert_ind(0)
  , rect_indices({0,1,3,1,2,3})
  , draw_rect(false)
  , update_rect_pos(false)
  , rect_pressed(false)
{
}

void
GraphRenderer::init()
{
    glGenVertexArrays(1, &VAO_v);
    glGenBuffers(1, &VBO_v);

    shader_v.build(graph_v_vs, graph_v_fs);

    glGenVertexArrays(1, &VAO_e);
    glGenBuffers(1, &VBO_e);

    shader_e.build(graph_e_vs, graph_e_fs);

    glGenVertexArrays(1, &VAO_r);
    glGenBuffers(1, &VBO_r);
    glGenBuffers(1, &EBO_r);

    shader_r.build(graph_r_vs, graph_r_fs);
}

void
GraphRenderer::draw(const View &view, const LandmarkModel &model)
{
    glEnable(GL_BLEND);

    prepare_data(view.current_zoom, model);

    shader_e.use();
    shader_e.set_mat4("model", glm::mat4(1.0f));
    shader_e.set_mat4("view", view.get_view_matrix());
    shader_e.set_mat4("proj", view.get_proj_matrix());

    glBindVertexArray(VAO_e);
    glDrawArrays(GL_LINES, 0, 2 * model.edges.size());

    shader_v.use();
    shader_v.set_mat4("model", glm::mat4(1.0f));
    shader_v.set_mat4("view", view.get_view_matrix());
    shader_v.set_mat4("proj", view.get_proj_matrix());

    glBindVertexArray(VAO_v);

    for (size_t i = 0; i < model.lodim_vertices.size(); ++i) {
        glDrawArrays(GL_TRIANGLE_FAN, i * num_all_vtxs, num_all_vtxs);
    }

    if(draw_rect) {
        shader_r.use();
        shader_r.set_mat4("model", glm::mat4(1.0f));
        shader_r.set_mat4("view", view.get_view_matrix());
        shader_r.set_mat4("proj", view.get_proj_matrix());

        glBindVertexArray(VAO_r);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);    
    }

    glDisable(GL_BLEND);
}

bool
GraphRenderer::is_vert_pressed(const View &view, glm::vec2 mouse)
{
    float radius = vertex_size;

    for (size_t i = 0; i < vertices.size(); ++i) {
        glm::vec2 vert = view.screen_coords(vertices[i]);

        if ((mouse.x >= roundf(vert.x) - radius) &&
            (mouse.x <= roundf(vert.x) + radius) &&
            (mouse.y >= roundf(vert.y) - radius) &&
            (mouse.y <= roundf(vert.y) + radius)) {
            vert_ind = i;
            return true;
        }
    }

    return false;
}

bool GraphRenderer::is_rect_pressed(glm::vec2 mouse_pos)
{
    float mouse_x = mouse_pos.x;
    float mouse_y = mouse_pos.y;

    float min_x = std::min(rect_vtxs[0].x, rect_vtxs[3].x);
    float min_y = std::min(rect_vtxs[0].y, rect_vtxs[1].y);
    float max_x = std::max(rect_vtxs[0].x, rect_vtxs[3].x);
    float max_y = std::max(rect_vtxs[0].y, rect_vtxs[1].y);

    if((mouse_x >= min_x && mouse_x <= max_x) && (mouse_y >= min_y && mouse_y <= max_y))
        rect_pressed = true;
    else 
        rect_pressed = false;

    max_diff_x = max_x - mouse_x;
    min_diff_x = mouse_x - min_x;
    max_diff_y = max_y - mouse_y;
    min_diff_y = mouse_y - min_y;

    return rect_pressed;
}

void GraphRenderer::set_rect_start_point(glm::vec2 mouse_pos)
{
    draw_rect = update_rect_pos = true;

    rect_vtxs[0] = mouse_pos;
    rect_vtxs[1] = mouse_pos;
    rect_vtxs[2] = mouse_pos;
    rect_vtxs[3] = mouse_pos;
}

void GraphRenderer::set_rect_end_point(glm::vec2 mouse_pos)
{
    float delta_x = mouse_pos.x - rect_vtxs[3].x;
    float delta_y = mouse_pos.y - rect_vtxs[3].y;
    rect_vtxs[0] = {rect_vtxs[3].x + delta_x, rect_vtxs[3].y};
    rect_vtxs[1] = {rect_vtxs[3].x + delta_x, rect_vtxs[3].y + delta_y};
    rect_vtxs[2] = {rect_vtxs[3].x, rect_vtxs[3].y + delta_y};    

    selected_landmarks.clear();

    for (size_t i = 0; i < vertices.size(); ++i)
    {
        if(is_within_rect(vertices[i]))
            selected_landmarks.emplace_back(i);
    }
}

void GraphRenderer::move_selection(glm::vec2 mouse_pos, LandmarkModel& landmarks)
{
    float x = mouse_pos.x;
    float y = mouse_pos.y;
    
    auto new_upper_r = glm::vec2(x + max_diff_x, y + max_diff_y);
    auto new_bottom_r = glm::vec2(x + max_diff_x, y - min_diff_y);
    auto new_bottom_l = glm::vec2(x - min_diff_x, y - min_diff_y);
    auto new_upper_l = glm::vec2(x - min_diff_x, y + max_diff_y);

    float delta_x = new_upper_r.x - rect_vtxs[0].x;
    float delta_y = new_upper_r.y - rect_vtxs[0].y;

    for (size_t i = 0; i < selected_landmarks.size(); ++i)
    {
        size_t ind = selected_landmarks[i];
        auto old_pos = landmarks.lodim_vertices[ind];
        glm::vec2 new_pos = old_pos + glm::vec2(delta_x, delta_y);
        landmarks.move(ind, new_pos);
    }
    
    rect_vtxs[0] = new_upper_r;
    rect_vtxs[1] = new_bottom_r;
    rect_vtxs[2] = new_bottom_l;
    rect_vtxs[3] = new_upper_l;
}

void
GraphRenderer::prepare_data(float current_zoom, const LandmarkModel &model)
{
    prepare_vertices(current_zoom, model);
    prepare_edges(model);
    prepare_rectangle();
}

void
GraphRenderer::add_circle(float middle_x,
                          float middle_y,
                          float zoom,
                          std::vector<float> &all_vtxs)
{
    int sides = 12;
    float radius = 0.05f;
    num_all_vtxs = sides + 2;

    double two_pi = 2.0f * M_PI;

    all_vtxs.emplace_back(middle_x);
    all_vtxs.emplace_back(middle_y);

    for (int i = 1; i < num_all_vtxs; i++) {
        all_vtxs.emplace_back(middle_x +
                              (radius * cos(i * two_pi / sides)) * zoom * 130);
        all_vtxs.emplace_back(middle_y +
                              (radius * sin(i * two_pi / sides)) * zoom * 130);
    }
}

void
GraphRenderer::prepare_vertices(float current_zoom, const LandmarkModel &model)
{
    if (vertices.size() != model.lodim_vertices.size()) {
        vertices.clear();
        vertices.resize(model.lodim_vertices.size());
    }

    std::vector<float> all_vtxs;

    for (size_t i = 0; i < vertices.size(); ++i) {
        vertices[i] = model.lodim_vertices[i];
        add_circle(vertices[i].x, vertices[i].y, current_zoom, all_vtxs);
    }

    glBindVertexArray(VAO_v);

    glBindBuffer(GL_ARRAY_BUFFER, VBO_v);
    glBufferData(GL_ARRAY_BUFFER,
                 all_vtxs.size() * sizeof(float),
                 &all_vtxs[0],
                 GL_DYNAMIC_DRAW);
    glVertexAttribPointer(
      0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);
}

void
GraphRenderer::prepare_edges(const LandmarkModel &model)
{
    std::vector<glm::vec2> edge_lines(2 * model.edges.size());
    for (size_t i = 0; i < model.edges.size(); ++i) {
        edge_lines[2 * i + 0] = vertices[model.edges[i].first];
        edge_lines[2 * i + 1] = vertices[model.edges[i].second];
    }

    glBindVertexArray(VAO_e);

    glBindBuffer(GL_ARRAY_BUFFER, VBO_e);
    glBufferData(GL_ARRAY_BUFFER,
                 edge_lines.size() * sizeof(glm::vec2),
                 &edge_lines[0],
                 GL_DYNAMIC_DRAW);
    glVertexAttribPointer(
      0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void *)0);
    glEnableVertexAttribArray(0);
}

void GraphRenderer::prepare_rectangle()
{
    glBindVertexArray(VAO_r);

    glBindBuffer(GL_ARRAY_BUFFER, VBO_r);
    glBufferData(GL_ARRAY_BUFFER, rect_vtxs.size() * sizeof(glm::vec2), &rect_vtxs[0], GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO_r);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, rect_indices.size() * sizeof(unsigned int), &rect_indices[0], GL_STATIC_DRAW);

    glVertexAttribPointer(
      0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void *)0);
    glEnableVertexAttribArray(0);
}

bool GraphRenderer::is_within_rect(glm::vec2 point) const
{
    float point_x = point.x;
    float point_y = point.y;

    float min_x = std::min(rect_vtxs[0].x, rect_vtxs[3].x);
    float min_y = std::min(rect_vtxs[0].y, rect_vtxs[1].y);
    float max_x = std::max(rect_vtxs[0].x, rect_vtxs[3].x);
    float max_y = std::max(rect_vtxs[0].y, rect_vtxs[1].y);

    if((point_x >= min_x && point_x <= max_x) && (point_y >= min_y && point_y <= max_y))
        return true; 
    else
        return false;
}
