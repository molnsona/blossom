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

#include "ui_renderer.h"

#include "shaders.h"

UiRenderer::UiRenderer()
    : rect_indices({ 0, 1, 3, 1, 2, 3 })
    , draw_rect(false)
    , update_rect_pos(false)
    , rect_pressed(false)
    , is_brushing_active(false)
    , draw_circle(false)
{

}

bool UiRenderer::init()
{
    glGenVertexArrays(1, &VAO_r);
    glGenBuffers(1, &VBO_r);
    glGenBuffers(1, &EBO_r);

    shader_r.build(ui_r_vs, ui_r_fs);

    glGenVertexArrays(1, &VAO_c);
    glGenBuffers(1, &VBO_c);    

    shader_c.build(ui_c_vs, ui_c_fs);
}

void UiRenderer::draw(const View &view)
{
    glEnable(GL_BLEND);

    prepare_data(view.current_zoom);

    if (draw_rect) {
        shader_r.use();
        shader_r.set_mat4("model", glm::mat4(1.0f));
        shader_r.set_mat4("view", view.get_view_matrix());
        shader_r.set_mat4("proj", view.get_proj_matrix());

        glBindVertexArray(VAO_r);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    }

    if(draw_circle)
    {
        shader_c.use();
        shader_c.set_mat4("model", glm::mat4(1.0f));
        shader_c.set_mat4("view", view.get_view_matrix());
        shader_c.set_mat4("proj", view.get_proj_matrix());

        glBindVertexArray(VAO_c);
        glDrawArrays(GL_LINES, 0, num_all_vtxs_circle);
    }

    glDisable(GL_BLEND);
}

void
UiRenderer::prepare_data(float current_zoom)
{
    prepare_rectangle();
    prepare_circle(current_zoom);
}

bool
UiRenderer::is_rect_pressed(glm::vec2 mouse_pos)
{
    float mouse_x = mouse_pos.x;
    float mouse_y = mouse_pos.y;

    float min_x = std::min(rect_vtxs[0].x, rect_vtxs[3].x);
    float min_y = std::min(rect_vtxs[0].y, rect_vtxs[1].y);
    float max_x = std::max(rect_vtxs[0].x, rect_vtxs[3].x);
    float max_y = std::max(rect_vtxs[0].y, rect_vtxs[1].y);

    if ((mouse_x >= min_x && mouse_x <= max_x) &&
        (mouse_y >= min_y && mouse_y <= max_y))
        rect_pressed = true;
    else
        rect_pressed = false;

    max_diff_x = max_x - mouse_x;
    min_diff_x = mouse_x - min_x;
    max_diff_y = max_y - mouse_y;
    min_diff_y = mouse_y - min_y;

    return rect_pressed;
}

void
UiRenderer::set_rect_start_point(glm::vec2 mouse_pos)
{
    draw_rect = update_rect_pos = true;

    rect_vtxs[0] = mouse_pos;
    rect_vtxs[1] = mouse_pos;
    rect_vtxs[2] = mouse_pos;
    rect_vtxs[3] = mouse_pos;
}

void
UiRenderer::set_rect_end_point(glm::vec2 mouse_pos, const LandmarkModel &model)
{
    float delta_x = mouse_pos.x - rect_vtxs[3].x;
    float delta_y = mouse_pos.y - rect_vtxs[3].y;
    rect_vtxs[0] = { rect_vtxs[3].x + delta_x, rect_vtxs[3].y };
    rect_vtxs[1] = { rect_vtxs[3].x + delta_x, rect_vtxs[3].y + delta_y };
    rect_vtxs[2] = { rect_vtxs[3].x, rect_vtxs[3].y + delta_y };

    selected_landmarks.clear();

    const auto &vertices = model.lodim_vertices;
    for (size_t i = 0; i < vertices.size(); ++i) {
        if (is_within_rect(vertices[i]))
            selected_landmarks.emplace_back(i);
    }
}

void
UiRenderer::move_selection(glm::vec2 mouse_pos, LandmarkModel &landmarks)
{
    float x = mouse_pos.x;
    float y = mouse_pos.y;

    auto new_upper_r = glm::vec2(x + max_diff_x, y + max_diff_y);
    auto new_bottom_r = glm::vec2(x + max_diff_x, y - min_diff_y);
    auto new_bottom_l = glm::vec2(x - min_diff_x, y - min_diff_y);
    auto new_upper_l = glm::vec2(x - min_diff_x, y + max_diff_y);

    float delta_x = new_upper_r.x - rect_vtxs[0].x;
    float delta_y = new_upper_r.y - rect_vtxs[0].y;

    for (size_t i = 0; i < selected_landmarks.size(); ++i) {
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

void UiRenderer::should_draw_circle(const View &view, glm::vec2 mouse_pos, float r)
{
    // Convert radius from screen to model space
    // Compute mouse + radius point in model
    auto right = mouse_pos + glm::vec2(r, 0);
    auto model_right = view.model_mouse_coords(right);
    auto model_mouse = view.model_mouse_coords(mouse_pos);
    float model_radius = fabs((model_right - model_mouse).x);
    
    draw_circle = true;
    circle_pos = model_mouse;
    circle_radius = model_radius;
}

void
UiRenderer::prepare_rectangle()
{
    glBindVertexArray(VAO_r);

    glBindBuffer(GL_ARRAY_BUFFER, VBO_r);
    glBufferData(GL_ARRAY_BUFFER,
                 rect_vtxs.size() * sizeof(glm::vec2),
                 &rect_vtxs[0],
                 GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO_r);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 rect_indices.size() * sizeof(unsigned int),
                 &rect_indices[0],
                 GL_STATIC_DRAW);

    glVertexAttribPointer(
      0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void *)0);
    glEnableVertexAttribArray(0);
}

void
UiRenderer::prepare_circle(float zoom)
{
    std::vector<float> circle_vtxs;

    int sides = 24;
    num_all_vtxs_circle = sides * 2;

    double two_pi = 2.0f * M_PI;

    for (int i = 0; i < sides + 1; ++i) {
        float x_coor = 
            circle_pos.x + (circle_radius * cos(i * two_pi / sides));
        float y_coor = 
            circle_pos.y + (circle_radius * sin(i * two_pi / sides));

        circle_vtxs.emplace_back(x_coor);
        circle_vtxs.emplace_back(y_coor);

        // Add each point twice --- end of line and start
        // of next line.
        if(i != 0 && i != sides) {
            circle_vtxs.emplace_back(x_coor);
            circle_vtxs.emplace_back(y_coor);
        }   
    }

    glBindVertexArray(VAO_c);

    glBindBuffer(GL_ARRAY_BUFFER, VBO_c);
    glBufferData(GL_ARRAY_BUFFER,
                 circle_vtxs.size() * sizeof(float),
                 &circle_vtxs[0],
                 GL_DYNAMIC_DRAW);
    glVertexAttribPointer(
      0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);
}

bool
UiRenderer::is_within_rect(glm::vec2 point) const
{
    float point_x = point.x;
    float point_y = point.y;

    float min_x = std::min(rect_vtxs[0].x, rect_vtxs[3].x);
    float min_y = std::min(rect_vtxs[0].y, rect_vtxs[1].y);
    float max_x = std::max(rect_vtxs[0].x, rect_vtxs[3].x);
    float max_y = std::max(rect_vtxs[0].y, rect_vtxs[1].y);

    if ((point_x >= min_x && point_x <= max_x) &&
        (point_y >= min_y && point_y <= max_y))
        return true;
    else
        return false;
}

bool UiRenderer::is_within_circle(const glm::vec2 &vert, 
    const glm::vec2 &pos, float radius)
{
    if ((pos.x + radius >= roundf(vert.x)) &&
        (pos.x - radius <= roundf(vert.x) ) &&
        (pos.y + radius >= roundf(vert.y) ) &&
        (pos.y - radius <= roundf(vert.y) )) {
        return true;
    }
}
