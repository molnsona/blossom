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

#ifndef UI_RENDERER_H
#define UI_RENDERER_H

#include <glm/glm.hpp>

#include <array>

#include "landmark_model.h"
#include "shader.h"
#include "view.h"

/**
 * @brief Renderer of the objects that are needed in the user
 * interaction with the dataset.
 *
 */
struct UiRenderer
{
    bool draw_rect;
    bool update_rect_pos;

    bool rect_pressed;

    /** If the brushing is active.*/
    bool is_brushing_active;

    bool draw_circle;

    UiRenderer();
    bool init();

    void draw(const View &v);

    bool is_rect_pressed(glm::vec2 mouse_pos);

    void set_rect_start_point(glm::vec2 mouse_pos);
    void set_rect_end_point(glm::vec2 mouse_pos, const LandmarkModel &model);

    void move_selection(glm::vec2 mouse_pos, LandmarkModel &landmarks);

    void should_draw_circle(const View &view, glm::vec2 mouse_pos, float r);
    bool is_within_circle(const glm::vec2 &vert,
                          const glm::vec2 &pos,
                          float radius);

private:
    Shader shader_r;
    unsigned int VAO_r;
    unsigned int VBO_r;
    unsigned int EBO_r;

    int num_all_vtxs_circle;
    Shader shader_c;
    unsigned int VAO_c;
    unsigned int VBO_c;

    std::array<glm::vec2, 4> rect_vtxs;
    const std::array<unsigned int, 6> rect_indices;

    float max_diff_x;
    float min_diff_x;
    float max_diff_y;
    float min_diff_y;

    std::vector<size_t> selected_landmarks;

    glm::vec2 circle_pos;
    float circle_radius;

    void prepare_data(float current_zoom);
    /**
     * @brief Prepare rectangle data used for multiselect.
     *
     */
    void prepare_rectangle();
    void prepare_circle(float current_zoom);

    bool is_within_rect(glm::vec2 point) const;
};

#endif //#ifndef UI_RENDERER_H
