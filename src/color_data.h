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

#ifndef COLOR_DATA_H
#define COLOR_DATA_H

#include <glm/glm.hpp>

#include "batch_size_gen.h"
#include "cluster_data.h"
#include "dirty.h"
#include "landmark_model.h"
#include "frame_stats.h"
#include "trans_data.h"

#include <string>
#include <vector>

/**
 * @brief Storage of the color data.
 *
 */
struct ColorData : public Sweeper
{
    /**
     * @brief Types of coloring.
     *
     */
    enum Coloring
    {
        EXPR,
        CLUSTER,
        BRUSHING
    };

    const glm::vec3 default_landmark_color = { 0.4, 0.4, 0.4 };

    Cleaner lm_watch;

    /** Colors of the 2D data points. Array has the size of the number of 2D
     * data points.
     */
    std::vector<glm::vec4> data;
    /** Colors of the landmarks and id of the cluster. Array has the size of the
     * number of landmarks. <color, cluster id>
     */
    std::vector<std::pair<const glm::vec3 *, int>> landmarks;
    /** Type of the coloring method. */
    int coloring;
    /** Index of the column used in expression coloring. */
    int expr_col;
    /** Name of the currently used color palette. */
    std::string col_palette;

    ClusterData clustering;
    /** Alpha channel of RGBA color. It is the same for all 2D data points. */
    float alpha;
    /** Flag indicating if the colors of the color palette should be reversed.
     */
    bool reverse;

    BatchSizeGen batch_size_gen;

    /**
     * @brief Calls @ref reset() method to set initial values.
     *
     */
    ColorData() { reset(); }

    /**
     * @brief Recomputes color of the 2D data points if user has changed any of
     * the color settings.
     *
     * @param td Transformed data received from the data flow pipeline.
     */
    void update(const TransData &td, const LandmarkModel &lm, FrameStats &frame_stats);    
    /**
     * @brief Notifies @ref Sweeper that the color settings has been modified
     * and that the data has to be recomputed.
     *
     */

    /**
     * @brief Color landmarks by active cluster.
     *
     * @param idxs Ids of landmarks that will be colored.
     */
    void color_landmarks(const std::vector<size_t> &idxs);

    /**
     * @brief Reset colors and cluster ids of all landmarks
     * in the cluster with input id.
     *
     * @param id Input
     */
    void reset_landmark_color(int id);

    void remove_landmark(size_t ind);

    void touch_config() { refresh(data.size()); }
    /**
     * @brief Resets color settings to their initial values.
     *
     */
    void reset();

private:
    /** Color the landmark according to the active cluster.*/
    void color_landmark(size_t ind);
};

#endif
