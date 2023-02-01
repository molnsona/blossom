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

#include "dirty.h"
#include "frame_stats.h"
#include "normal_gen.h"
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
        CLUSTER
    };

    /** Colors of the 2D data points. Array has the size of number of 2D data
     * points.
     */
    std::vector<glm::vec4> data;
    /** Type of the coloring method. */
    int coloring;
    /** Index of the column used in expression coloring. */
    int expr_col;
    /** Name of the currently used color palette. */
    std::string col_palette;
    /**  Index of the column used in cluster coloring. */
    int cluster_col;
    /** Count of the clusters used in cluster coloring. */
    int cluster_cnt;
    /** Alpha channel of RGBA color. It is the same for all 2D data points. */
    float alpha;
    /** Flag indicating if the colors of the color palette should be reversed.
     */
    bool reverse;

    NormalGen gen;

    /**
     * @brief Calls @ref reset() method to set initial values.
     *
     */
    ColorData()
      :
#ifndef ENABLE_CUDA
      gen(750, 250) // 500 -- 1000
#else
      gen(37500, 12500) // 25k -- 50k
#endif
    {
        reset();
    }

    /**
     * @brief Recomputes color of the 2D data points if user has changed any of
     * the color settings.
     *
     * @param td Transformed data received from the data flow pipeline.
     */
    void update(const TransData &td, FrameStats &frame_stats);
    /**
     * @brief Notifies @ref Sweeper that the color settings has been modified
     * and that the data has to be recomputed.
     *
     */
    void touch_config()
    {
        refresh(data.size());
    }
    /**
     * @brief Resets color settings to their initial values.
     *
     */
    void reset();
};

#endif
