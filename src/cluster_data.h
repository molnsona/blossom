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

#ifndef CLUSTER_DATA_H
#define CLUSTER_DATA_H

#include <glm/glm.hpp>

#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "landmark_model.h"
#include "trans_data.h"

/**
 * @brief Storage of data used for cluster coloring.
 *
 */
struct ClusterData
{
    const glm::vec3 default_cluster_color = { 17.0f / 255.0f,
                                              170.0f / 255.0f,
                                              222.0f / 255.0f };

    /**  Index of the column used in cluster coloring. */
    int cluster_col;
    /** Count of the clusters used in cluster coloring. */
    int cluster_cnt;

    /** Cluster colors and names for brushing, with id of cluster as a key.
     * <cluster id, <color, name>>*/
    std::map<int, std::pair<glm::vec3, std::string>> clusters;

    /** Index of the active cluster (into @ref clusters) that is used for
     * brushing.*/
    int active_cluster;

    /** Last used id, new cluster will get this value plus one.*/
    int last_id;

    /** Size of the brushing radius circle for mouse.*/
    float radius_size;

    void do_cluster_coloring(float alpha,
                             size_t ri,
                             size_t rn,
                             const TransData &td,
                             std::vector<glm::vec4> &point_colors);
    void do_brushing(
      float alpha,
      const std::vector<std::pair<const glm::vec3 *, int>> &landmark_colors,
      const LandmarkModel &lm,
      size_t ri,
      size_t rn,
      const TransData &td,
      std::vector<glm::vec4> &point_colors);

    void add_cluster();

    void reset();
};

#endif
