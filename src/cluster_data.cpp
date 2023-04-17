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

#include "cluster_data.h"

#include <cmath>
#include <limits>
#include <tuple>
#include <vector>

/**
 * @brief Converts hsv color to rgb color system.
 *
 * @param h Hue
 * @param s Saturation
 * @param v Value
 * @return std::tuple<uint8_t, uint8_t, uint8_t> Color in the RGB color system.
 */
static std::tuple<uint8_t, uint8_t, uint8_t>
hsv2rgb(float h, float s, float v)
{
    float chroma = v * s;
    float m = v - chroma;
    h *= 6;
    int hi = truncf(h);
    float rest = h - hi;
    hi %= 6;

    float rgb[3] = { 0, 0, 0 };
    switch (hi) {
        case 0:
            rgb[0] = 1;
            rgb[1] = rest;
            break;
        case 1:
            rgb[1] = 1;
            rgb[0] = 1 - rest;
            break;
        case 2:
            rgb[1] = 1;
            rgb[2] = rest;
            break;
        case 3:
            rgb[2] = 1;
            rgb[1] = 1 - rest;
            break;
        case 4:
            rgb[2] = 1;
            rgb[0] = rest;
            break;
        case 5:
        default:
            rgb[0] = 1;
            rgb[2] = 1 - rest;
            break;
    }

    for (size_t i = 0; i < 3; ++i)
        rgb[i] = (chroma * rgb[i] + m) * 255;
    return { rgb[0], rgb[1], rgb[2] };
}

/**
 * @brief Creates color palette with the size of the cluster count.
 *
 * @param[in] clusters The number of colors used in the new color palette.
 * @param[out] color_palette Created color palette.
 */
static void
create_col_palette(
  size_t clusters,
  std::vector<std::tuple<unsigned char, unsigned char, unsigned char>>
    &color_palette)
{
    color_palette.resize(clusters);
    for (size_t i = 0; i < clusters; ++i)
        color_palette[i] = hsv2rgb(
          float(i) / (clusters), i % 2 ? 1.0f : 0.7f, i % 2 ? 0.7f : 1.0f);
}

void
ClusterData::do_cluster_coloring(float alpha,
                                 size_t ri,
                                 size_t rn,
                                 const TransData &td,
                                 std::vector<glm::vec4> &point_colors)
{
    size_t n = td.n;
    size_t d = td.dim();

    if (cluster_col >= d)
        cluster_col = 0;

    std::vector<std::tuple<unsigned char, unsigned char, unsigned char>> pal;

    create_col_palette(cluster_cnt, pal);

    for (; rn-- > 0; ++ri) {
        if (ri >= n)
            ri = 0;

        auto cluster = td.data[ri * d + cluster_col];

        auto [r, g, b] =
          std::isnan(cluster)
            ? std::make_tuple<unsigned char, unsigned char, unsigned char>(
                128, 128, 128)
            : pal[(int)roundf(cluster) % (cluster_cnt)];
        point_colors[ri] = glm::vec4(r / 255.0f, g / 255.0f, b / 255.0f, alpha);
    }
}

void
ClusterData::do_brushing(
  float alpha,
  const std::vector<std::pair<const glm::vec3 *, int>> &landmark_colors,
  const LandmarkModel &lm,
  size_t ri,
  size_t rn,
  const TransData &td,
  std::vector<glm::vec4> &point_colors)
{
    size_t n = td.n;
    size_t d = td.dim();

    // Loop in a cycle of data.
    // Compute the closest landmark for each data point
    // and paint the point according the color of the
    // landmark.
    for (; rn-- > 0; ++ri) {
        if (ri >= n)
            ri = 0;

        // Find the closest landmark.
        size_t best = 0;
        float best_sqdist = std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < lm.n_landmarks(); ++i) {
            float sqd = 0;
            for (size_t di = 0; di < d; ++di)
                sqd +=
                  pow(lm.hidim_vertices[i * d + di] - td.data[ri * d + di], 2);

            if (sqd < best_sqdist) {
                best = i;
                best_sqdist = sqd;
            }
        }

        // Color the point with the color of the landmark.
        point_colors[ri] = glm::vec4(*landmark_colors[best].first, alpha);
    }
}

void
ClusterData::add_cluster()
{
    clusters[++last_id] = std::make_pair(default_cluster_color, "cluster name");
}

void
ClusterData::reset()
{
    cluster_col = 0;
    cluster_cnt = 10;

    active_cluster = -1;
    last_id = -1;

    radius_size = 40.0f;

    clusters.clear();
}
