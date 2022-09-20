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

#include "color_data.h"
#include "pnorm.h"
#include "vendor/colormap/palettes.hpp"

#include <cmath>
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
ColorData::update(const TransData &td)
{
    if (td.n != data.size()) {
        data.resize(td.n, glm::vec4(0, 0, 0, 0));
        refresh(td);
    }

    const size_t max_points =
#ifndef ENABLE_CUDA
      1000
#else
      50000
#endif
      ;

    auto [ri, rn] = dirty_range(td);
    if (!rn)
        return;
    if (rn > max_points)
        rn = max_points;

    size_t n = td.n;
    size_t d = td.dim();

    clean_range(td, rn);
    switch (coloring) {
        case int(ColorData::Coloring::EXPR): {
            if (expr_col >= d)
                expr_col = 0;

            auto pal = colormap::palettes.at(col_palette).rescale(0, 1);
            float mean = td.sums[expr_col] / n;
            float sdev = sqrt(td.sqsums[expr_col] / n - mean * mean);

            for (; rn-- > 0; ++ri) {
                if (ri >= n)
                    ri = 0;
                auto c =
                  reverse
                    ? pal(1 - pnormf(td.data[ri * d + expr_col], mean, sdev))
                    : pal(pnormf(td.data[ri * d + expr_col], mean, sdev));
                data[ri] = glm::vec4(c.channels[0].val / 255.0f,
                                     c.channels[1].val / 255.0f,
                                     c.channels[2].val / 255.0f,
                                     alpha);
            }
        } break;

        case int(ColorData::Coloring::CLUSTER):
            if (cluster_col >= d)
                cluster_col = 0;

            std::vector<std::tuple<unsigned char, unsigned char, unsigned char>>
              pal;

            create_col_palette(cluster_cnt, pal);

            for (; rn-- > 0; ++ri) {
                if (ri >= n)
                    ri = 0;

                auto cluster = td.data[ri * d + cluster_col];

                auto [r, g, b] =
                  std::isnan(cluster)
                    ? std::make_tuple<unsigned char,
                                      unsigned char,
                                      unsigned char>(128, 128, 128)
                    : pal[(int)roundf(cluster) % (cluster_cnt)];
                data[ri] = glm::vec4(r / 255.0f, g / 255.0f, b / 255.0f, alpha);
            }
            break;
    }
}

void
ColorData::reset()
{
    coloring = (int)Coloring::EXPR;
    expr_col = 0;
    col_palette = "rdbu";
    cluster_col = 0;
    cluster_cnt = 10;
    alpha = 0.5f;
    reverse = false;
    touch_config();
}
