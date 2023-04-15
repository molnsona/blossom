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

void
ColorData::update(const TransData &td, const LandmarkModel &lm)
{
    if (td.n != data.size()) {
        data.resize(td.n, glm::vec4(0, 0, 0, 0));
        refresh(td);
    }

    if (lm_watch.dirty(lm)) {
        landmarks.resize(lm.n_landmarks(), {0.4,0.4,0.4,0.6});
        refresh(td);
        lm_watch.clean(lm);
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
            cluster_coloring.do_cluster_coloring(alpha, ri, rn, td, data);
            break;
        case int(ColorData::Coloring::BRUSHING):

            break;
    }
}

void
ColorData::reset()
{
    coloring = (int)Coloring::EXPR;
    expr_col = 0;
    col_palette = "rdbu";
    alpha = 0.5f;
    reverse = false;
    cluster_coloring.reset();
    touch_config();
}
