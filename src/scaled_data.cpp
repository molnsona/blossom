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

#include "scaled_data.h"

#include <cmath>

void
ScaledData::update(const TransData &td, FrameStats &frame_stats)
{
    if (dirty(td) && (td.dim() != dim() || td.n != n)) {
        n = td.n;
        config.resize(td.dim());
        touch_config();
        data.resize(n * dim());
        clean(td);
        frame_stats.reset(frame_stats.scaled_t);
        batch_size_gen.reset();
    }

    auto [ri, rn] = dirty_range(td);
    if (!rn) {
        frame_stats.reset(frame_stats.scaled_t);
        return;
    }

    const size_t max_points = batch_size_gen.next(frame_stats.scaled_t, frame_stats.scaled_duration);

    if (rn > max_points)
        rn = max_points;
    clean_range(td, rn);

    std::vector<float> means = td.sums;
    std::vector<float> isds = td.sqsums;
    size_t d = dim();

    frame_stats.add_const_time();

    for (size_t di = 0; di < d; ++di) {
        means[di] /= n;
        isds[di] /= n;
        isds[di] = 1 / sqrt(isds[di] - means[di] * means[di]);
        if (isds[di] > 10000)
            isds[di] = 10000;
    }

    for (; rn-- > 0; ++ri) {
        if (ri >= n)
            ri = 0;
        for (size_t di = 0; di < d; ++di)
            data[ri * d + di] =
              (td.data[ri * d + di] - means[di]) *
              (config[di].scale ? config[di].sdev * isds[di] : 1);
    }

    frame_stats.store_time(frame_stats.scaled_t);

    touch();
}

void
ScaledData::reset()
{
    for (auto &c : config)
        c = ScaleConfig();
    touch_config();
}
