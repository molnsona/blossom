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

#include "trans_data.h"
#include <cmath>

void
RawDataStats::update(const DataModel &dm)
{
    if (!dirty(dm))
        return;

    means.clear();
    means.resize(dm.d, 0);
    sds = means;

    size_t d = dm.d;
    for (size_t ni = 0; ni < dm.n; ++ni) {
        for (size_t di = 0; di < d; ++di) {
            float tmp = dm.data[ni * d + di];
            means[di] += tmp;
            sds[di] += tmp * tmp;
        }
    }

    for (size_t di = 0; di < d; ++di) {
        means[di] /= dm.n;
        sds[di] /= dm.n;
        sds[di] = sqrt(sds[di] - means[di] * means[di]);
    }

    clean(dm);
    touch();
}

void
TransData::update(const DataModel &dm,
                  const RawDataStats &s,
                  FrameStats &frame_stats)
{
    if (dirty(dm)) {
        config.resize(dm.d);
        n = dm.n;
        data.clear(); // TODO this needs to be updated if rolling stats should
                      // work
        data.resize(n * dim(), 0);
        sums.clear();
        sums.resize(dim(), 0);
        sqsums = sums;
        touch();
        clean(dm);
        frame_stats.reset(frame_stats.trans_t);
        batch_size_gen.reset();
    }

    if (stat_watch.dirty(s)) {
        refresh(dm);
        stat_watch.clean(s);
    }

    // make sure we're the right size
    auto [ri, rn] = dirty_range(dm);
    if (!rn) {
        frame_stats.reset(frame_stats.trans_t);
        return;
    }
      
    const size_t max_points = batch_size_gen.next(frame_stats.trans_t, frame_stats.trans_duration);

    if (rn > max_points)
        rn = max_points;

    clean_range(dm, rn);
    const size_t d = dim();
    std::vector<float> sums_adjust(d, 0), sqsums_adjust(d, 0);

    frame_stats.add_const_time();

    for (; rn-- > 0; ++ri) {
        if (ri >= n)
            ri = 0;
        for (size_t di = 0; di < d; ++di) {
            const auto &c = config[di];

            float tmp = data[ri * d + di];
            sums_adjust[di] -= tmp;
            sqsums_adjust[di] -= tmp * tmp;
            tmp = dm.data[ri * d + di];

            tmp += c.affine_adjust;
            if (c.asinh)
                tmp = asinhf(tmp / c.asinh_cofactor);

            data[ri * d + di] = tmp;
            sums_adjust[di] += tmp;
            sqsums_adjust[di] += tmp * tmp;
        }
    }

    for (size_t di = 0; di < d; ++di) {
        sums[di] += sums_adjust[di];
        sqsums[di] += sqsums_adjust[di];
    }

    frame_stats.store_time(frame_stats.trans_t);

    touch();
}

void
TransData::reset()
{
    for (auto &c : config)
        c = TransConfig();
    touch_config();
}

#if 0
void
TransData::disable_col(size_t c)
{
    // TODO update config, remove the column from output if needed, reduce `d`,
    // ...
}

void
TransData::enable_col(size_t c)
{
    // TODO reverse of disable_col
}
#endif
