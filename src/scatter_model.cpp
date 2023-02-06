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

#include "scatter_model.h"
#include "embedsom.h"

#ifdef ENABLE_CUDA
#include "embedsom_cuda.h"
#endif

void
ScatterModel::update(const ScaledData &d,
                     const LandmarkModel &lm,
                     const TrainingConfig &tc,
                     FrameStats &frame_stats)
{
    if (dirty(d)) {
        points.resize(d.n);
        refresh(d);
        clean(d);
    }

    if (lm_watch.dirty(lm)) {
        refresh(d);
        lm_watch.clean(lm);
    }

    auto [ri, rn] = dirty_range(d);
    if (!rn) {
        frame_stats.embedsom_t = 0.00001f;
        frame_stats.embedsom_n = 0;
        batch_size_gen.reset();
        return;
    }

    frame_stats.embedsom_n = batch_size_gen.next(frame_stats.embedsom_t,
                                                 frame_stats.embedsom_duration);
    const size_t max_points = frame_stats.embedsom_n;

    if (rn > max_points)
        rn = max_points;

    if (lm.d != d.dim()) {
        frame_stats.embedsom_t = 0.00001f;
        frame_stats.embedsom_n = 0;
        batch_size_gen.reset();
        return;
    }

    clean_range(d, rn);

    auto do_embedsom = [&](size_t from, size_t n) {
        frame_stats.timer.tick();
        frame_stats.constant_time += frame_stats.timer.frametime * 1000;

        frame_stats.timer.tick();
#ifdef ENABLE_CUDA
        embedsom_cuda.run
#else
        embedsom
#endif
          (n,
           lm.n_landmarks(),
           d.dim(),
           tc.boost,
           tc.topn,
           tc.adjust,
           d.data.data() + d.dim() * from,
           lm.hidim_vertices.data(),
           &lm.lodim_vertices[0][0],
           &points[from][0]);
        frame_stats.timer.tick();
        frame_stats.embedsom_t =
          frame_stats.timer.frametime * 1000; // to get milliseconds

        frame_stats.timer.tick();
    };

    if (ri + rn >= d.n) {
        size_t diff = d.n - ri;
        do_embedsom(ri, diff);
        ri = 0;
        rn -= diff;
    }

    if (rn)
        do_embedsom(ri, rn);
}
