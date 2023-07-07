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
        frame_stats.reset(frame_stats.embedsom_t, frame_stats.embedsom_n);
        batch_size_gen.reset();
    }

    if (lm_watch.dirty(lm)) {
        refresh(d);
        lm_watch.clean(lm);
    }

    // It gives the beginning index ri of the data that should be
    // processed and the number of elements rn that should be
    // processed. The number of the elements can be zero if nothing
    // has to be recomputed.
    auto [ri, rn] = dirty_range(d);
    if (!rn) {
        frame_stats.embedsom_t = 0.00001f;
        return;
    }

    frame_stats.embedsom_n = batch_size_gen.next(frame_stats.embedsom_t,
                                                 frame_stats.embedsom_duration);
    const size_t max_points = frame_stats.embedsom_n;

    // If the number of elements that need to be recomputed is larger
    // than the maximum possible points that can be processed in this
    // frame, the number of elements lowers to this value.
    if (rn > max_points)
        rn = max_points;

    if (lm.d != d.dim()) {
        frame_stats.reset(frame_stats.embedsom_t, frame_stats.embedsom_n);
        batch_size_gen.reset();
        return;
    }

    // Say that rn data in the cache will be refreshed. Where rn is the
    // number of the data that will be refreshed.
    clean_range(d, rn);

    auto do_embedsom = [&](size_t from, size_t n) {
        frame_stats.add_const_time();

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

        frame_stats.store_time(frame_stats.embedsom_t);
    };

    // If the index in the elements is over the size of the data
    // It means it is cyclic and needs to continue from the
    // beginning of the data.
    if (ri + rn >= d.n) {
        // So firstly the elements that remain to the end of the data
        // are processed.
        size_t diff = d.n - ri;
        do_embedsom(ri, diff);
        // Then the index cycles back to the beginning
        ri = 0;
        // And the number of elements that need to be processed
        // is lowered by the already processed elements.
        rn -= diff;
    }

    // Process the elements that are left to processing.
    if (rn)
        do_embedsom(ri, rn);
}
