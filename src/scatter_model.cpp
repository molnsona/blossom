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
                     const TrainingConfig &tc)
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
    if (!rn)
        return;
    if (rn > 1000)
        rn = 1000;

    if (lm.d != d.dim())
        return;

    clean_range(d, rn);

    auto do_embedsom = [&](size_t from, size_t n) {
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
           lm.lodim_vertices[0].data(),
           points[from].data());
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
