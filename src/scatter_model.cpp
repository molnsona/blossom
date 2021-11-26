
#include "scatter_model.h"
#include "embedsom.h"

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

    clean_range(d, rn);

    auto do_embedsom = [&](size_t from, size_t n) {
        embedsom(n,
                 lm.n_landmarks(),
                 d.dim(), // should be the same as landmarks.d
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
