
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
#ifndef ENABLE_CUDA
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

#else // ENABLE_CUDA
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
    // TODO we just refresh the whole thing for now
    clean_range(d, rn);

    // TODO call these methods scarcely
    embedsom_cuda.setDim(d.dim());
    embedsom_cuda.setK(tc.topn);

    embedsom_cuda.setPoints(d.n, d.data.data());
    embedsom_cuda.setLandmarks(lm.lodim_vertices.size(),
                               lm.hidim_vertices.data(),
                               lm.lodim_vertices[0].data());

    // TODO this is the only thing that needs to be called repeatedly
    embedsom_cuda.embedsom(tc.boost, tc.adjust, points[0].data());

#if 0
    static std::size_t counter = 0;

    if (++counter >= 10) {
        std::cout << esom_cuda.getAvgPointsUploadTime() << "ms \t"
                  << esom_cuda.getAvgLandmarksUploadTime() << "ms \t"
                  << esom_cuda.getAvgProcessingTime() << "ms" << std::endl;
    }
    counter %= 10;
#endif

#endif
}
