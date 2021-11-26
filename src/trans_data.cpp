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
TransData::update(const DataModel &dm, const RawDataStats &s)
{
    if (dirty(dm)) {
        config.resize(dm.d);
        n = dm.n;
        reset();
        data.clear(); // TODO this needs to be updated if rolling stats should
                      // work
        data.resize(n * dim(), 0);
        sums.clear();
        sums.resize(dim(), 0);
        sqsums = sums;
        touch();
        clean(dm);
    }

    if (stat_watch.dirty(s)) {
        refresh(dm);
        stat_watch.clean(s);
    }

    // make sure we're the right size
    auto [ri, rn] = dirty_range(dm);
    if (!rn)
        return;
    // TODO: make this constant configurable (and much bigger)
    if (rn > 10000)
        rn = 10000;

    clean_range(dm, rn);
    const size_t d = dim();
    std::vector<float> sums_adjust(d, 0), sqsums_adjust(d, 0);

    for (; rn-- > 0; ++ri) {
        if (ri >= n)
            ri = 0;
        for (size_t di = 0; di < d; ++di) {
            const auto &c = config[di];

            float tmp = data[ri * d + di];
            sums_adjust[di] -= tmp;
            sqsums_adjust[di] -= tmp * tmp;
            tmp = dm.data[ri * d + di];

            // TODO if(c.zscale) ...
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
