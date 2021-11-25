#include "trans_data.h"
#include <cmath>

void
DataStats::update(const DataModel &dm)
{
    if (!dirty(dm))
        return;

    means.resize(dm.d);
    std::fill(means.begin(), means.end(), 0);
    isds = means;

    size_t d = dm.d;
    for (size_t ni = 0; ni < dm.n; ++ni) {
        for (size_t di = 0; di < d; ++di) {
            float tmp = dm.data[ni * d + di];
            means[di] += tmp;
            isds[di] += tmp * tmp;
        }
    }

    for (size_t di = 0; di < d; ++di) {
        means[di] /= dm.n;
        isds[di] /= dm.n;
        isds[di] = sqrt(isds[di] - means[di]);
        if (isds[di] < 0.0001)
            isds[di] = 0.0001;
    }

    clean(dm);
    touch();
}

void
TransData::update(const DataModel &dm, const DataStats &s)
{
    if (dirty(dm)) {
        config.resize(dm.d);
        n = dm.n;
        reset();
        data.resize(n * dm.d);
        touch();
        clean(dm);
    }

    // TODO wrong
    if (stat_watch.dirty(s)) {
        refresh(dm);
        stat_watch.clean(s);
    }

    // make sure we're the right size
    auto [ri, rn] = dirty_range(dm);
    if (!rn)
        return;
    // TODO: make this constant configurable (and much bigger)
    if (rn > 100)
        rn = 100;

    clean_range(dm, rn);
    size_t d = dim();
    for (; rn-- > 0; ++ri) {
        if (ri > n)
            ri = 0;
        for (size_t di = 0; di < d; ++di) {
            data[ri * d + di] =
              (dm.data[ri * d + di] - s.means[di]) / s.isds[di];
        }
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
