
#include "scaled_data.h"

#include <cmath>

void
ScaledData::update(const TransData &td)
{
    if (dirty(td) && (td.dim() != dim() || td.n != n)) {
        n = td.n;
        config.resize(td.dim());
        touch_config();
        data.resize(n * dim());
        clean(td);
    }

    auto [ri, rn] = dirty_range(td);
    if (!rn)
        return;
    if (rn > 100)
        rn = 100;
    clean_range(td, rn);

    std::vector<float> means = td.sums;
    std::vector<float> isds = td.sqsums;
    size_t d = dim();
    for (size_t di = 0; di < d; ++di) {
        means[di] /= n;
        isds[di] /= n;
        isds[di] = 1 / sqrt(isds[di] - means[di]);
        if (isds[di] > 10000)
            isds[di] = 10000;
    }

    for (; rn-- > 0; ++ri) {
        if (ri >= n)
            ri = 0;
        for (size_t di = 0; di < d; ++di)
            data[ri * d + di] =
              (td.data[ri * d + di] - means[di]) *
              (config[d].scale ? config[d].sdev * isds[di] : isds[di]);
    }
    touch();
}
