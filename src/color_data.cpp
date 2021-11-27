
#include "color_data.h"
#include "extern/colormap/palettes.hpp"
#include "pnorm.h"

#include <cmath>

void
ColorData::update(const TransData &td)
{
    if (td.n != data.size()) {
        data.resize(td.n, Magnum::Color3(0, 0, 0));
        refresh(td);
    }

    auto [ri, rn] = dirty_range(td);
    if (!rn)
        return;
    if (rn > 1000)
        rn = 1000;

    size_t n = td.n;
    size_t d = td.dim();
    if (color >= d)
        return;

    clean_range(td, rn);
    auto pal = colormap::palettes.at(col_pallette).rescale(0, 1);
    float mean = td.sums[color] / n;
    float sdev = sqrt(td.sqsums[color] / n - mean * mean);

    for (; rn-- > 0; ++ri) {
        if (ri >= n)
            ri = 0;
        auto c = pal(pnormf(td.data[ri * d + color], mean, sdev));
        data[ri] = Magnum::Color3(c.channels[0].val / 255.0f,
                                  c.channels[1].val / 255.0f,
                                  c.channels[2].val / 255.0f);
    }
}
