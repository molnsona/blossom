
#include "color_data.h"
#include "pnorm.h"
#include "vendor/colormap/palettes.hpp"

#include <cmath>
#include <tuple>
#include <vector>

static std::tuple<uint8_t, uint8_t, uint8_t>
hsv2rgb(float h, float s, float v)
{
    float chroma = v * s;
    float m = v - chroma;
    h *= 6;
    int hi = truncf(h);
    float rest = h - hi;
    hi %= 6;

    float rgb[3] = { 0, 0, 0 };
    switch (hi) {
        case 0:
            rgb[0] = 1;
            rgb[1] = rest;
            break;
        case 1:
            rgb[1] = 1;
            rgb[0] = 1 - rest;
            break;
        case 2:
            rgb[1] = 1;
            rgb[2] = rest;
            break;
        case 3:
            rgb[2] = 1;
            rgb[1] = 1 - rest;
            break;
        case 4:
            rgb[2] = 1;
            rgb[0] = rest;
            break;
        case 5:
        default:
            rgb[0] = 1;
            rgb[2] = 1 - rest;
            break;
    }

    for (size_t i = 0; i < 3; ++i)
        rgb[i] = (chroma * rgb[i] + m) * 255;
    return { rgb[0], rgb[1], rgb[2] };
}

// Zeroth color is gray - represents points in no cluster.
static void
create_col_palette(
  std::vector<std::tuple<unsigned char, unsigned char, unsigned char>>
    &color_palette,
  size_t clusters)
{
    color_palette.resize(clusters);
    for (size_t i = 0; i < clusters; ++i)
        color_palette[i] = hsv2rgb(
          float(i) / (clusters), i % 2 ? 1.0f : 0.7f, i % 2 ? 0.7f : 1.0f);
}

void
ColorData::update(const TransData &td)
{
    if (td.n != data.size()) {
        data.resize(td.n, Magnum::Color4(0, 0, 0, 0));
        reset();
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
    switch (coloring) {
        case (int)ColorData::Coloring::EXPR: {
            auto pal = colormap::palettes.at(col_palette).rescale(0, 1);
            float mean = td.sums[color] / n;
            float sdev = sqrt(td.sqsums[color] / n - mean * mean);

            for (; rn-- > 0; ++ri) {
                if (ri >= n)
                    ri = 0;
                auto c =
                  reverse ? pal(1 - pnormf(td.data[ri * d + color], mean, sdev))
                          : pal(pnormf(td.data[ri * d + color], mean, sdev));
                data[ri] = Magnum::Color4(c.channels[0].val / 255.0f,
                                          c.channels[1].val / 255.0f,
                                          c.channels[2].val / 255.0f,
                                          alpha);
            }
        } break;

        case int(ColorData::Coloring::CLUSTER):
            std::vector<std::tuple<unsigned char, unsigned char, unsigned char>>
              pal;

            create_col_palette(pal, cluster_cnt);

            for (; rn-- > 0; ++ri) {
                if (ri >= n)
                    ri = 0;

                // I suppose the last column is "label" column.
                auto cluster = td.data[ri * d + d - 1];

                auto [r, g, b] =
                  std::isnan(cluster)
                    ? std::make_tuple<unsigned char,
                                      unsigned char,
                                      unsigned char>(128, 128, 128)
                    : pal[(int)roundf(cluster) % (cluster_cnt)];
                data[ri] =
                  Magnum::Color4(r / 255.0f, g / 255.0f, b / 255.0f, alpha);
            }
            break;
    }
}

void
ColorData::reset()
{
    coloring = (int)Coloring::EXPR;
    color = 0;
    col_palette = "rdbu";
    cluster_cnt = 10;
    alpha = 0.5f;
    reverse = false;
}
