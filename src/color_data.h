
#ifndef COLOR_DATA_H
#define COLOR_DATA_H

#include "dirty.h"
#include "trans_data.h"
#include <Magnum/Magnum.h>
#include <Magnum/Math/Color.h>
#include <string>
#include <vector>

struct ColorData : public Sweeper
{
    std::vector<Magnum::Color4> data;
    int color;
    std::string col_palette;
    float alpha;
    bool reverse;

    ColorData()
      : color(0)
      , col_palette("rdbu")
      , alpha(0.5f)
      , reverse(false)
    {}

    void update(const TransData &td);
    void touch_config() { refresh(data.size()); }
};

#endif
