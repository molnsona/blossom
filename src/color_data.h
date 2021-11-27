
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
    std::string col_pallette;
    float alpha;

    ColorData()
      : color(0)
      , col_pallette("rdbu")
      , alpha(0.5f)
    {}

    void update(const TransData &td);
    void touch_config() { refresh(data.size()); }
};

#endif
