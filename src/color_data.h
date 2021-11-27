
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
    std::vector<Magnum::Color3> data;
    int color;
    std::string col_pallette;

    ColorData()
      : color(0)
      , col_pallette("rdbu")
    {}

    void update(const TransData &td);
    void touch_config() { refresh(data.size()); }
};

#endif
