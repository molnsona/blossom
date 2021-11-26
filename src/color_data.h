
#ifndef COLOR_DATA_H
#define COLOR_DATA_H

#include "dirty.h"
#include "trans_data.h"
#include <Magnum/Magnum.h>
#include <Magnum/Math/Color.h>
#include <vector>

struct ColorData : public Sweeper
{
    std::vector<Magnum::Color3> data;
    int color;

    ColorData()
      : color(0)
    {}

    void update(const TransData &td);
    void touch_config() { refresh(data.size()); }
};

#endif
