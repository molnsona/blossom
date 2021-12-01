
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
    enum Coloring
    {
        EXPR,
        CLUSTER
    };

    std::vector<Magnum::Color4> data;
    int coloring;
    int color;
    std::string col_palette;
    int cluster_cnt;
    float alpha;
    bool reverse;

    ColorData()
      : coloring((int)Coloring::EXPR)
      , color(0)
      , col_palette("rdbu")
      , cluster_cnt(10)
      , alpha(0.5f)
      , reverse(false)
    {}

    void update(const TransData &td);
    void touch_config() { refresh(data.size()); }
    void reset();
};

#endif
