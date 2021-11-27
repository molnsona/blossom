#ifndef SCALED_DATA_H
#define SCALED_DATA_H

#include "trans_data.h"
#include <vector>

struct ScaleConfig
{
    bool scale;
    float sdev;

    ScaleConfig()
      : scale(false)
      , sdev(1)
    {}
};

struct ScaledData
  : public Sweeper
  , public Dirts
{

    std::vector<float> data;
    std::vector<ScaleConfig> config;

    size_t dim() const { return config.size(); }

    void touch_config() { refresh(*this); }

    void update(const TransData &td);
    void reset();
};

#endif
