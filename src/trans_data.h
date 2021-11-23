#ifndef TRANS_DATA_H
#define TRANS_DATA_H

#include <thread>
#include <vector>

#include "data_model.h"
#include "ui_trans_data.h"

struct TransConfig
{
    bool included;              // do not touch this directly
    std::string transformation; // e.g., "" or "asinh" etc.
    float cofactor;
    bool scale;
    float sdev;

    // TODO observed means+sdevs

    TransConfig()
    {
        included = true;
        cofactor = 500;
        scale = false;
        sdev = 1;
    }
};

struct TransData
{
    size_t n, d;
    std::vector<float> data;
    std::vector<TransConfig> config;
    size_t dirty;

    TransData()
      : n(0)
      , d(0)
    {}

    void update(const DataModel &d);
    void reset(const DataModel &d);

    // UI interface. config can be touched directly except for adding/removing
    // cols. After touching the config, call touch() to cause (gradual)
    // recomputation.
    void disable_col(size_t);
    void enable_col(size_t);
    void touch();
};

#endif // #ifndef TRANS_DATA_H
