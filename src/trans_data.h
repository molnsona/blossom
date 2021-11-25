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
        scale = true;
        sdev = 1;
    }
};

struct DataStats
  : public Cleaner
  , public Dirt
{
    // TODO this needs to go _after_ transformations.
    std::vector<float> means, isds;

    void update(const DataModel &dm);
};

struct TransData
  : public Sweeper
  , public Dirts
{
    std::vector<float> data;

    std::vector<TransConfig> config;
    size_t dim() const { return config.size(); }
    void touch_config() { refresh(*this); }

    Cleaner stat_watch;
    void update(const DataModel &dm, const DataStats &s);
    void reset();

    // UI interface. config can be touched directly except for adding/removing
    // cols. After touching the config, call touch() to cause (gradual)
    // recomputation.
    void disable_col(size_t);
    void enable_col(size_t);
};

#endif // #ifndef TRANS_DATA_H
