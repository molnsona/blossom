#ifndef TRANS_DATA_H
#define TRANS_DATA_H

#include <thread>
#include <vector>

#include "data_model.h"
#include "ui_trans_data.h"

/** Statistics from the untransformed dataset
 *
 * Pipeline part that gets preliminary statistics for use in later processes,
 * esp. transformations and parameter guessing.
 */
struct RawDataStats
  : public Cleaner
  , public Dirt
{
    std::vector<float> means, sds;

    void update(const DataModel &dm);
};

/** Configuration of single-dimension transformation */
struct TransConfig
{
    bool zscale; // TODO implement
    float affine_adjust;
    bool asinh;
    float asinh_cofactor;

    TransConfig()
      : zscale(false)
      , affine_adjust(0)
      , asinh(false)
      , asinh_cofactor(500)
    {}
};

struct TransData
  : public Sweeper
  , public Dirts
{
    std::vector<float> data;

    std::vector<float> sums, sqsums;

    std::vector<TransConfig> config;
    size_t dim() const { return config.size(); }
    void touch_config() { refresh(*this); }

    Cleaner stat_watch;
    void update(const DataModel &dm, const RawDataStats &s);
    void reset();

    // UI interface. config can be touched directly except for adding/removing
    // cols. After touching the config, call touch() to cause (gradual)
    // recomputation.
    // TODO void disable_col(size_t);
    // TODO void enable_col(size_t);
};

#endif // #ifndef TRANS_DATA_H
