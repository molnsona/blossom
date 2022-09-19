/* This file is part of BlosSOM.
 *
 * Copyright (C) 2021 Mirek Kratochvil
 *                    Sona Molnarova
 *
 * BlosSOM is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option)
 * any later version.
 *
 * BlosSOM is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * BlosSOM. If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef TRANS_DATA_H
#define TRANS_DATA_H

#include <thread>
#include <vector>

#include "data_model.h"

/** Statistics from the untransformed dataset
 *
 * Pipeline part that gets preliminary statistics for use in later processes,
 * esp. transformations and parameter guessing.
 */
struct RawDataStats
  : public Cleaner
  , public Dirt
{
    /** Array containing means for each dimension. */
    std::vector<float> means;
    /** Array containing standard deviations for each dimension. */
    std::vector<float> sds;

    /**
     * @brief Recomputes the statistics if the input data changed.
     *
     * @param dm Original data parsed from the input file.
     */
    void update(const DataModel &dm);
};

/** Configuration of single-dimension transformation */
struct TransConfig
{
    float affine_adjust;
    bool asinh;
    float asinh_cofactor;

    TransConfig()
      : affine_adjust(0)
      , asinh(false)
      , asinh_cofactor(500)
    {}
};

/**
 * @brief Storage of the transformed data.
 *
 */
struct TransData
  : public Sweeper
  , public Dirts
{
    /** Transformed data in the same format as @ref DataModel::data. */
    std::vector<float> data;

    /** Array representing sums for each dimension. */
    std::vector<float> sums;
    /** Array representing square sums for each dimension. */
    std::vector<float> sqsums;

    /** Separate configurations for each dimension. */
    std::vector<TransConfig> config;

    /**
     * @brief Returns dimension of the transformed data.
     *
     * @return size_t Dimension of the transformed data.
     */
    size_t dim() const { return config.size(); }
    /**
     * @brief Notifies @ref Sweeper that the config has been modified and that
     * the data has to be recomputed.
     *
     */
    void touch_config() { refresh(*this); }

    Cleaner stat_watch;

    /**
     * @brief Recomputes the data if any of the config has been touched.
     *
     * @param dm Original data parsed from the input file.
     * @param s Statistics from the untransformed dataset.
     */
    void update(const DataModel &dm, const RawDataStats &s);
    /**
     * @brief Resets configurations to their initial values.
     *
     */
    void reset();

    // UI interface. config can be touched directly except for adding/removing
    // cols. After touching the config, call touch() to cause (gradual)
    // recomputation.
    // TODO void disable_col(size_t);
    // TODO void enable_col(size_t);
};

#endif // #ifndef TRANS_DATA_H
