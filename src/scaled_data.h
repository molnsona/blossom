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

#ifndef SCALED_DATA_H
#define SCALED_DATA_H

#include "frame_stats.h"
#include "normal_gen.h"
#include "trans_data.h"
#include <vector>

/**
 * @brief Configuration of the single-dimension scaling.
 *
 */
struct ScaleConfig
{
    /** Factor of the scaling. */
    bool scale;
    /** Standard deviation. */
    float sdev;

    ScaleConfig()
      : scale(false)
      , sdev(1)
    {
    }
};

/**
 * @brief Storage of the scaled data.
 *
 */
struct ScaledData
  : public Sweeper
  , public Dirts
{
    /** Scaled data in the same format as @ref DataModel::data. */
    std::vector<float> data;
    /** Separate configurations for each dimension. */
    std::vector<ScaleConfig> config;

    NormalGen gen;

    ScaledData()
      :
#ifndef ENABLE_CUDA
      gen(7500, 2500) // 5k -- 10k
#else
      gen(37500, 12500) // 25k -- 50k
#endif
    {
    }

    /**
     * @brief Returns dimension of the scaled data.
     *
     * @return size_t Dimension of the scaled data.
     */
    size_t dim() const
    {
        return config.size();
    }

    /**
     * @brief Notifies @ref Sweeper that the config has been modified and that
     * the data has to be recomputed.
     *
     */
    void touch_config()
    {
        refresh(*this);
    }

    /**
     * @brief Recomputes the data if any of the config has been touched.
     *
     * @param td Transformed data received from the data flow pipeline.
     */
    void update(const TransData &td, FrameStats &frame_stats);
    /**
     * @brief Resets configurations to their initial values.
     *
     */
    void reset();
};

#endif
