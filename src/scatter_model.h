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

#ifndef SCATTER_MODEL_H
#define SCATTER_MODEL_H

#include <glm/glm.hpp>

#include <vector>

#include "dirty.h"
#include "landmark_model.h"
#include "normal_gen.h"
#include "scaled_data.h"
#include "training_config.h"

#if ENABLE_CUDA
#include "embedsom_cuda.h"
#endif

/**
 * @brief Model of the two-dimensional data points.
 *
 */
struct ScatterModel : public Sweeper
{
    /** Coordinates of the two-dimensional data points. */
    std::vector<glm::vec2> points;

#if ENABLE_CUDA
    EmbedSOMCUDAContext embedsom_cuda;
#endif

    Cleaner lm_watch;

    NormalGen gen;

    ScatterModel() :
#ifndef ENABLE_CUDA
      gen(7500, 2500) // 5k -- 10k
#else
      gen(37500, 12500) // 25k -- 50k
#endif
    {}

    /**
     * @brief Recomputes the coordinates if any of the the parameters of the
     * embedsom algorithm has changed.
     *
     * @param d Scaled data used for recomputation.
     * @param lm Landmark model used for recomputation.
     * @param tc Dynamic parameters of the algorithms set in the GUI.
     */
    void update(const ScaledData &d,
                const LandmarkModel &lm,
                const TrainingConfig &tc,
                FrameStats& frame_stats);

    /**
     * @brief Notifies @ref Sweeper that the parameters of the embedsom
     * algorithm has been modified and that the coordinates has to be
     * recomputed.
     *
     */
    void touch_config()
    {
        refresh(points.size());
    }
};

#endif
