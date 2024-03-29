/* This file is part of BlosSOM.
 *
 * Copyright (C) 2021 Sona Molnarova
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

#ifndef FRAME_STATS_H
#define FRAME_STATS_H

#include <cstddef>
#include <vector>

#include "timer.h"

struct FrameStats
{
    // Actual computation time of the methods.
    float trans_t = 0.00001f;
    float embedsom_t = 0.00001f;
    float scaled_t = 0.00001f;
    float color_t = 0.00001f;

    Timer timer;

    // Maximal duration of the measured method
    // (in milliseconds) for estimation of the
    // batch size.
    float trans_duration = 5.0f;
    float embedsom_duration = 5.0f;
    float scaled_duration = 5.0f;
    float color_duration = 5.0f;

    // Duration of the constant functions, per one frame,
    // that does not need to estimate batch size
    // (in milliseconds).
    float constant_time = 0.0f;
    // Time left for computation of method with estimated batch
    // size for the next frame.
    float est_time = 0.1f;

    float embed_priority = 0.05f;
    float color_priority = 0.05f;
    float trans_priority = 0.45f;
    float scaled_priority = 0.45f;

    void start_frame()
    {
        constant_time = 0.0f;
        timer.tick();
    }

    void end_frame()
    {
        add_const_time();

        // Because we want the frame to last ~50ms (~20 FPS).
        float diff = 50.0f - constant_time;
        est_time = diff < 0.0001f ? 1.0f : diff;
    }

    void add_const_time()
    {
        timer.tick();
        constant_time += timer.frametime * 1000; // to get milliseconds
    }

    void store_time(float &to)
    {
        timer.tick();
        to = timer.frametime * 1000; // to get milliseconds
    }

    void reset(float &t) { t = 0.00001f; }

    /**
     * @brief Compute durations of the estimation batch sizes computations.
     *
     */
    void update_times()
    {
        float high = 0.45f;
        float low = 0.05f;
        // First compute statistics
        // if all 4 are computing
        if (trans_t > 0.00001f && scaled_t > 0.00001f &&
            embedsom_t > 0.00001f && color_t > 0.00001f) {
            trans_priority = high;
            scaled_priority = high;
            color_priority = low;
            embed_priority = low;
        } else
            // if trans finished and all other are computing
            if (trans_t <= 0.00001f && scaled_t > 0.00001f &&
                embedsom_t > 0.00001f && color_t > 0.00001f) {
                trans_priority = 0.0f;
                scaled_priority = high + high;
                color_priority = low;
                embed_priority = low;
            } else
                // if scaled finished and all others are computing
                if (trans_t > 0.00001f && scaled_t <= 0.00001f &&
                    embedsom_t > 0.00001f && color_t > 0.00001f) {
                    trans_priority = high + high;
                    scaled_priority = 0.0f;
                    color_priority = low;
                    embed_priority = low;
                } else
                    // if trans and scaled finished computing
                    if (trans_t <= 0.00001f && scaled_t <= 0.00001f &&
                        embedsom_t > 0.00001f && color_t > 0.00001f) {
                        trans_priority = 0.0f;
                        scaled_priority = 0.0f;
                        color_priority = 0.75f;
                        embed_priority = 0.25f;
                    } else
                        // if trans, scaled and color finished computing
                        if (trans_t <= 0.00001f && scaled_t <= 0.00001f &&
                            embedsom_t > 0.00001f && color_t <= 0.00001f) {
                            trans_priority = 0.0f;
                            scaled_priority = 0.0f;
                            color_priority = 0.0f;
                            embed_priority = 1.0f;
                        } else
                            // if trans, scaled and embedsom finished computing
                            if (trans_t <= 0.00001f && scaled_t <= 0.00001f &&
                                embedsom_t <= 0.00001f && color_t > 0.00001f) {
                                trans_priority = 0.0f;
                                scaled_priority = 0.0f;
                                color_priority = 1.0f;
                                embed_priority = 0.0f;
                            } else
                                // if trans, scaled, color and embedsom finished
                                // computing
                                if (trans_t <= 0.00001f &&
                                    scaled_t <= 0.00001f &&
                                    embedsom_t <= 0.00001f &&
                                    color_t <= 0.00001f) {
                                    trans_priority = 0.0f;
                                    scaled_priority = 0.0f;
                                    color_priority = 0.0f;
                                    embed_priority = 0.0f;
                                }

        if (trans_priority == 0.0f)
            trans_duration = 0.0f;
        else
            trans_duration = est_time * trans_priority;
        if (embed_priority == 0.0f)
            embedsom_duration = 0.0f;
        else
            embedsom_duration = est_time * embed_priority;
        if (scaled_priority == 0.0f)
            scaled_duration = 0.0f;
        else
            scaled_duration = est_time * scaled_priority;
        if (color_priority == 0.0f)
            color_duration = 0.0f;
        else
            color_duration = est_time * color_priority;
    }
};

#endif // FRAME_STATS_H
