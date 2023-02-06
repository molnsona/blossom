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

#define DEBUG

struct FrameStats
{
    // TODO: remove these Ns, they are only
    // for debug output
    size_t trans_n;
    size_t embedsom_n;
    size_t scaled_n;
    size_t color_n;

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

    float embed_priority = 0.125f;
    float color_priority = 0.125f;
    float trans_priority = 0.375f;
    float scaled_priority = 0.375f;

#ifdef DEBUG
    float gl_finish_time = 0.0f;
    float prev_const_time = 0.0f;
    float dt;
#endif

    /**
     * @brief Compute durations of the estimation batch sizes computations.
     * 
     */
    void update_times() {
        // First compute statistics
        // if all 4 are computing
        if(trans_t > 0.00001f && scaled_t > 0.00001f &&
            embedsom_t > 0.00001f && color_t > 0.00001f) {
            trans_priority = 0.45;
            scaled_priority = 0.45f;
            color_priority = 0.05f;
            embed_priority = 0.05f;        
        } 
        else
        // if trans finished and all other are computing
        if(trans_t <= 0.00001f && scaled_t > 0.00001f &&
            embedsom_t > 0.00001f && color_t > 0.00001f) {
            trans_priority = 0.0f;
            scaled_priority = 0.9f;
            color_priority = 0.05f;
            embed_priority = 0.05f;
        }
        else
        // if scaled finished and all others are computing
        if(trans_t > 0.00001f && scaled_t <= 0.00001f &&
            embedsom_t > 0.00001f && color_t > 0.00001f) {
            trans_priority = 0.9f;
            scaled_priority = 0.0f;
            color_priority = 0.05f;
            embed_priority = 0.05f;
        }
        else
        // if trans and scaled finished computing
        if(trans_t <= 0.00001f && scaled_t <= 0.00001f &&
            embedsom_t > 0.00001f && color_t > 0.00001f) {
            trans_priority = 0.0f;
            scaled_priority = 0.0f;
            color_priority = 0.75f;
            embed_priority = 0.25f;
        }
        else
        // if trans, scaled and color finished computing
        if(trans_t <= 0.00001f && scaled_t <= 0.00001f &&
            embedsom_t > 0.00001f && color_t <= 0.00001f) {
            trans_priority = 0.0f;
            scaled_priority = 0.0f;
            color_priority = 0.0f;
            embed_priority = 1.0f;
        }
        else
        // if trans, scaled and embedsom finished computing
        if(trans_t <= 0.00001f && scaled_t <= 0.00001f &&
            embedsom_t <= 0.00001f && color_t > 0.00001f) {
            trans_priority = 0.0f;
            scaled_priority = 0.0f;
            color_priority = 1.0f;
            embed_priority = 0.0f;
        }
        else
        // if trans, scaled, color and embedsom finished computing
        if(trans_t <= 0.00001f && scaled_t <= 0.00001f &&
            embedsom_t <= 0.00001f && color_t <= 0.00001f) {
            trans_priority = 0.0f;
            scaled_priority = 0.0f;
            color_priority = 0.0f;
            embed_priority = 0.0f;
        }

        float alpha = 0.05f;
        float coalpha = 1 - 0.05f;
        if(trans_priority == 0.0f) trans_duration = 0.0f;
        else trans_duration =
            trans_duration * coalpha + 
            est_time * trans_priority * alpha;
        if(embed_priority == 0.0f) embedsom_duration = 0.0f;
        else embedsom_duration =
            embedsom_duration * coalpha +
            est_time * embed_priority * alpha;
        if(scaled_priority == 0.0f) scaled_duration = 0.0f;
        else scaled_duration =
            scaled_duration * coalpha +
            est_time * scaled_priority * alpha;
        if(color_priority == 0.0f) color_duration = 0.0f;
        else color_duration = 
            color_duration * coalpha + 
            est_time * color_priority * alpha;    
    }
};

#endif // FRAME_STATS_H
