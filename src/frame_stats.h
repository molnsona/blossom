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
    float trans_t = 0.00001;
    float embedsom_t = 0.00001;
    float scaled_t = 0.00001;
    float color_t = 0.00001;

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
};

#endif // FRAME_STATS_H
