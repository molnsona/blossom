/* This file is part of BlosSOM.
 *
 * Copyright (C) 2021 Mirek Kratochvil
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

#ifndef TIMER_H
#define TIMER_H

#include <chrono>

/**
 * @brief Handler for frametime computation.
 *
 */
struct Timer
{
    using timepoint = std::chrono::time_point<std::chrono::steady_clock>;

    /** Duration of the last frame. */
    float frametime;
    /** Time of the last tick. */
    timepoint last_tick;

    /**
     * @brief Calls @ref tick() and sets frametime to zero.
     *
     */
    Timer()
    {
        tick();
        frametime = 0.0;
    }

    /**
     * @brief Counts \p frametime and sets \p last_tick variable to current
     * time.
     *
     */
    void tick()
    {
        timepoint now = std::chrono::steady_clock::now();
        frametime = std::chrono::duration<float>(now - last_tick).count();
        last_tick = now;
    }
};

#endif
