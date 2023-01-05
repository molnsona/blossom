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

struct FrameStats {
    std::vector<size_t> trans_items;
    std::vector<size_t> scatter_items;
    std::vector<size_t> scaled_items;
    std::vector<size_t> color_items;

    std::vector<float> frame_times;
};

#endif // FRAME_STATS_H
