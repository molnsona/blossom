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

#ifndef DATA_MODEL_H
#define DATA_MODEL_H

#include <string>
#include <vector>

#include "dirty.h"

/**
 * @brief Storage of data from loaded input file.
 *
 */
struct DataModel : public Dirts
{
    /** One-dimensional array storing d-dimensional input data in
     * row-major order. */
    std::vector<float> data;
    /** Names of the dimensions. */
    std::vector<std::string> names;
    /** Dimension size. */
    size_t d;

    /**
     * @brief Calls @ref clear().
     *
     */
    DataModel() { clear(); }

    /**
     * @brief Clears all DataModel data to their default values.
     *
     */
    void clear()
    {
        d = n = 0;
        data.clear();
        names.clear();
        touch();
    }
};

#endif
