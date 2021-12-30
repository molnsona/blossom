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

#ifndef UTILS_HPP
#define UTILS_HPP

#include <sstream>
#include <string>
#include <vector>

/**
 * @brief Shifts value from [a,b] to [c, d].
 *
 * @param value Old value.
 * @param a Old interval - from.
 * @param b Old interval - to.
 * @param c New interval - from.
 * @param d New interval - to.
 * @return float New value.
 */
float
shift_interval(float value, float a, float b, float c, float d)
{
    return c + ((d - c) / (b - a)) * (value - a);
}

/**
 * @brief Splits a given string into words by a given delimiter.
 *
 * @param str Input string for splitting.
 * @param delim Delimiter used for splitting.
 * @return std::vector<std::string> Array of resulting words.
 */
std::vector<std::string>
split(const std::string &str, char delim)
{
    std::vector<std::string> result;
    std::stringstream ss(str);
    std::string item;

    while (getline(ss, item, delim)) {
        result.emplace_back(item);
    }

    return result;
}

#endif // #ifndef UTILS_HPP
