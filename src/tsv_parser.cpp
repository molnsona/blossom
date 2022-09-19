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

#include "tsv_parser.h"

#include <fstream>
#include <sstream>

// TODO replace by inplace ops
/**
 * @brief Splits a given string into words by a given delimiter.
 *
 * @param str Input string for splitting.
 * @param delim Delimiter used for splitting.
 * @return std::vector<std::string> Array of resulting words.
 *
 * \todo TODO replace by inplace ops
 */
static std::vector<std::string>
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

void
parse_TSV(const std::string &filename, DataModel &dm)
{
    std::ifstream handle(filename, std::ios::in);
    if (!handle)
        throw std::domain_error("Can not open file");

    std::string line;

    while (std::getline(handle, line)) {
        std::vector<std::string> values = split(line, '\t');
        if (values.size() == 0)
            continue;

        if (dm.d == 0) {
            // first line that contains anything is a header with data dimension
            dm.d = values.size();
            dm.names = values;
            continue;
        } else if (dm.d != values.size())
            throw std::length_error("Row length mismatch");
        for (auto &&value : values)
            dm.data.emplace_back(std::stof(value));
        ++dm.n;
    }

    if (!dm.n)
        throw std::domain_error("File contained no data!");
}
