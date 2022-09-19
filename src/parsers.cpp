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

#include "parsers.h"

#include "data_model.h"
#include "fcs_parser.h"
#include "tsv_parser.h"

#include <exception>
#include <filesystem>

void
parse_generic(const std::string &filename, DataModel &dm)
{
    auto parse_with = [&](auto f) {
        dm.clear();
        f(filename, dm);

        // TODO temporary precaution, remove later
#if 0
        if (dm.n > 1000) {
            dm.n = 1000;
            dm.data.resize(dm.d * dm.n);
        }
#endif
    };

    std::string ext = std::filesystem::path(filename).extension().string();

    if (ext == ".fcs")
        parse_with(parse_FCS);
    else if (ext == ".tsv")
        parse_with(parse_TSV);
    else
        throw std::domain_error("Unsupported file format.");
}
