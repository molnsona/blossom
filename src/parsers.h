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

#ifndef PARSERS_H
#define PARSERS_H

#include "data_model.h"
#include <string>

/**
 * @brief Parses data from input file.
 *
 * The chosen parser depends on the file type.
 *
 * @param[in] file_path Path to the file that will be parsed.
 * @param[out] dm @ref DataModel instance filled by data from the parsed file.
 */
void
parse_generic(const std::string &file_path, DataModel &dm);

#endif
