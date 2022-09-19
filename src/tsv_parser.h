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

#ifndef TSV_PARSER_H
#define TSV_PARSER_H

#include "data_model.h"
#include <string>

/**
 * @brief Parses FCS file and fills @ref DataModel data.
 *
 * @param[in] file_path File path to the FCS file.
 * @param[out] dm @ref DataModel instance filled by data from parsed FCS file.
 *
 * \exception std::domain_error Throws when the file cannot be opened.
 * \exception std::length_error Throws when some row has different number of
 * columns than is the dimension of the data.
 */
void
parse_TSV(const std::string &file_path, DataModel &dm);

#endif // #ifndef TSV_PARSER_H
