#ifndef FCS_PARSER_H
#define FCS_PARSER_H

#include "data_model.h"
#include <string>

void
parse_FCS(const std::string &file_path, DataModel &dm);

#endif
