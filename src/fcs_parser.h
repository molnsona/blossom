#ifndef FCS_PARSER_H
#define FCS_PARSER_H

#include <string>
#include "data_model.h"

void parse_FCS(const std::string &file_path,
DataModel&dm);

#endif
