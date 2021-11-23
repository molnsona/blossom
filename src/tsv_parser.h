#ifndef TSV_PARSER_H
#define TSV_PARSER_H

#include "data_model.h"
#include <string>

void
parse_TSV(const std::string &file_path, DataModel &dm);

#endif // #ifndef TSV_PARSER_H
