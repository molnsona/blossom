#ifndef TSV_PARSER_H
#define TSV_PARSER_H

#include <string>
#include <vector>

void parse_TSV(const std::string &file_path,
                      size_t points_count,
                      std::vector<float> &out_data,
                      size_t &dim,
                      size_t &n,
                      std::vector<std::string> &param_names);

#endif // #ifndef TSV_PARSER_H
