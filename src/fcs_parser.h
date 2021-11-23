#ifndef FCS_PARSER_H
#define FCS_PARSER_H

#include <cstdint>
#include <string>
#include <vector>

void parse_FCS(const std::string &file_path,
	      size_t points_count,
	      std::vector<float> &out_data,
	      size_t &dim,
	      size_t &n,
	      std::vector<std::string> &param_names);

#endif // #ifndef FCS_PARSER_H
