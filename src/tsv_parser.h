#ifndef TSV_PARSER_H
#define TSV_PARSER_H

#include <string>
#include <vector>

#include "parser.h"

class TSVParser : public Parser
{
public:
    void parse(const std::string &file_path,
               size_t points_count,
               std::vector<float> &out_data,
               size_t &dim,
               size_t &n);

private:
    std::vector<std::string> split(const std::string &str, char delim);
};

#endif // #ifndef TSV_PARSER_H
