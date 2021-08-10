#ifndef FCS_PARSER_H
#define FCS_PARSER_H

#include <cstdint>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

class FCSParser
{
public:
    /**
     * Parses input fcs file header information.
     */
    static void parse(const std::string &file_path,
                      size_t points_count,
                      std::vector<float> &out_data,
                      size_t &dim,
                      size_t &n,
                      std::vector<std::string> &param_names);

private:
    /** Parses information from the header of the given file.*/
    static void parse_info(std::ifstream &file_reader,
                           size_t &data_begin_offset,
                           size_t &data_end_offset,
                           size_t &params_count,
                           size_t &events_count,
                           bool &is_be,
                           std::vector<std::string> &params_names);
    /** Parses data from the file*/
    static void parse_data(std::ifstream &file_reader,
                           size_t points_count,
                           size_t &data_begin_offset,
                           size_t &data_end_offset,
                           size_t &params_count,
                           size_t &events_count,
                           bool &is_be,
                           std::vector<float> &out_data);

    /** Parses number from the input word, in the format: P[0-9]+N.*/
    static size_t parse_id(const std::string &word);
};

#endif // #ifndef FCS_PARSER_H
