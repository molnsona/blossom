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
    void parse(const std::string &file_path);

    // Getters and setters.
    std::string get_file_name() const { return file_name; }
    std::string get_file_path() const { return file_path; }
    size_t get_data_begin_offset() const { return data_begin_offset; }
    size_t get_data_end_offset() const { return data_end_offset; }

    size_t get_params_count() const { return params_count; }
    size_t get_events_count() const { return events_count; }
    bool get_is_be() const { return is_be; }

    const std::string &get_param_name(size_t idx) { return params_names[idx]; }

    const std::vector<std::string> &get_params_by_num() const
    {
        return params_names;
    }

private:
    /** Parses information from the header of the given file.*/
    void parse_info(std::ifstream &file_reader);
    /** Parses number from the input word, in the format: P[0-9]+N.*/
    static size_t parse_id(const std::string &word);

    std::string file_path;
    std::string file_name;

    // Variables from header and text segment
    size_t data_begin_offset = 0;
    size_t data_end_offset = 0;
    /** Number of measured characteristics of the cells.*/
    size_t params_count = 0;
    /** Total number of events in the dataset, (i.e., number of cells).*/
    size_t events_count = 0;
    bool is_be = false;

    /**
     * List of the names of the measured characteristics.
     * Index of the vector is the index of the characteristic.
     */
    std::vector<std::string> params_names;
};

#endif // #ifndef FCS_PARSER_H
