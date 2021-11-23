
#include "fcs_parser.h"

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <regex>
#include <sstream>
#include <string>

static size_t
parse_id(const std::string &word)
{
    std::stringstream ss(word);
    std::stringstream output;
    char c;
    while (ss >> c) {
        if (isdigit(c) != 0)
            output << c;
    }

    return stoi(output.str());
}

static void
parse_info(std::ifstream &handle,
           size_t &data_begin_offset,
           size_t &data_end_offset,
           size_t &params_count,
           size_t &events_count,
           bool &is_be,
           std::vector<std::string> &params_names)
{
    size_t text_begin_offset;
    size_t text_end_offset;

    // Offset of the name and version.
    constexpr int off = 7;
    // Ignore name and version.
    handle.ignore(off, ' ');

    // Save text begin and end offset.
    handle >> text_begin_offset >> text_end_offset;

    handle.seekg(text_begin_offset);

    // Read delimiter
    char delim = handle.get();

    std::string word;
    // Can convert to long int, because it is only header, and it will never be
    // greater than long int.
    while (size_t(handle.tellg()) < text_end_offset + 1) {
        std::getline(handle, word, delim);

        if (word == "$BEGINDATA") {
            std::getline(handle, word, delim);
            data_begin_offset = static_cast<size_t>(stoul(word));
            continue;
        }

        if (word == "$BYTEORD") {
            std::getline(handle, word, delim);
            is_be = word == "4,3,2,1";
            continue;
        }

        if (word == "$ENDDATA") {
            std::getline(handle, word, delim);
            data_end_offset = static_cast<size_t>(stoul(word));
            continue;
        }

        if (std::regex_match(word, std::regex("\\$P[0-9]+N"))) {
            size_t id = parse_id(word);

            std::getline(handle, word, delim);

            // If id is greater than size of vector, it needs to be resized
            if (params_names.size() < id)
                params_names.resize(id, "");
            params_names[id - 1] = word;

            continue;
        }

        if (word == "$PAR") {
            std::getline(handle, word, delim);
            params_count = static_cast<size_t>(stoul(word));
            continue;
        }

        if (word == "$TOT") {
            std::getline(handle, word, delim);
            events_count = static_cast<size_t>(stoul(word));
            continue;
        }
    }
}

static void
parse_data(std::ifstream &handle,
           size_t data_begin_offset,
           size_t data_end_offset,
           size_t params_count,
           size_t events_count,
           bool is_be,
           std::vector<float> &out_data)
{
    // If not enough points.
    auto diff = data_end_offset - data_begin_offset;
    if (diff < params_count * events_count * sizeof(float))
        events_count = diff / params_count / sizeof(float);

    handle.seekg(data_begin_offset);
    handle.read(reinterpret_cast<char *>(out_data.data()),
                params_count * events_count * sizeof(float));

    if (is_be)
        std::transform(
          out_data.begin(), out_data.end(), out_data.begin(), [](float n) {
              uint8_t *tmp = reinterpret_cast<uint8_t *>(&n);
              uint32_t w1 = *tmp;
              uint32_t w2 = *(tmp + 1);
              uint32_t w3 = *(tmp + 2);
              uint32_t w4 = *(tmp + 3);
              uint32_t res = w4 << 0 | w3 << 8 | w2 << 16 | w1 << 24;
              return *reinterpret_cast<float *>(&res);
          });
    else
        std::transform(
          out_data.begin(), out_data.end(), out_data.begin(), [](float n) {
              uint8_t *tmp = reinterpret_cast<uint8_t *>(&n);
              uint32_t w1 = *tmp;
              uint32_t w2 = *(tmp + 1);
              uint32_t w3 = *(tmp + 2);
              uint32_t w4 = *(tmp + 3);
              uint32_t res = w1 << 0 | w2 << 8 | w3 << 16 | w4 << 24;
              return *reinterpret_cast<float *>(&res);
          });
}

void
parse_FCS(const std::string &filename, DataModel &dm)
{
    std::ifstream handle(filename, std::ios::in);
    if (!handle)
        throw std::domain_error("Can not open file");

    size_t data_begin_offset = 0;
    size_t data_end_offset = 0;
    bool is_be = false;

    parse_info(
      handle, data_begin_offset, data_end_offset, dm.d, dm.n, is_be, dm.names);
    parse_data(
      handle, data_begin_offset, data_end_offset, dm.d, dm.n, is_be, dm.data);
    handle.close();
}
