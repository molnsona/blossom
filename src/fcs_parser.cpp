#include "fcs_parser.h"

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <random>
#include <regex>
#include <sstream>
#include <string>

void
FCSParser::parse(const std::string &fp,
                 size_t points_count,
                 std::vector<float> &out_data,
                 size_t &dim,
                 size_t &n)
{
    file_path = fp;
    file_name = std::filesystem::path(fp).filename().string();

    std::ifstream file_reader;
    try {
        file_reader.open(file_path, std::ios::binary | std::ios::in);
    } catch (int e) {
        std::cerr << "Did not open." << e << std::endl;
        return;
    }
    if (!file_reader.is_open()) {
        std::cerr << "Did not open." << std::endl;
        return;
    }

    parse_info(file_reader);
    dim = params_count;
    n = points_count;
    parse_data(file_reader, points_count, out_data);
    file_reader.close();
}

void
FCSParser::parse_info(std::ifstream &file_reader)
{
    size_t text_begin_offset;
    size_t text_end_offset;

    // Offset of the name and version.
    constexpr int off = 7;
    // Ignore name and version.
    file_reader.ignore(off, ' ');

    // Save text begin and end offset.
    file_reader >> text_begin_offset >> text_end_offset;

    file_reader.seekg(text_begin_offset);

    // Read delimiter
    char delim = file_reader.get();

    std::string word;
    // Can convert to long int, because it is only header, and it will never be
    // greater than long int.
    while (size_t(file_reader.tellg()) < text_end_offset + 1) {
        std::getline(file_reader, word, delim);

        if (word == "$BEGINDATA") {
            std::getline(file_reader, word, delim);
            data_begin_offset = static_cast<size_t>(stoul(word));
            continue;
        }

        if (word == "$BYTEORD") {
            std::getline(file_reader, word, delim);
            if (word == "4,3,2,1")
                is_be = true;

            continue;
        }

        if (word == "$ENDDATA") {
            std::getline(file_reader, word, delim);
            data_end_offset = static_cast<size_t>(stoul(word));
            continue;
        }

        if (std::regex_match(word, std::regex("\\$P[0-9]+N"))) {
            size_t id = parse_id(word);

            std::getline(file_reader, word, delim);

            // If id is greater than size of vector, it needs to be resized
            if (params_names.size() < id)
                params_names.resize(id, "");
            params_names[id - 1] = word;

            continue;
        }

        if (word == "$PAR") {
            std::getline(file_reader, word, delim);
            params_count = static_cast<size_t>(stoul(word));
            continue;
        }

        if (word == "$TOT") {
            std::getline(file_reader, word, delim);
            events_count = static_cast<size_t>(stoul(word));
            continue;
        }
    }
}

void
FCSParser::parse_data(std::ifstream &file_reader,
                      size_t points_count,
                      std::vector<float> &out_data)
{
    // If not enough points.
    auto diff = data_end_offset - data_begin_offset;
    if (diff < params_count * points_count * sizeof(float))
        points_count = diff / params_count / sizeof(float);

#if 0
    std::vector<float> all_values(events_count);

    file_reader.seekg(data_begin_offset);
    file_reader.read(reinterpret_cast<char *>(all_values.data()),
                     events_count * sizeof(float));

    // pick randomly 1000 points
    std::default_random_engine gen;
    std::uniform_int_distribution<size_t> dist(0, points_count);

    out_data.resize(params_count * points_count);

    for (size_t i = 0; i < points_count; ++i) {
        size_t ind = dist(gen);
        for (size_t j = 0; j < params_count; ++j) {
            out_data[i * params_count + j] = all_values[ind * params_count + j];
        }
    }
#endif

    // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    // Pick first 1000 points from data set
    // TODO: Later use all points.
    out_data.resize(params_count * points_count);
    file_reader.seekg(data_begin_offset);
    file_reader.read(reinterpret_cast<char *>(out_data.data()),
                     params_count * points_count * sizeof(float));
    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    if (!is_be)
        // little endian
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
    else
        // big endian
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
}

size_t
FCSParser::parse_id(const std::string &word)
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
