#include "tsv_parser.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>

void
TSVParser::parse(const std::string &fp,
                 size_t points_count,
                 std::vector<float> &out_data,
                 size_t &dim,
                 size_t &n)
{
    std::string file_path = fp;
    std::string file_name = std::filesystem::path(fp).filename().string();

    std::ifstream file_reader;
    try {
        file_reader.open(file_path, std::ios::in);
    } catch (int e) {
        std::cerr << "Did not open." << e << std::endl;
        return;
    }
    if (!file_reader.is_open()) {
        std::cerr << "Did not open." << std::endl;
        return;
    }

    out_data.clear();

    std::string line;
    std::vector<std::string> values;
    std::vector<float> all_values;
    while (std::getline(file_reader, line)) {
        values = split(line, '\t');
        for (auto &&value : values) {
            all_values.push_back(std::stof(value));
        }
    }

    // pick randomly 1000 points
    std::default_random_engine gen;
    std::uniform_int_distribution<size_t> dist(0, points_count);

    out_data.resize(values.size() * points_count);

    for (size_t i = 0; i < points_count; ++i) {
        size_t ind = dist(gen);
        for (size_t j = 0; j < values.size(); ++j) {
            out_data[i * values.size() + j] =
              all_values[ind * values.size() + j];
        }
    }

    dim = values.size();
    n = points_count;
}

std::vector<std::string>
TSVParser::split(const std::string &str, char delim)
{
    std::vector<std::string> result;
    std::stringstream ss(str);
    std::string item;

    while (getline(ss, item, delim)) {
        result.emplace_back(item);
    }

    return result;
}
