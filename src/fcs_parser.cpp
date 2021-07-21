#include "fcs_parser.h"

#include <cstddef>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>

void
FCSParser::parse(const std::string &fp)
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
