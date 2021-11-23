
#include "tsv_parser.h"

#include <fstream>

// TODO replace by inplace ops
static std::vector<std::string>
split(const std::string &str, char delim)
{
    std::vector<std::string> result;
    std::stringstream ss(str);
    std::string item;

    while (getline(ss, item, delim)) {
        result.emplace_back(item);
    }

    return result;
}

parse_TSV(const std::string &filename, DataModel &dm)
{
    std::ifstream handle(filename, std::ios::in);
    if (!handle)
        throw std::domain_error("Can not open file");

    std::string line;

    dm.clear();

    while (std::getline(handle, line)) {
        std::vector<std::string> values = split(line, '\t');
        if (values.size() == 0)
            continue;

        if (dm.d == 0) {
            // first line that contains anything is a header with data dimension
            dm.d = values.size();
            dm.names = values;
            continue;
        } else if (dm.d != values.size())
            throw std::length_error("Row length mismatch");
        for (auto &&value : values)
            dm.data.emplace_back(std::stof(value));
        ++dm.n;
    }

    if (!dm.n)
        throw std::domain_error("File contained no data!");
}
