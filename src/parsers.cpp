
#include "parsers.h"

#include "data_model.h"
#include "fcs_parser.h"
#include "tsv_parser.h"

#include <exception>
#include <filesystem>

void
parse_generic(const std::string &filename, DataModel &dm)
{
    auto parse_with = [&](auto f) {
        dm.clear();
        f(filename, dm);

        // TODO temporary precaution, remove later
#if 0
        if (dm.n > 1000) {
            dm.n = 1000;
            dm.data.resize(dm.d * dm.n);
        }
#endif
    };

    std::string ext = std::filesystem::path(filename).extension().string();

    if (ext == ".fcs")
        parse_with(parse_FCS);
    else if (ext == ".tsv")
        parse_with(parse_TSV);
    else
        throw std::domain_error("Unsupported file format.");
}
