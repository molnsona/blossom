#ifndef EXPORTER_H
#define EXPORTER_H

#include "state.h"

#include <array>
#include <string>

struct Exporter
{
    enum Types
    {
        POINTS_HD,
        LAND_HD,
        POINTS_2D,
        LAND_2D,
        COUNT // Number of possible export types
    };

    bool all;
    std::array<bool, Types::COUNT> data_flags;
    std::array<std::string, Types::COUNT> file_names;

    Exporter();

    void export_data(const State &state, const std::string &dir_name);
    void write(Exporter::Types type,
               const State &state,
               const std::string &dir_name);
};

#endif // #ifndef EXPORTER_H
