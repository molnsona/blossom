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

    bool points_hd;
    bool landmarks_hd;
    bool points_2d;
    bool landmarks_2d;
    bool all;

    std::array<std::string, Types::COUNT> file_names;

    Exporter();

    void export_points(const State &state, const std::string &dir_name);
};

#endif // #ifndef EXPORTER_H
