
#include "exporter.h"

#include <fstream>

Exporter::Exporter()
  : points_hd(false)
  , landmarks_hd(false)
  , points_2d(false)
  , landmarks_2d(false)
  , all(false)
  , file_names{ "points_hd.tsv",
                "landmarks_hd.tsv",
                "points_2d.tsv",
                "landmarks_2d.tsv" }
{}

void
Exporter::export_data(const State &state, const std::string &dir_name)
{
    if (points_hd)
        write(Exporter::Types::POINTS_HD, state, dir_name);

    if (landmarks_hd)
        write(Exporter::Types::LAND_HD, state, dir_name);

    if (points_2d)
        write(Exporter::Types::POINTS_2D, state, dir_name);

    if (landmarks_2d)
        write(Exporter::Types::LAND_2D, state, dir_name);
}

static void
write_data_float(size_t size,
                 size_t dim,
                 const std::vector<float> &data,
                 std::ofstream &handle)
{
    for (size_t i = 0; i < size; i += dim) {
        for (size_t j = 0; j < dim - 1; ++j) {
            handle << data[i + j] << '\t';
        }
        handle << data[i + dim - 1] << '\n';
    }
};

static void
write_data_2d(size_t size,
              const std::vector<Magnum::Vector2> &data,
              std::ofstream &handle)
{
    for (size_t i = 0; i < size; ++i) {
        handle << data[i].x() << '\t' << data[i].y() << '\n';
    }
};

void
Exporter::write(Exporter::Types type,
                const State &state,
                const std::string &dir_name)
{
    std::string path = dir_name + "/" + file_names[type];
    std::ofstream handle(path, std::ios::out);
    if (!handle)
        throw std::domain_error("Can not open file");

    switch (type) {
        case Exporter::Types::POINTS_HD:
            write_data_float(
              state.data.data.size(), state.data.d, state.data.data, handle);

            break;
        case Exporter::Types::LAND_HD:
            write_data_float(state.landmarks.hidim_vertices.size(),
                             state.landmarks.d,
                             state.landmarks.hidim_vertices,
                             handle);
            break;
        case Exporter::Types::POINTS_2D:
            write_data_2d(
              state.scatter.points.size(), state.scatter.points, handle);
            break;
        case Exporter::Types::LAND_2D:
            write_data_2d(state.landmarks.lodim_vertices.size(),
                          state.landmarks.lodim_vertices,
                          handle);
            break;
    }

    handle.close();
}
