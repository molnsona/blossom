
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
Exporter::export_points(const State &state, const std::string &dir_name)
{
    auto write_data_float = [](size_t size,
                               size_t dim,
                               const std::vector<float> &data,
                               std::ofstream &handle) {
        for (size_t i = 0; i < size; i += dim) {
            for (size_t j = 0; j < dim - 1; ++j) {
                handle << data[i + j] << '\t';
            }
            handle << data[i + dim - 1] << '\n';
        }
    };

    auto write_data_2d = [](size_t size,
                            const std::vector<Magnum::Vector2> &data,
                            std::ofstream &handle) {
        for (size_t i = 0; i < size; ++i) {
            handle << data[i].x() << '\t' << data[i].y() << '\n';
        }
    };

    if (points_hd) {
        std::string path =
          dir_name + "/" + file_names[Exporter::Types::POINTS_HD];
        std::ofstream handle(path, std::ios::out);
        if (!handle)
            throw std::domain_error("Can not open file");

        write_data_float(
          state.data.data.size(), state.data.d, state.data.data, handle);
    }

    if (landmarks_hd) {
        std::string path =
          dir_name + "/" + file_names[Exporter::Types::LAND_HD];
        std::ofstream handle(path, std::ios::out);
        if (!handle)
            throw std::domain_error("Can not open file");

        write_data_float(state.landmarks.hidim_vertices.size(),
                         state.landmarks.d,
                         state.landmarks.hidim_vertices,
                         handle);
    }

    if (points_2d) {
        std::string path =
          dir_name + "/" + file_names[Exporter::Types::POINTS_2D];
        std::ofstream handle(path, std::ios::out);
        if (!handle)
            throw std::domain_error("Can not open file");

        write_data_2d(
          state.scatter.points.size(), state.scatter.points, handle);
    }

    if (landmarks_2d) {
        std::string path =
          dir_name + "/" + file_names[Exporter::Types::LAND_2D];
        std::ofstream handle(path, std::ios::out);
        if (!handle)
            throw std::domain_error("Can not open file");

        write_data_2d(state.landmarks.lodim_vertices.size(),
                      state.landmarks.lodim_vertices,
                      handle);
    }
}
