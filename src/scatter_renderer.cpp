
#include <extern/colormap/color.hpp>
#include <extern/colormap/map.hpp>
#include <extern/colormap/palettes.hpp>
#include <extern/colormap/pixmap.hpp>

#include "scatter_renderer.h"

#include <Magnum/MeshTools/Interleave.h>
#include <Magnum/Trade/MeshData.h>

#include <algorithm>
#include <limits>

using namespace Magnum;
using namespace Math::Literals;

ScatterRenderer::ScatterRenderer()
{
    point_mesh.setPrimitive(MeshPrimitive::Points);
}

void
ScatterRenderer::draw(const View &view,
                      const ScatterModel &model,
                      const TransData &trans_data)
{
    std::vector<Color3> color = fill_color(trans_data);
    GL::Buffer buffer;
    buffer.setData(MeshTools::interleave(
      Corrade::Containers::ArrayView(model.points.data(), model.points.size()),
      Corrade::Containers::ArrayView(color.data(), color.size())));

    point_mesh.setCount(model.points.size())
      .addVertexBuffer(std::move(buffer),
                       0,
                       decltype(flat_shader)::Position{},
                       decltype(flat_shader)::Color3{});

    auto proj = view.projection_matrix();

    flat_shader.setTransformationProjectionMatrix(proj).draw(point_mesh);
}

std::vector<Color3>
ScatterRenderer::fill_color(const TransData &trans_data)
{
    std::vector<Color3> color(trans_data.n);

    float min{ std::numeric_limits<float>::max() },
      max{ std::numeric_limits<float>::min() };

    for (size_t i = 0; i < trans_data.n; ++i) {
        // take second parameter
        max = std::max(max, trans_data.get_data().at(i * trans_data.d + 1));
        min = std::min(min, trans_data.get_data().at(i * trans_data.d + 1));
    }

    auto col_palette = colormap::palettes.at("rdbu").rescale(min, max);

    for (size_t i = 0; i < trans_data.n; ++i) {
        auto c = col_palette(trans_data.get_data().at(i * trans_data.d + 1));
        std::vector<unsigned char> res;
        c.get_rgb(res);
        color[i] = Color3(res[0] / 255.0f, res[1] / 255.0f, res[2] / 255.0f);
    }

    return color;
}
