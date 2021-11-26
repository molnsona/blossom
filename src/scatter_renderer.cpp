
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
  : flat_shader{ Magnum::Shaders::FlatGL2D::Flag::VertexColor }
{
    point_mesh.setPrimitive(MeshPrimitive::Points);
}

void
ScatterRenderer::draw(const View &view,
                      const ScatterModel &model,
                      const ColorData &colors)
{
    GL::Buffer buffer;
    size_t n =
      std::min(model.points.size(),
               colors.data.size()); // misalignment aborts it, be careful

    buffer.setData(MeshTools::interleave(
      Corrade::Containers::ArrayView(model.points.data(), n),
      Corrade::Containers::ArrayView(colors.data.data(), n)));

    point_mesh.setCount(n).addVertexBuffer(std::move(buffer),
                                           0,
                                           decltype(flat_shader)::Position{},
                                           decltype(flat_shader)::Color3{});

    auto proj = view.projection_matrix();

    flat_shader.setTransformationProjectionMatrix(proj).draw(point_mesh);
}
