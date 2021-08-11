
#include "scatter_renderer.h"

#include <Magnum/Math/Color.h>
#include <Magnum/MeshTools/Interleave.h>
#include <Magnum/Trade/MeshData.h>

using namespace Magnum;
using namespace Math::Literals;

ScatterRenderer::ScatterRenderer()
{
    point_mesh.setPrimitive(MeshPrimitive::Points);
}

void
ScatterRenderer::draw(const View &view, const ScatterModel &model)
{
    std::vector<Color3> color(model.points.size(), 0x000000_rgbf);
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
