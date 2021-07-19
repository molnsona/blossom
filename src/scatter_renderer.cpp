
#include "scatter_renderer.h"

#include <Magnum/Math/Color.h>
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
    GL::Buffer buffer;
    buffer.setData(
      Corrade::Containers::ArrayView(model.points.data(), model.points.size()));

    point_mesh.setCount(model.points.size())
      .addVertexBuffer(std::move(buffer), 0, decltype(flat_shader)::Position{});

    auto proj = view.projection_matrix();

    flat_shader.setColor(0x88ccff_rgbf)
      .setTransformationProjectionMatrix(proj)
      .draw(point_mesh);
}
