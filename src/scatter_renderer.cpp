
#include "scatter_renderer.h"

#include <Magnum/GL/Mesh.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/MeshTools/Interleave.h>
#include <Magnum/Trade/MeshData.h>

#include <algorithm>
#include <limits>

using namespace Magnum;
using namespace Math::Literals;

ScatterRenderer::ScatterRenderer()
  : flat_shader{ Magnum::Shaders::FlatGL2D::Flag::VertexColor }
{
    // Setup proper blending function.
    GL::Renderer::setBlendFunction(
      GL::Renderer::BlendFunction::SourceAlpha,
      GL::Renderer::BlendFunction::OneMinusSourceAlpha);
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

    GL::Renderer::enable(GL::Renderer::Feature::Blending);

    Magnum::GL::Mesh point_mesh;
    point_mesh.setPrimitive(MeshPrimitive::Points);
    point_mesh.setCount(n).addVertexBuffer(std::move(buffer),
                                           0,
                                           decltype(flat_shader)::Position{},
                                           decltype(flat_shader)::Color4{});

    auto proj = view.projection_matrix();

    flat_shader.setTransformationProjectionMatrix(proj).draw(point_mesh);

    GL::Renderer::disable(GL::Renderer::Feature::Blending);
}
