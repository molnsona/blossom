
#include "graph_renderer.h"

#include <Magnum/GL/Renderer.h>
#include <Magnum/Math/Color.h>
#include <Magnum/MeshTools/Compile.h>
#include <Magnum/Primitives/Circle.h>
#include <Magnum/Trade/MeshData.h>

#include <cmath>

using namespace Magnum;
using namespace Math::Literals;

GraphRenderer::GraphRenderer()
{
    line_mesh.setPrimitive(MeshPrimitive::Lines);
    circle_mesh = MeshTools::compile(Primitives::circle2DSolid(36));
}

GraphRenderer::~GraphRenderer() {}

void
GraphRenderer::draw(const View &view,
                    const LandmarkModel &model,
                    float vertex_size)
{
    // TODO cache these allocations in GraphRenderer object
    // std::vector<Vector2> vertices(model.lodim_vertices.size());
    if (vertices.size() != model.lodim_vertices.size()) {
        vertices.clear();
        vertices.resize(model.lodim_vertices.size());
    }

    for (size_t i = 0; i < vertices.size(); ++i) {
        vertices[i] = view.screen_coords(model.lodim_vertices[i]);
    }

    std::vector<Vector2> edge_lines(2 * model.edges.size());
    for (size_t i = 0; i < model.edges.size(); ++i) {
        edge_lines[2 * i + 0] = vertices[model.edges[i].first];
        edge_lines[2 * i + 1] = vertices[model.edges[i].second];
    }

    GL::Buffer buffer;
    buffer.setData(
      Corrade::Containers::ArrayView(edge_lines.data(), edge_lines.size()));

    GL::Renderer::enable(GL::Renderer::Feature::Blending);
    GL::Renderer::setBlendFunction(GL::Renderer::BlendFunction::One,
                                   GL::Renderer::BlendFunction::One);

    line_mesh.setCount(edge_lines.size())
      .addVertexBuffer(std::move(buffer), 0, decltype(flat_shader)::Position{});

    auto screen_proj = view.screen_projection_matrix();

    flat_shader.setTransformationProjectionMatrix(screen_proj)
      .setColor(0xc01010_rgbf)
      .draw(line_mesh);

    flat_shader.setColor(0x666666_rgbf);
    for (auto &&v : vertices) {
        flat_shader
          .setTransformationProjectionMatrix(
            screen_proj * Matrix3::translation(v) *
            Matrix3::scaling(Vector2(vertex_size)))
          .draw(circle_mesh);
    }

    GL::Renderer::disable(GL::Renderer::Feature::Blending);
}

bool
GraphRenderer::is_vert_pressed(Magnum::Vector2 mouse,
                               float vertex_size,
                               std::size_t &vert_ind)
{
    float radius = vertex_size;

    for (size_t i = 0; i < vertices.size(); ++i) {
        auto vert = vertices[i];
        if ((mouse.x() >= roundf(vert.x()) - radius) &&
            (mouse.x() <= roundf(vert.x()) + radius) &&
            (mouse.y() >= roundf(vert.y()) - radius) &&
            (mouse.y() <= roundf(vert.y()) + radius)) {
            vert_ind = i;
            return true;
        }
    }

    return false;
}