/* This file is part of BlosSOM.
 *
 * Copyright (C) 2021 Mirek Kratochvil
 *                    Sona Molnarova
 *
 * BlosSOM is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option)
 * any later version.
 *
 * BlosSOM is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * BlosSOM. If not, see <https://www.gnu.org/licenses/>.
 */

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

    // Setup proper blending function.
    GL::Renderer::setBlendFunction(
      GL::Renderer::BlendFunction::SourceAlpha,
      GL::Renderer::BlendFunction::OneMinusSourceAlpha);
}

GraphRenderer::~GraphRenderer() {}

void
GraphRenderer::draw(const View &view, const LandmarkModel &model)
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

    line_mesh.setCount(edge_lines.size())
      .addVertexBuffer(std::move(buffer), 0, decltype(flat_shader)::Position{});

    auto screen_proj = view.screen_projection_matrix();

    flat_shader.setTransformationProjectionMatrix(screen_proj)
      .setColor(0xc01010_rgbf)
      .draw(line_mesh);

    flat_shader.setColor(0x66666666_rgbaf);
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
GraphRenderer::is_vert_pressed(Magnum::Vector2 mouse, size_t &vert_ind) const
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
