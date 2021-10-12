
#ifndef SCATTER_RENDERER_H
#define SCATTER_RENDERER_H

#include <Magnum/GL/Mesh.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Shaders/FlatGL.h>

#include "scatter_model.h"
#include "trans_data.h"
#include "view.h"

using namespace Magnum;
using namespace Math::Literals;

struct ScatterRenderer
{
    ScatterRenderer();

    void draw(const View &v,
              const ScatterModel &m,
              const TransData &trans_data,
              std::size_t col_ind);

private:
    std::vector<Color3> fill_color(const TransData &trans_data,
                                   std::size_t col_ind);

    Magnum::GL::Mesh point_mesh;
    Magnum::Shaders::FlatGL2D flat_shader{
        Shaders::FlatGL2D::Flag::VertexColor
    };
};

#endif
