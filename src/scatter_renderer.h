
#ifndef SCATTER_RENDERER_H
#define SCATTER_RENDERER_H

#include <Magnum/GL/Mesh.h>
#include <Magnum/Shaders/FlatGL.h>

#include "scatter_model.h"
#include "view.h"

struct ScatterRenderer
{
    ScatterRenderer();

    void draw(const View &v, const ScatterModel &m);

private:
    Magnum::GL::Mesh point_mesh;
    Magnum::Shaders::FlatGL2D flat_shader;
};

#endif
