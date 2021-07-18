
#ifndef GRAPH_RENDERER_H
#define GRAPH_RENDERER_H

#include <Magnum/GL/Mesh.h>
#include <Magnum/Shaders/FlatGL.h>

#include <vector>

#include "graph_model.h"
#include "view.h"

struct GraphRenderer
{
    GraphRenderer();
    ~GraphRenderer();

    void draw(const View &v, const GraphModel &m, float vertex_size);

private:
    Magnum::GL::Mesh line_mesh;
    Magnum::GL::Mesh circle_mesh;
    Magnum::Shaders::FlatGL2D flat_shader;

    // cached allocations
    std::vector<Magnum::Vector2> line_buf;
};

#endif // #ifndef GRAPH_RENDERER_H
