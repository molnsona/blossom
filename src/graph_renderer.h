
#ifndef GRAPH_RENDERER_H
#define GRAPH_RENDERER_H

#include <Magnum/GL/Mesh.h>
#include <Magnum/Shaders/FlatGL.h>

#include <vector>

#include "landmark_model.h"
#include "view.h"

struct GraphRenderer
{
    GraphRenderer();
    ~GraphRenderer();

    // TODO: this should not know about actual Landmarks, we should pass actual
    // vertex + edge positions as with the layouter.
    void draw(const View &v, const LandmarkModel &m, float vertex_size);

    // If some vertex is pressed it returns true and index of the vertex
    bool is_vert_pressed(Magnum::Vector2 mouse,
                         float vertex_size,
                         std::size_t &vert_ind);

private:
    Magnum::GL::Mesh line_mesh;
    Magnum::GL::Mesh circle_mesh;
    Magnum::Shaders::FlatGL2D flat_shader;

    // cached allocations
    std::vector<Magnum::Vector2> line_buf;
    // In screen coordinates.
    std::vector<Magnum::Vector2> vertices;
};

#endif // #ifndef GRAPH_RENDERER_H
