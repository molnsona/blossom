#ifndef GRAPH_H
#define GRAPH_H

#include <Magnum/SceneGraph/MatrixTransformation3D.h>
#include <Magnum/SceneGraph/Drawable.h>
#include <Magnum/SceneGraph/Object.h>
#include <Magnum/SceneGraph/Camera.h>
#include <Magnum/Shaders/Flat.h>
#include <Magnum/GL/Texture.h>
#include <Magnum/GL/TextureFormat.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/Math/Color.h>

#include <memory>

#include "../app/state.h"
#include "../pickable.hpp"

using namespace Magnum;
using namespace Math::Literals;

typedef SceneGraph::Object<SceneGraph::MatrixTransformation3D> Object3D;
typedef SceneGraph::Scene<SceneGraph::MatrixTransformation3D> Scene3D;

class Graph {
public:
    Graph() = delete;
    Graph(State* p_state, Object3D& parent, SceneGraph::DrawableGroup3D& drawables);

    void update(State* p_state);

    void draw_event(State* p_state, Object3D& parent, SceneGraph::DrawableGroup3D& drawables);
private:
    static constexpr std::size_t vtx_count = 49;
    
    Color3 _vert_color{0xdb4437_rgbf};
    Color3 _edge_color{0x000000_rgbf};

    std::vector<std::unique_ptr<PickableObject>> _vertices;
    std::vector<GL::Mesh> _circle_meshes;

    std::vector<std::unique_ptr<PickableObject>> _edges;
    std::size_t edges_size;
    std::vector<GL::Mesh> _edge_meshes;

    //std::vector<GL::Mesh> meshes;
    std::vector<GL::Mesh> _hor_line_meshes;
    std::vector<GL::Mesh> _ver_line_meshes;

    // Shaders::Flat3D _textured_shader{Shaders::Flat3D::Flag::Textured};

};

#endif // #ifndef GRAPH_H