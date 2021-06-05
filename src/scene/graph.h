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

#include "../app/state.h"

using namespace Magnum;
using namespace Math::Literals;

typedef SceneGraph::Object<SceneGraph::MatrixTransformation3D> Object3D;
typedef SceneGraph::Scene<SceneGraph::MatrixTransformation3D> Scene3D;

class PickableObject: public Object3D, SceneGraph::Drawable3D {
    public:
        explicit PickableObject(UnsignedInt id, Color3 color, GL::Mesh& mesh, Object3D& parent, SceneGraph::DrawableGroup3D& drawables): 
            Object3D{&parent}, 
            SceneGraph::Drawable3D{*this, &drawables}, 
            _id{id}, 
            _selected{false}, 
            _mesh(mesh),
            _color(color)
        { }

        void setSelected(bool selected) { _selected = selected; }
    private:
        virtual void draw(const Matrix4& transformationMatrix, SceneGraph::Camera3D& camera);
        
        UnsignedInt _id;
        bool _selected;
        Shaders::Flat3D _shader{Shaders::Flat3D::Flag::ObjectId};
        GL::Mesh& _mesh;
        bool _changed = false;
        Color3 _color;
};

class Graph {
public:
    Graph(State* p_state, Object3D& parent, SceneGraph::DrawableGroup3D& drawables);

    void draw_event(State* p_state);
private:
    Color3 _vert_color{0xdb4437_rgbf};
    Color3 _edge_color{0x000000_rgbf};

    std::vector<PickableObject*> _vertices;
    std::vector<GL::Mesh> _circle_meshes;

    std::vector<PickableObject*> _edges;
    std::vector<GL::Mesh> _hor_line_meshes;
    std::vector<GL::Mesh> _ver_line_meshes;

    // Shaders::Flat3D _textured_shader{Shaders::Flat3D::Flag::Textured};

};

#endif // #ifndef GRAPH_H