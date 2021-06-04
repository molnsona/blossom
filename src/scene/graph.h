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
        explicit PickableObject(UnsignedInt id, GL::Mesh& mesh, Object3D& parent, SceneGraph::DrawableGroup3D& drawables): Object3D{&parent}, SceneGraph::Drawable3D{*this, &drawables}, _id{id}, _selected{false}, _mesh(mesh)
        { }

        void setSelected(bool selected) { _selected = selected; }
    private:
        virtual void draw(const Matrix4& transformationMatrix, SceneGraph::Camera3D& camera);
        
        UnsignedInt _id;
        bool _selected;
        Shaders::Flat3D _shader{Shaders::Flat3D::Flag::ObjectId};
        GL::Mesh& _mesh;
        bool _changed = false;
};

class Graph {
public:
    Graph(Object3D& parent, SceneGraph::DrawableGroup3D& drawables);

    void draw_event(State* p_state);
private:
    std::vector<PickableObject*> _vertices;
    GL::Mesh _circle_mesh;
    // Shaders::Flat3D _textured_shader{Shaders::Flat3D::Flag::Textured};

};

#endif // #ifndef GRAPH_H