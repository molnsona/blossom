#ifndef PICKABLE_HPP
#define PICKABLE_HPP

#include <Magnum/SceneGraph/MatrixTransformation3D.h>
#include <Magnum/SceneGraph/Drawable.h>
#include <Magnum/SceneGraph/Object.h>
#include <Magnum/SceneGraph/Camera.h>
#include <Magnum/Shaders/Flat.h>
#include <Magnum/GL/Texture.h>
#include <Magnum/GL/TextureFormat.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/Math/Color.h>

#include <iostream>

using namespace Magnum;
using namespace Math::Literals;

typedef SceneGraph::Object<SceneGraph::MatrixTransformation3D> Object3D;
typedef SceneGraph::Scene<SceneGraph::MatrixTransformation3D> Scene3D;

class PickableObject: public Object3D, SceneGraph::Drawable3D 
{
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
    virtual void draw(const Matrix4& transformationMatrix, SceneGraph::Camera3D& camera)
    {
        Color3 color = _selected ? _picked_color : _color;
        //if(_selected) std::cout << _id << std::endl;
        _shader.setTransformationProjectionMatrix(camera.projectionMatrix() * transformationMatrix)
            .setColor(color)
            .setObjectId(_id)
            .draw(_mesh);
    }

    UnsignedInt _id;
    bool _selected;
    Shaders::Flat3D _shader{Shaders::Flat3D::Flag::ObjectId};
    GL::Mesh& _mesh;
    bool _changed = false;
    Color3 _color;
    Color3 _picked_color{0x7b1e16_rgbf};
};

#endif // #ifndef PICKABLE_HPP