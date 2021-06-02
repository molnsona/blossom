#ifndef CANVAS_H
#define CANVAS_H

#include <Magnum/SceneGraph/MatrixTransformation3D.h>
#include <Magnum/SceneGraph/Drawable.h>
#include <Magnum/SceneGraph/Object.h>
#include <Magnum/SceneGraph/Camera.h>
#include <Magnum/Shaders/Flat.h>
#include <Magnum/GL/Texture.h>
#include <Magnum/GL/TextureFormat.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/Math/Color.h>

using namespace Magnum;
using namespace Math::Literals;

typedef SceneGraph::Object<SceneGraph::MatrixTransformation3D> Object3D;
typedef SceneGraph::Scene<SceneGraph::MatrixTransformation3D> Scene3D;

class PickableObject: public Object3D, SceneGraph::Drawable3D {
    public:
        explicit PickableObject(UnsignedInt id, Shaders::Flat3D& shader, GL::Mesh& mesh, Object3D& parent, SceneGraph::DrawableGroup3D& drawables): Object3D{&parent}, SceneGraph::Drawable3D{*this, &drawables}, _id{id}, _selected{false}, _shader(shader), _mesh(mesh)
        { }

        void setConfig(int cell_cnt, int mean, int std_dev) 
        {
            if(_cell_cnt == cell_cnt && _mean == mean && _std_dev == std_dev) 
            {
                _changed = false;
                return;
            }          

            _changed = true;
            _cell_cnt = cell_cnt;
            _mean = mean;                                        
            _std_dev = std_dev;                                             
        }

        void setSelected(bool selected) { _selected = selected; }

    private:
        virtual void draw(const Matrix4& transformationMatrix, SceneGraph::Camera3D& camera);

        UnsignedInt _id;
        bool _selected;
        Shaders::Flat3D& _shader;
        GL::Mesh& _mesh;
        GL::Texture2D _texture;
        int _cell_cnt;
        int _mean;
        int _std_dev;
        bool _changed = false;
};


class Canvas {
public:
    Canvas(Object3D& parent, SceneGraph::DrawableGroup3D& drawables);

    void draw_event();
private:
    PickableObject* _canvas;
    GL::Mesh _canvas_mesh;
    Shaders::Flat3D _textured_shader{Shaders::Flat3D::Flag::Textured};

};

#endif // #ifndef CANVAS_H