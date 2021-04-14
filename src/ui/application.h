#ifndef APPLICATION_H
#define APPLICATION_H

#include <Corrade/Containers/StridedArrayView.h>
#include <Corrade/Containers/Reference.h>
#include <Corrade/Utility/Resource.h>
#include <Magnum/Image.h>
#include <Magnum/PixelFormat.h>
#include <Magnum/GL/Context.h>
#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/Framebuffer.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/GL/Renderbuffer.h>
#include <Magnum/GL/RenderbufferFormat.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/GL/Texture.h>
#include <Magnum/GL/TextureFormat.h>
#include <Magnum/GL/Version.h>
#include <Magnum/Math/Color.h>
#include <Magnum/MeshTools/Compile.h>
#include <Magnum/Platform/Sdl2Application.h>
#include <Magnum/Primitives/Cube.h>
#include <Magnum/Primitives/Plane.h>
#include <Magnum/Primitives/UVSphere.h>
#include <Magnum/Trade/MeshData.h>
#include <Magnum/SceneGraph/Scene.h>
#include <Magnum/SceneGraph/MatrixTransformation3D.h>
#include <Magnum/SceneGraph/Camera.h>
#include <Magnum/SceneGraph/Drawable.h>
#include <Magnum/Shaders/Phong.h>
#include <Magnum/Shaders/Flat.h>

#include <Magnum/GL/Buffer.h>
#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/Math/Color.h>
#include <Magnum/ImGuiIntegration/Context.hpp>
#include <Magnum/Platform/Sdl2Application.h>
#include <Magnum/Shaders/VertexColor.h>

#include "imgui_utils.hpp"

using namespace Magnum;
using namespace Math::Literals;
using namespace std::chrono;

namespace fazul{


typedef SceneGraph::Object<SceneGraph::MatrixTransformation3D> Object3D;
typedef SceneGraph::Scene<SceneGraph::MatrixTransformation3D> Scene3D;

class PickableObject: public Object3D, SceneGraph::Drawable3D {
    public:
        explicit PickableObject(UnsignedInt id, Shaders::Flat3D& shader, const Color3& color, GL::Mesh& mesh, Object3D& parent, SceneGraph::DrawableGroup3D& drawables): Object3D{&parent}, SceneGraph::Drawable3D{*this, &drawables}, _id{id}, _selected{false}, _shader(shader), _color{color}, _mesh(mesh) {}

        void setSelected(bool selected) { _selected = selected; }

    private:
        virtual void draw(const Matrix4& transformationMatrix, SceneGraph::Camera3D& camera) {
            _shader.setTransformationProjectionMatrix(camera.projectionMatrix() * transformationMatrix)
                //.setNormalMatrix(transformationMatrix.normalMatrix())
                //.setProjectionMatrix(camera.projectionMatrix())
                //.setAmbientColor(_selected ? _color*0.3f : Color3{})
                //.setDiffuseColor(_color*(_selected ? 2.0f : 1.0f))
                //.setShininess(20000.0f)
                /* relative to the camera */
                //.setLightPositions({{0.0f, 0.0f, 1000.0f, 0.0f}})
                //.setObjectId(_id)
                .draw(_mesh);
        }

        UnsignedInt _id;
        bool _selected;
        Shaders::Flat3D& _shader;
        Color3 _color;
        GL::Mesh& _mesh;
};

class Application: public Platform::Application {
public: 
    explicit Application(const Arguments& arguments);

private:
    void drawEvent() override;

    void viewportEvent(ViewportEvent& event) override;

    void keyPressEvent(KeyEvent& event) override;
    void keyReleaseEvent(KeyEvent& event) override;

    void mousePressEvent(MouseEvent& event) override;
    void mouseReleaseEvent(MouseEvent& event) override;
    void mouseMoveEvent(MouseMoveEvent& event) override;
    void mouseScrollEvent(MouseScrollEvent& event) override;
    void textInputEvent(TextInputEvent& event) override;

    void init_imgui();
    
    struct ImGuiVars {
        ImGuiIntegration::Context _imgui{NoCreate};

        Color4 _clearColor = 0x72909aff_rgbaf;
        Float _floatValue = 0.0f;
    };

    std::string _domain;
    ImGuiVars _imgui_vars;
    
    Shaders::Flat3D _textured_shader{Shaders::Flat3D::Flag::Textured};
    Color3 _bg_color{0xffffff_rgbf};

    GL::Mesh _plane;
    Scene3D _scene;
    SceneGraph::Camera3D* _camera;
    Object3D* _cameraObject;
    SceneGraph::DrawableGroup3D _drawables;
    PickableObject* _objects[1];

    Vector2i _previousMousePosition, _mousePressPosition;

    float zoom_depth = 40.0f;

    ImFont* _p_font;
	GL::Texture2D font_texture;

    bool show_tools = false;
    Vector3 camera_trans = Vector3{0.0f, 0.0f, 0.0f};
};



}

#endif // #ifndef APPLICATION_H