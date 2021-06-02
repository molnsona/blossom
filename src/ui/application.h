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
#include <Magnum/Primitives/Square.h>
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
#include <Magnum/GL/PixelFormat.h>
#include <Magnum/ImageView.h>

#include <memory>

#include "imgui_utils.hpp"
//#include "../utils.hpp"
#include "canvas.h"
#include "graph.h"
#include "ui_imgui.h"

//class UiImgui;

using namespace Magnum;
using namespace Math::Literals;
using namespace std::chrono;

typedef SceneGraph::Object<SceneGraph::MatrixTransformation3D> Object3D;
typedef SceneGraph::Scene<SceneGraph::MatrixTransformation3D> Scene3D;

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

    std::unique_ptr<Canvas> _p_canvas;
    std::unique_ptr<Graph> _p_graph;
    std::unique_ptr<UiImgui> _p_ui_imgui;
    
    Shaders::Flat3D _textured_shader{Shaders::Flat3D::Flag::Textured};
    Color3 _bg_color{0xffffff_rgbf};

    GL::Mesh _plane;
    Scene3D _scene;
    SceneGraph::Camera3D* _camera;
    Object3D* _cameraObject;
    SceneGraph::DrawableGroup3D _drawables;

    Vector2i _previousMousePosition, _mousePressPosition;

    Vector2 _zoom_depth;


    Vector3 _camera_trans = Vector3{0.0f, 0.0f, 0.0f};
};

#endif // #ifndef APPLICATION_H