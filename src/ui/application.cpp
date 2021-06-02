#include "application.h"

#include <iostream>

// https://github.com/juliettef/IconFontCppHeaders
#include <IconsFontAwesome5.h>

using namespace Magnum;

Application::Application(const Arguments& arguments):
    Platform::Application{arguments, Configuration{}
        .setTitle("BlosSOM")
        .setWindowFlags(Configuration::WindowFlag::Maximized)}
{
    GL::Renderer::enable(GL::Renderer::Feature::DepthTest);
    GL::Renderer::enable(GL::Renderer::Feature::FaceCulling);
    
    _zoom_depth = {PLOT_WIDTH - 40, PLOT_HEIGHT - 40};

    _p_ui_imgui = std::make_unique<UiImgui>(this);
    _p_canvas = std::make_unique<Canvas>(_scene, _drawables);
    // /* Set up meshes */
    // _plane = MeshTools::compile(Primitives::squareSolid(Primitives::SquareFlag::TextureCoordinates));    
    //  /* Set up objects */
    // (*(_objects[0] = new PickableObject{3, _textured_shader, _scene, _drawables}))
    //     .rotateX(-90.0_degf)        
    //     .scale(Vector3{PLOT_WIDTH / 2, 0.0f, PLOT_HEIGHT / 2});
    
    /* Configure camera */
    _cameraObject = new Object3D{&_scene};
    (*_cameraObject)
        .translate(Vector3::zAxis(20.0f))
        .rotateX(-90.0_degf);
    _camera_trans.z() = 20.0f;

    _camera = new SceneGraph::Camera3D{*_cameraObject};
    _camera->setAspectRatioPolicy(SceneGraph::AspectRatioPolicy::Extend)
        .setProjectionMatrix(Matrix4::orthographicProjection(_zoom_depth, 0.001f, 100.0f))
        .setViewport(GL::defaultFramebuffer.viewport().size());

    
    /* Loop at 60 Hz max */
    setSwapInterval(1);
    setMinimalLoopPeriod(16);
}

void Application::drawEvent() {
    GL::defaultFramebuffer.clear(GL::FramebufferClear::Color|GL::FramebufferClear::Depth);
   
    _p_canvas->draw_event();

    _camera->draw(_drawables);

    _p_ui_imgui->draw_event(this);

    swapBuffers();
    redraw();
}

void Application::viewportEvent(ViewportEvent& event) {
    GL::defaultFramebuffer.setViewport({{}, event.framebufferSize()});
    _camera->setViewport(event.windowSize());

    _p_ui_imgui->viewport_event(event);
}

void Application::keyPressEvent(KeyEvent& event) {
    if(_p_ui_imgui->key_press_event(event)) return;

    /* Movement */
    float speed = 0.5f;
    switch(event.key()) 
    {
        case KeyEvent::Key::Down:
            _cameraObject->translate(Vector3::zAxis(speed));
            _camera_trans.z() += speed;
            break;
        case KeyEvent::Key::Up:
            _cameraObject->translate(Vector3::zAxis(-speed));
            _camera_trans.z() -= speed;
            break;
        case KeyEvent::Key::Left:
            _cameraObject->translate(Vector3::xAxis(-speed));
            _camera_trans.x() -= speed;
            break;
        case KeyEvent::Key::Right:
            _cameraObject->translate(Vector3::xAxis(speed));
            _camera_trans.x() += speed;
            break;
        case KeyEvent::Key::Space: {
                Vector2 view_size(++_zoom_depth.x(), +_zoom_depth.y());
                _camera->setAspectRatioPolicy(SceneGraph::AspectRatioPolicy::Extend)
                    .setProjectionMatrix(Matrix4::orthographicProjection(view_size, 0.001f, 100.0f))
                    .setViewport(GL::defaultFramebuffer.viewport().size());
                break;
            }
        case KeyEvent::Key::Esc: {
                Vector2 view_size(--_zoom_depth.x(), -_zoom_depth.y());
                _camera->setAspectRatioPolicy(SceneGraph::AspectRatioPolicy::Extend)
                    .setProjectionMatrix(Matrix4::orthographicProjection(view_size, 0.001f, 100.0f))
                    .setViewport(GL::defaultFramebuffer.viewport().size());
                break;
            }
        // Reset camera, TODO
        case KeyEvent::Key::Tab: {
                _zoom_depth = {PLOT_WIDTH, PLOT_HEIGHT};
                Vector2 view_size(_zoom_depth);
                _camera->setAspectRatioPolicy(SceneGraph::AspectRatioPolicy::Extend)
                    .setProjectionMatrix(Matrix4::orthographicProjection(view_size, 0.001f, 100.0f))
                    .setViewport(GL::defaultFramebuffer.viewport().size());
                (*_cameraObject)
                    .translate(-_camera_trans)
                // .translate(Vector3::yAxis(5.0f))
                    .rotateX(-90.0_degf);
                _camera_trans = Vector3(0.0f, 0.0f, 0.0f);
                break;
            }            
    }
    event.setAccepted();
}

void Application::keyReleaseEvent(KeyEvent& event) {
   if(_p_ui_imgui->key_release_event(event)) return;
}

void Application::mousePressEvent(MouseEvent& event) {
    if(_p_ui_imgui->mouse_press_event(event)) {
        event.setAccepted(true);
        return;
    }

    _mousePressPosition = event.position();
}

void Application::mouseReleaseEvent(MouseEvent& event) {
   if(_p_ui_imgui->mouse_release_event(event)) {
        event.setAccepted(true);
        return;
   }

    // TODO add change speed of draging
    float speed = 2.5f;
    auto delta = Magnum::Vector2d(event.position()) - Magnum::Vector2d(_mousePressPosition);
    if(delta != Vector2d(0, 0)) {
        auto norm_delta = delta.normalized();
        _cameraObject->translate(Vector3::xAxis(-float(speed * norm_delta.x())));
        _cameraObject->translate(Vector3::zAxis(-float(speed * norm_delta.y())));
        _camera_trans.x() = -float(speed * norm_delta.x());
        _camera_trans.z() = -float(speed * norm_delta.y());
    }
}

void Application::mouseMoveEvent(MouseMoveEvent& event) {
   if(_p_ui_imgui->mouse_move_event(event)) {
        event.setAccepted(true);
        return;
    }
}

void Application::mouseScrollEvent(MouseScrollEvent& event) {
    if(_p_ui_imgui->mouse_scroll_event(event)) {
        /* Prevent scrolling the page */
        event.setAccepted();
        return;
    }

    if(!event.offset().y()) return;

    Vector2 view_size{_zoom_depth};
    if(1.0f - (event.offset().y()) > 0) 
        view_size = Vector2(++_zoom_depth.x(), +_zoom_depth.y());
    else {
        if(--_zoom_depth.x() <= 0) +_zoom_depth.x();
        if(--_zoom_depth.y() <= 0) +_zoom_depth.y();
        view_size = _zoom_depth;
    }

    _camera->setAspectRatioPolicy(SceneGraph::AspectRatioPolicy::Extend)
    .setProjectionMatrix(Matrix4::orthographicProjection(view_size, 0.001f, 100.0f))
    .setViewport(GL::defaultFramebuffer.viewport().size());
}

void Application::textInputEvent(TextInputEvent& event) {
    if(_p_ui_imgui->text_input_event(event)) {
        event.setAccepted(true);
        return;
    }
}

MAGNUM_APPLICATION_MAIN(Application)
