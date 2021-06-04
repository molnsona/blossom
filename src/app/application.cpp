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
    
    _p_state = std::make_unique<State>();
    _p_ui_imgui = std::make_unique<UiImgui>(this);
    _p_scn_mngr = std::make_unique<SceneMngr>(_p_state.get(), _scene, _drawables);
    
    /* Loop at 60 Hz max */
    setSwapInterval(1);
    setMinimalLoopPeriod(16);
}

void Application::drawEvent() {
    GL::defaultFramebuffer.clear(GL::FramebufferClear::Color|GL::FramebufferClear::Depth);
   
    _p_scn_mngr->draw_event(_p_state.get(), _drawables);
    _p_ui_imgui->draw_event(_p_state.get(), this);

    swapBuffers();
    redraw();
}

void Application::viewportEvent(ViewportEvent& event) {
    GL::defaultFramebuffer.setViewport({{}, event.framebufferSize()});

    _p_scn_mngr->viewport_event(event);
    _p_ui_imgui->viewport_event(event);
}

void Application::keyPressEvent(KeyEvent& event) {
    if(_p_ui_imgui->key_press_event(event)) return;

    // /* Movement */
    // float speed = 0.5f;
    // switch(event.key()) 
    // {
    //     case KeyEvent::Key::Down:
    //         _cameraObject->translate(Vector3::zAxis(speed));
    //         _camera_trans.z() += speed;
    //         break;
    //     case KeyEvent::Key::Up:
    //         _cameraObject->translate(Vector3::zAxis(-speed));
    //         _camera_trans.z() -= speed;
    //         break;
    //     case KeyEvent::Key::Left:
    //         _cameraObject->translate(Vector3::xAxis(-speed));
    //         _camera_trans.x() -= speed;
    //         break;
    //     case KeyEvent::Key::Right:
    //         _cameraObject->translate(Vector3::xAxis(speed));
    //         _camera_trans.x() += speed;
    //         break;
    //     case KeyEvent::Key::Space: {
    //             Vector2 view_size(++_zoom_depth.x(), +_zoom_depth.y());
    //             _camera->setAspectRatioPolicy(SceneGraph::AspectRatioPolicy::Extend)
    //                 .setProjectionMatrix(Matrix4::orthographicProjection(view_size, 0.001f, 100.0f))
    //                 .setViewport(GL::defaultFramebuffer.viewport().size());
    //             break;
    //         }
    //     case KeyEvent::Key::Esc: {
    //             Vector2 view_size(--_zoom_depth.x(), -_zoom_depth.y());
    //             _camera->setAspectRatioPolicy(SceneGraph::AspectRatioPolicy::Extend)
    //                 .setProjectionMatrix(Matrix4::orthographicProjection(view_size, 0.001f, 100.0f))
    //                 .setViewport(GL::defaultFramebuffer.viewport().size());
    //             break;
    //         }
    //     // Reset camera, TODO
    //     case KeyEvent::Key::Tab: {
    //             _zoom_depth = {PLOT_WIDTH, PLOT_HEIGHT};
    //             Vector2 view_size(_zoom_depth);
    //             _camera->setAspectRatioPolicy(SceneGraph::AspectRatioPolicy::Extend)
    //                 .setProjectionMatrix(Matrix4::orthographicProjection(view_size, 0.001f, 100.0f))
    //                 .setViewport(GL::defaultFramebuffer.viewport().size());
    //             (*_cameraObject)
    //                 .translate(-_camera_trans)
    //             // .translate(Vector3::yAxis(5.0f))
    //                 .rotateX(-90.0_degf);
    //             _camera_trans = Vector3(0.0f, 0.0f, 0.0f);
    //             break;
    //         }            
    // }
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

    _p_state->mouse_press_pos = event.position();
}

void Application::mouseReleaseEvent(MouseEvent& event) {
   if(_p_ui_imgui->mouse_release_event(event)) {
        event.setAccepted(true);
        return;
   }

    // TODO add change speed of draging
    float speed = 2.5f;
    _p_state->mouse_delta = Magnum::Vector2d(event.position()) - Magnum::Vector2d(_p_state->mouse_press_pos);
    // if(delta != Vector2d(0, 0)) {
    //     auto norm_delta = _p_state->mouse_delta.normalized();
    //     _cameraObject->translate(Vector3::xAxis(-float(speed * norm_delta.x())));
    //     _cameraObject->translate(Vector3::zAxis(-float(speed * norm_delta.y())));
    //     _camera_trans.x() = -float(speed * norm_delta.x());
    //     _camera_trans.z() = -float(speed * norm_delta.y());
    // }
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

    // Vector2 view_size{_zoom_depth};
    // if(1.0f - (event.offset().y()) > 0) 
    //     view_size = Vector2(++_zoom_depth.x(), +_zoom_depth.y());
    // else {
    //     if(--_zoom_depth.x() <= 0) +_zoom_depth.x();
    //     if(--_zoom_depth.y() <= 0) +_zoom_depth.y();
    //     view_size = _zoom_depth;
    // }

    // _camera->setAspectRatioPolicy(SceneGraph::AspectRatioPolicy::Extend)
    // .setProjectionMatrix(Matrix4::orthographicProjection(view_size, 0.001f, 100.0f))
    // .setViewport(GL::defaultFramebuffer.viewport().size());
}

void Application::textInputEvent(TextInputEvent& event) {
    if(_p_ui_imgui->text_input_event(event)) {
        event.setAccepted(true);
        return;
    }
}

MAGNUM_APPLICATION_MAIN(Application)
