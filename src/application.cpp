#include "application.h"
#include <Magnum/Shaders/Flat.h>

#define DEBUG 0
#ifdef DEBUG
#include <iostream>
#endif

// https://github.com/juliettef/IconFontCppHeaders
#include <IconsFontAwesome5.h>

using namespace Magnum;

Application::Application(const Arguments &arguments)
  : Platform::Application{ arguments,
                           Configuration{}
                             .setTitle("BlosSOM")
                             .setWindowFlags(
                               Configuration::WindowFlag::Maximized)
                             .setWindowFlags(
                               Configuration::WindowFlag::Resizable) }
  , _framebuffer{ GL::defaultFramebuffer.viewport() }
  , _ui_imgui(this)
  , _scn_mngr(&state)

{
    MAGNUM_ASSERT_GL_VERSION_SUPPORTED(GL::Version::GL330);

    GL::Renderer::disable(GL::Renderer::Feature::DepthTest);
    GL::Renderer::enable(GL::Renderer::Feature::FaceCulling);

    /* Configure framebuffer. Using a 32-bit int for object ID, which is likely
      enough. Use a smaller type if you have less objects to save memory. */
    _color.setStorage(GL::RenderbufferFormat::RGBA8,
                      GL::defaultFramebuffer.viewport().size());
    _objectId.setStorage(GL::RenderbufferFormat::R32UI,
                         GL::defaultFramebuffer.viewport().size());
    _depth.setStorage(GL::RenderbufferFormat::DepthComponent24,
                      GL::defaultFramebuffer.viewport().size());
    _framebuffer
      .attachRenderbuffer(GL::Framebuffer::ColorAttachment{ 0 }, _color)
      .attachRenderbuffer(GL::Framebuffer::ColorAttachment{ 1 }, _objectId)
      .attachRenderbuffer(GL::Framebuffer::BufferAttachment::Depth, _depth)
      .mapForDraw({ { Shaders::Flat3D::ColorOutput,
                      GL::Framebuffer::ColorAttachment{ 0 } },
                    { Shaders::Flat3D::ObjectIdOutput,
                      GL::Framebuffer::ColorAttachment{ 1 } } });
    CORRADE_INTERNAL_ASSERT(
      _framebuffer.checkStatus(GL::FramebufferTarget::Draw) ==
      GL::Framebuffer::Status::Complete);

    // state = State();
    //_ui_imgui = UiImgui(this);
    //_scn_mngr = SceneMngr(&state);
    // state.lengths);

    /* Loop at 100 Hz max */
    setMinimalLoopPeriod(10);
    setSwapInterval(1);

    timer.tick(); // discard initialization time
    view.set_fb_size(Vector2i(GL::defaultFramebuffer.viewport().sizeX(),
                              GL::defaultFramebuffer.viewport().sizeY()));
}

void
Application::drawEvent()
{
    timer.tick();

    view.update(timer.frametime);
    state.update(timer.frametime);

    GL::defaultFramebuffer.clear(GL::FramebufferClear::Color |
                                 GL::FramebufferClear::Depth);

#if 0

    _scn_mngr.update(&state);
    /* Draw to custom framebuffer */
    _framebuffer.clearColor(0, Color3{ 0.125f })
      .clearColor(1, Vector4ui{})
      .clearDepth(1.0f)
      .bind();

    _scn_mngr.draw_event(&state);

    /* Clear the main buffer. Even though it seems unnecessary, if this is not
       done, it can cause flicker on some drivers. */
    /* Blit color to window framebuffer */
    GL::AbstractFramebuffer::blit(_framebuffer,
                                  GL::defaultFramebuffer,
                                  _framebuffer.viewport(),
                                  GL::FramebufferBlit::Color);

#endif

    graph_renderer.draw(view, state.model, 16);
    _ui_imgui.draw_event(&state, this);

    swapBuffers();
    redraw();
}

void
Application::viewportEvent(ViewportEvent &event)
{
    GL::defaultFramebuffer.setViewport({ {}, event.framebufferSize() });

    view.set_fb_size(event.framebufferSize());

    _framebuffer.setViewport(GL::defaultFramebuffer.viewport());

    /* Configure framebuffer. Using a 32-bit int for object ID, which is likely
      enough. Use a smaller type if you have less objects to save memory. */
    _color.setStorage(GL::RenderbufferFormat::RGBA8,
                      GL::defaultFramebuffer.viewport().size());
    _objectId.setStorage(GL::RenderbufferFormat::R32UI,
                         GL::defaultFramebuffer.viewport().size());
    _depth.setStorage(GL::RenderbufferFormat::DepthComponent24,
                      GL::defaultFramebuffer.viewport().size());
    _framebuffer
      .attachRenderbuffer(GL::Framebuffer::ColorAttachment{ 0 }, _color)
      .attachRenderbuffer(GL::Framebuffer::ColorAttachment{ 1 }, _objectId)
      .attachRenderbuffer(GL::Framebuffer::BufferAttachment::Depth, _depth)
      .mapForDraw({ { Shaders::Flat3D::ColorOutput,
                      GL::Framebuffer::ColorAttachment{ 0 } },
                    { Shaders::Flat3D::ObjectIdOutput,
                      GL::Framebuffer::ColorAttachment{ 1 } } });
    CORRADE_INTERNAL_ASSERT(
      _framebuffer.checkStatus(GL::FramebufferTarget::Draw) ==
      GL::Framebuffer::Status::Complete);

    _scn_mngr.viewport_event(event);
    _ui_imgui.viewport_event(event);
}

void
Application::keyPressEvent(KeyEvent &event)
{
    if (_ui_imgui.key_press_event(event))
        return;

    const float speed = 0.1;
    switch (event.key()) {
        case KeyEvent::Key::Left:
            // all stuff is scaled to vertical size, so the .y() here is OK
            view.mid_target.x() -= view.view_size().y() * speed;
            break;
        case KeyEvent::Key::Right:
            view.mid_target.x() += view.view_size().y() * speed;
            break;
        case KeyEvent::Key::Down:
            view.mid_target.y() -= view.view_size().y() * speed;
            break;
        case KeyEvent::Key::Up:
            view.mid_target.y() += view.view_size().y() * speed;
            break;
    }
    event.setAccepted();
}

void
Application::keyReleaseEvent(KeyEvent &event)
{
    if (_ui_imgui.key_release_event(event))
        return;
}

void
Application::mousePressEvent(MouseEvent &event)
{
    if (_ui_imgui.mouse_press_event(event)) {
        event.setAccepted(true);
        return;
    }

    if (event.button() == MouseEvent::Button::Middle) {
#ifdef DEBUG
        std::cout << view.screen_mouse_coords(event.position()).x()
                  << ", "
                  << view.screen_mouse_coords(event.position()).y()
                  << std::endl;
#endif

        view.center(event.position());
        return;
    }

    state.mouse_pressed = true;

    const Vector2i position =
      event.position() * Vector2{ framebufferSize() } / Vector2{ windowSize() };
    const Vector2i fbPosition{ position.x(),
                               GL::defaultFramebuffer.viewport().sizeY() -
                                 position.y() - 1 };

    _mouse_prev_pos = _mouse_press_pos = fbPosition;
    // state.mouse_pos = {0, 0};
    state.mouse_pos =
      Vector2(fbPosition - (GL::defaultFramebuffer.viewport().size() / 2));

    /* Read object ID at given click position, and then switch to the color
       attachment again so drawEvent() blits correct buffer */
    _framebuffer.mapForRead(GL::Framebuffer::ColorAttachment{ 1 });
    Image2D data = _framebuffer.read(Range2Di::fromSize(fbPosition, { 1, 1 }),
                                     { PixelFormat::R32UI });
    _framebuffer.mapForRead(GL::Framebuffer::ColorAttachment{ 0 });

    /* Highlight object under mouse and deselect all other */
    state.vtx_selected = true;
    UnsignedInt id = data.pixels<UnsignedInt>()[0][0];
    state.vtx_ind = id;
}

void
Application::mouseReleaseEvent(MouseEvent &event)
{
    if (_ui_imgui.mouse_release_event(event)) {
        event.setAccepted(true);
        return;
    }

    // TODO add change speed of draging
    float speed = 2.5f;

    const Vector2i position =
      event.position() * Vector2{ framebufferSize() } / Vector2{ windowSize() };
    const Vector2i fbPosition{ position.x(),
                               GL::defaultFramebuffer.viewport().sizeY() -
                                 position.y() - 1 };

    //    state.mouse_pos = Vector2(fbPosition - _mouse_press_pos);
    state.mouse_pos =
      Vector2(fbPosition - (GL::defaultFramebuffer.viewport().size() / 2));

    _mouse_press_pos = fbPosition;
    state.mouse_pressed = false;
    state.vtx_selected = false;

    // if(delta != Vector2d(0, 0)) {
    //     auto norm_delta = state.mouse_pos.normalized();
    //     _cameraObject->translate(Vector3::xAxis(-float(speed *
    //     norm_delta.x())));
    //     _cameraObject->translate(Vector3::zAxis(-float(speed
    //     * norm_delta.y()))); _camera_trans.x() = -float(speed *
    //     norm_delta.x()); _camera_trans.z() = -float(speed * norm_delta.y());
    // }
}

void
Application::mouseMoveEvent(MouseMoveEvent &event)
{
    if (_ui_imgui.mouse_move_event(event)) {
        event.setAccepted(true);
        return;
    }

    if (state.mouse_pressed) {
        state.vtx_selected = true;

        const Vector2i position = event.position() *
                                  Vector2{ framebufferSize() } /
                                  Vector2{ windowSize() };
        const Vector2i fbPosition{ position.x(),
                                   GL::defaultFramebuffer.viewport().sizeY() -
                                     position.y() - 1 };

        // state.mouse_pos = Vector2(fbPosition - _mouse_prev_pos);
        state.mouse_pos =
          Vector2(fbPosition - (GL::defaultFramebuffer.viewport().size() / 2));

        _mouse_prev_pos = fbPosition;
    }
}

void
Application::mouseScrollEvent(MouseScrollEvent &event)
{
    if (_ui_imgui.mouse_scroll_event(event)) {
        event.setAccepted();
        return;
    }

    view.zoom(0.2 * event.offset().y(), event.position());
    event.setAccepted();
}

void
Application::textInputEvent(TextInputEvent &event)
{
    if (_ui_imgui.text_input_event(event)) {
        event.setAccepted(true);
        return;
    }
}

Vector3
Application::windowPos2WorldPos(const Vector2i &windowPosition)
{
    /* First scale the position from being relative to window size to being
       relative to framebuffer size as those two can be different on HiDPI
       systems */
    const Vector2i position =
      windowPosition * Vector2{ framebufferSize() } / Vector2{ windowSize() };

    /* Compute inverted model view projection matrix */
    // const Matrix3 invViewProjMat =
    // (_camera->projectionMatrix()*_camera->cameraMatrix()).inverted();
    const Matrix4 invViewProjMat = _scn_mngr.getInvViewProjMat();

    /* Compute the world coordinate from window coordinate */
    const Vector2i flippedPos =
      Vector2i(position.x(), framebufferSize().y() - position.y());
    const Vector2 ndcPos =
      Vector2(flippedPos) / Vector2(framebufferSize()) * Vector2{ 2.0f } -
      Vector2{ 1.0f };
    return invViewProjMat.transformPoint({ ndcPos.x(), ndcPos.y(), 0.0f });
}

MAGNUM_APPLICATION_MAIN(Application)
