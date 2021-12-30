/* This file is part of BlosSOM.
 *
 * Copyright (C) 2021 Mirek Kratochvil
 *                    Sona Molnarova
 *
 * BlosSOM is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option)
 * any later version.
 *
 * BlosSOM is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * BlosSOM. If not, see <https://www.gnu.org/licenses/>.
 */

#include "application.h"
#include <Magnum/Shaders/Flat.h>

using namespace Magnum;

Application::Application(const Arguments &arguments)
  : Platform::Application{ arguments,
                           Configuration{}
                             .setTitle("BlosSOM")
                             .setWindowFlags(
                               Configuration::WindowFlag::Maximized)
                             .setWindowFlags(
                               Configuration::WindowFlag::Resizable) }
  , ui_imgui(*this)
{
    MAGNUM_ASSERT_GL_VERSION_SUPPORTED(GL::Version::GL330);

    GL::Renderer::disable(GL::Renderer::Feature::DepthTest);
    GL::Renderer::enable(GL::Renderer::Feature::FaceCulling);

    // Set white background
    GL::Renderer::setClearColor(0xffffff_rgbf);

    // Loop at 100 Hz max
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

    scatter_renderer.draw(view, state.scatter, state.colors);
    graph_renderer.draw(view, state.landmarks);
    ui_imgui.draw_event(*this);

    swapBuffers();
    redraw();
}

void
Application::viewportEvent(ViewportEvent &event)
{
    GL::defaultFramebuffer.setViewport({ {}, event.framebufferSize() });

    view.set_fb_size(event.framebufferSize());

    ui_imgui.viewport_event(event);
}

void
Application::keyPressEvent(KeyEvent &event)
{
    if (ui_imgui.key_press_event(event))
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
        case KeyEvent::Key::LeftCtrl:
            state.keyboard.ctrl_pressed = true;
            break;
        case KeyEvent::Key::RightCtrl:
            state.keyboard.ctrl_pressed = true;
            break;
    }
    event.setAccepted();
}

void
Application::keyReleaseEvent(KeyEvent &event)
{
    if (ui_imgui.key_release_event(event))
        return;

    switch (event.key()) {
        case KeyEvent::Key::LeftCtrl:
            state.keyboard.ctrl_pressed = false;
            break;
        case KeyEvent::Key::RightCtrl:
            state.keyboard.ctrl_pressed = false;
            break;
    }
    event.setAccepted();
}

void
Application::mousePressEvent(MouseEvent &event)
{
    if (ui_imgui.mouse_press_event(event)) {
        event.setAccepted(true);
        return;
    }

    // Close menu if it was opened and clicked somewhere else.
    ui_imgui.menu.close_menu();

    switch (event.button()) {
        case MouseEvent::Button::Middle:
            view.lookat_screen(event.position());
            return;
        case MouseEvent::Button::Left:
            state.mouse.left_pressed = true;
            state.mouse.mouse_pos = event.position();

            if (graph_renderer.is_vert_pressed(
                  Vector2(view.screen_mouse_coords(state.mouse.mouse_pos)),
                  state.mouse.vert_ind)) {
                if (state.keyboard.ctrl_pressed) {
                    state.landmarks.duplicate(state.mouse.vert_ind);
                } else {
                    state.mouse.vert_pressed = true;
                    state.landmarks.move(
                      state.mouse.vert_ind,
                      view.model_mouse_coords(state.mouse.mouse_pos));
                }
            } else if (state.keyboard.ctrl_pressed) {
                state.landmarks.add(
                  view.model_mouse_coords(state.mouse.mouse_pos));
            }

            break;
        case MouseEvent::Button::Right:
            state.mouse.right_pressed = true;
            state.mouse.mouse_pos = event.position();

            if (graph_renderer.is_vert_pressed(
                  Vector2(view.screen_mouse_coords(state.mouse.mouse_pos)),
                  state.mouse.vert_ind)) {
                if (state.keyboard.ctrl_pressed) {
                    state.landmarks.remove(state.mouse.vert_ind);
                }
            }
            break;
    }
}

void
Application::mouseReleaseEvent(MouseEvent &event)
{
    if (ui_imgui.mouse_release_event(event)) {
        event.setAccepted(true);
        return;
    }

    if (event.button() != MouseEvent::Button::Left) {
        event.setAccepted(true);
        return;
    }

    state.mouse.mouse_pos = event.position();
    state.mouse.left_pressed = false;
    state.mouse.vert_pressed = false;
}

void
Application::mouseMoveEvent(MouseMoveEvent &event)
{
    if (ui_imgui.mouse_move_event(event)) {
        event.setAccepted(true);
        return;
    }

    if (state.mouse.vert_pressed) {
        state.mouse.mouse_pos = event.position();
        state.landmarks.move(state.mouse.vert_ind,
                             view.model_mouse_coords(state.mouse.mouse_pos));
    }
}

void
Application::mouseScrollEvent(MouseScrollEvent &event)
{
    if (ui_imgui.mouse_scroll_event(event)) {
        event.setAccepted();
        return;
    }

    view.zoom(0.2 * event.offset().y(), event.position());
    event.setAccepted();
}

void
Application::textInputEvent(TextInputEvent &event)
{
    if (ui_imgui.text_input_event(event)) {
        event.setAccepted(true);
        return;
    }
}

MAGNUM_APPLICATION_MAIN(Application)
