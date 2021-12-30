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

#ifndef UI_IMGUI_H
#define UI_IMGUI_H

#include <Magnum/GL/Texture.h>
#include <Magnum/GL/TextureFormat.h>
#include <Magnum/ImGuiIntegration/Context.hpp>
#include <Magnum/Math/Color.h>
#include <Magnum/Platform/Sdl2Application.h>

#include <string>

// TRICKY: do not include this file, cyclic import breaks it.
// Include application.h instead.
#include "application.h"
#include "ui_menu.h"

using namespace Magnum;
using namespace Math::Literals;

/**
 * @brief ImGUI handler that handles all events and sets ImGUI for further
 * rendering of the menu windows.
 *
 */
class UiImgui
{
public:
    UiImgui() = delete;
    UiImgui(const Application &app);

    void draw_event(Application &app);

    void viewport_event(Platform::Application::ViewportEvent &event);
    bool key_press_event(Platform::Application::KeyEvent &event);
    bool key_release_event(Platform::Application::KeyEvent &event);
    bool mouse_press_event(Platform::Application::MouseEvent &event);
    bool mouse_release_event(Platform::Application::MouseEvent &event);
    bool mouse_move_event(Platform::Application::MouseMoveEvent &event);
    bool mouse_scroll_event(Platform::Application::MouseScrollEvent &event);
    bool text_input_event(Platform::Application::TextInputEvent &event);

    uiMenu menu;

private:
    ImGuiIntegration::Context context{ NoCreate };

    ImFont *p_font;
    GL::Texture2D font_texture;
};

#endif // #ifndef UI_IMGUI_H
