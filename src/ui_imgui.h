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
