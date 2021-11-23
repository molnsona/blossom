
#ifndef UI_MENU_H
#define UI_MENU_H

#include "ui_data.h"
#include "ui_load.h"

class Application;

struct uiMenu
{
    uiLoader loader;

    uiMenu();
    void render(Application &app);
    void close_menu() { show_menu = false; }

    // legacy part follows
    void draw_add_window(const Vector2i &window_size);
    void draw_menu_window(const Vector2i &window_size, UiData &ui);
    void draw_scale_window(UiTransData &ui);
    void draw_sliders_window(UiSlidersData &ui);
    void draw_color_window(UiData &ui);
    void hover_info(const std::string &text);

    bool show_menu;
    bool show_scale;
    bool show_sliders;
    bool show_color;
    UiData ui;
};

#include "application.h"

#endif
