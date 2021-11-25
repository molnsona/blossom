
#ifndef UI_MENU_H
#define UI_MENU_H

#include "ui_data.h"
#include "ui_load.h"
#include "ui_train.h"

class Application;

struct uiMenu
{
    uiLoader loader;
    uiTrainingSettings training_set;

    uiMenu();
    void render(Application &app);
    void close_menu() { show_menu = false; }

    // legacy part follows
    UiData ui;

private:
    void draw_menu_window(const Vector2i &window_size, UiData &ui);

    void draw_scale_window(UiTransData &ui);
    void draw_color_window(UiData &ui);

    bool show_menu;
    bool show_scale;
    bool show_color;
};

#include "application.h"

#endif
