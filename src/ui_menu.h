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

#ifndef UI_MENU_H
#define UI_MENU_H

#include "ui_color.h"
#include "ui_load.h"
#include "ui_save.h"
#include "ui_scale.h"
#include "ui_train.h"

class Application;

/**
 * @brief ImGUI handler for rendering main menu window.
 *
 * It also holds handlers of all menu item windows.
 *
 */
struct uiMenu
{
    /** Open file dialog window handler. */
    uiLoader loader;
    /** Save file dialog window handler.*/
    uiSaver saver;
    /** Scale&transform data window handler. */
    uiScaler scaler;
    /** Training settings window handler. */
    uiTrainingSettings training_set;
    /** Color setting window handler. */
    uiColorSettings color_set;

    uiMenu();
    /**
     * @brief Renders main menu window, the `plus` button and currently opened
     * menu item windows.
     *
     * @param app Application context.
     */
    void render(Application &app);
    /**
     * @brief Closes main menu window.
     *
     */
    void close_menu() { show_menu = false; }

private:
    /**
     * @brief Draws main menu window.
     *
     * @param window_size Size of the main application window used for placement
     * of the main menu window.
     */
    void draw_menu_window(const Vector2i &window_size);

    /** If the main menu window should be rendered. */
    bool show_menu;
};

#include "application.h"

#endif
