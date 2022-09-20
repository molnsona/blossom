
#include <iostream>

#include "renderer.h"
#include "state.h"
#include "timer.h"
#include "wrapper_glad.h"
#include "wrapper_glfw.h"
#include "wrapper_imgui.h"

int
main()
{
    GlfwWrapper glfw;
    GladWrapper glad;
    ImGuiWrapper imgui;
    Renderer renderer;

    Timer timer;
    State state;

    if (!glfw.init("BlosSOM")) {
        std::cout << "GLFW initialization failed." << std::endl;
        return -1;
    }

    if (!glad.init()) {
        std::cout << "GLAD initialization failed." << std::endl;
        glfw.destroy();
        return -1;
    }

    if (!imgui.init(glfw.window)) {
        std::cout << "Dear ImGui initialization failed." << std::endl;
        glfw.destroy();
        return -1;
    }

    if (!renderer.init()) {
        std::cout << "Renderer initialization failed." << std::endl;
        imgui.destroy();
        glfw.destroy();
        return -1;
    }

    int x, y, width, height;
    while (!glfw.window_should_close()) {
        timer.tick();

        state.update(timer.frametime);

        renderer.render(state);

        imgui.render(glfw.callbacks, state);

        glfw.end_frame();
    }

    imgui.destroy();
    glfw.destroy();
    return 0;
}