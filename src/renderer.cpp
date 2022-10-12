#include "renderer.h"

#include <iostream>

#include "shaders.h"
#include "glm/gtc/matrix_transform.hpp"

Renderer::Renderer() {}

bool
Renderer::init()
{
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_DEPTH_TEST);

    scatter_renderer.init();
    graph_renderer.init();

    return true;
}


void
Renderer::update(State &state, View &view, const CallbackValues &callbacks)
{
    process_mouse(view, callbacks);

    render(state, view);
}

void
Renderer::render(State &state, View &view)
{
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    scatter_renderer.draw(view, state.scatter, state.colors);
    graph_renderer.draw(view, state.landmarks);
}

void Renderer::process_mouse(const View &view, const CallbackValues &cb)
{
    if (cb.button == GLFW_MOUSE_BUTTON_LEFT && cb.mouse_action == GLFW_PRESS)
    {
        glm::vec2 screen_mouse = view.screen_mouse_coords(glm::vec2(cb.xpos, cb.ypos));
        size_t point_ind;
        // std::cout << "xpos: " << cb.xpos << "ypos: " << cb.ypos << std::endl;
        //std::cout << "screen_mouse: " << screen_mouse.x << ", " << screen_mouse.y << std::endl;
        if(graph_renderer.is_vert_pressed(view, screen_mouse, point_ind))
        {
            std::cout << "point_ind: " << point_ind << std::endl;
        }        
    }
}
