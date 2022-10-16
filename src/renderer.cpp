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
    process_keyboard(state, view, callbacks);
    process_mouse(state, view, callbacks);

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

void Renderer::process_mouse(State &state, const View &view, const CallbackValues &cb)
{        
    if (cb.button == GLFW_MOUSE_BUTTON_LEFT && cb.mouse_action == GLFW_PRESS)
    {
        if(!state.mouse.vert_pressed) {
            glm::vec2 screen_mouse = view.screen_mouse_coords(glm::vec2(cb.xpos, cb.ypos));
            if(graph_renderer.is_vert_pressed(view, screen_mouse, state.mouse.vert_ind))
            {
                state.mouse.vert_pressed = true;
            }
        }

        if(state.keyboard.ctrl_pressed)
            // Copy pressed landmark
            if(state.mouse.vert_pressed)        
                state.landmarks.duplicate(state.mouse.vert_ind);
            // Add landmark
            else            
                state.landmarks.add(view.model_mouse_coords(glm::vec2(cb.xpos, cb.ypos)));             
    }

    if (cb.button == GLFW_MOUSE_BUTTON_LEFT && cb.mouse_action == GLFW_RELEASE)
    {
        state.mouse.vert_pressed = false;
    }

    if(state.mouse.vert_pressed)
    {
        if(!state.keyboard.ctrl_pressed)
        { // Move landmark
            glm::vec2 model_mouse = view.model_mouse_coords(glm::vec2(cb.xpos, cb.ypos));

            state.landmarks.move(state.mouse.vert_ind, model_mouse);
        }
    }   
}

void Renderer::process_keyboard(State &state, const View &view, const CallbackValues &cb)
{  
    if (cb.key == GLFW_KEY_LEFT_CONTROL && (cb.key_action == GLFW_PRESS || cb.key_action == GLFW_REPEAT))
    {
        state.keyboard.ctrl_pressed = true;
    } else if(cb.key == GLFW_KEY_LEFT_CONTROL && cb.key_action == GLFW_RELEASE)
    {
        state.keyboard.ctrl_pressed = false;
    }

}
