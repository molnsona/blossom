#include "renderer.h"

#include <iostream>

#include "glm/gtc/matrix_transform.hpp"
#include "shaders.h"

Renderer::Renderer() {}

bool
Renderer::init()
{
    glBlendEquation(GL_FUNC_ADD);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    scatter_renderer.init();
    graph_renderer.init();

    return true;
}

void
Renderer::render(State &state, View &view)
{
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    scatter_renderer.draw(view, state.scatter, state.colors);
    graph_renderer.draw(view, state.landmarks);
}

bool Renderer::is_vert_pressed(const View &view,
                        glm::vec2 mouse,
                        size_t &vert_ind) const{
    graph_renderer.is_vert_pressed(view, mouse, vert_ind);
}
