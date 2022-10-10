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
Renderer::render(State &state, View &view)
{
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    scatter_renderer.draw(view, state.scatter, state.colors);
    graph_renderer.draw(view, state.landmarks);
}
