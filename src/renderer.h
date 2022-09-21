#ifndef RENDERER_H
#define RENDERER_H

#include "graph_renderer.h"
#include "scatter_renderer.h"
#include "state.h"
#include "view.h"

#include "shader.h"

class Renderer
{
public:
    Renderer();
    bool init();
    void render(State &state, View &view);

private:
    ScatterRenderer scatter_renderer;
    GraphRenderer graph_renderer;

    Shader ex_shader;
};

#endif // RENDERER_H