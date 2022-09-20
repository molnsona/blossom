#ifndef RENDERER_H
#define RENDERER_H

#include "graph_renderer.h"
#include "scatter_renderer.h"
#include "state.h"

class Renderer
{
public:
    Renderer();
    bool init();
    void render(State &state);

private:
    ScatterRenderer scatter_renderer;
    GraphRenderer graph_renderer;
};

#endif // RENDERER_H