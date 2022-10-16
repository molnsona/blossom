#ifndef RENDERER_H
#define RENDERER_H

#include "graph_renderer.h"
#include "scatter_renderer.h"
#include "state.h"
#include "view.h"

class Renderer
{
public:
    Renderer();
    bool init();
    void update(State &state, View &view, const CallbackValues &callbacks);


private:
    ScatterRenderer scatter_renderer;
    GraphRenderer graph_renderer;

    void render(State &state, View &view);

    void process_mouse(State &state, const View &view, const CallbackValues &callbacks);
    void process_keyboard(State &state, const View &view, const CallbackValues &callbacks);
};

#endif // RENDERER_H