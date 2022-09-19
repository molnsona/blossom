#ifndef RENDERER_H
#define RENDERER_H

#include "shader.h"

class Renderer {
public:
    Renderer();
    bool init();
    void render();
private:
    Shader scatter_shader;
    unsigned int VAO;
};

#endif // RENDERER_H