#ifndef RENDERER_H
#define RENDERER_H

#include "shader.h"

#include <GLFW/glfw3.h>

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