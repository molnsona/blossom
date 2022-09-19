
#include <iostream>

#include "shaders.h"
#include "shader.h"
#include "wrapper_glad.h"
#include "wrapper_glfw.h"
#include "wrapper_imgui.h"

int main()
{
    GlfwWrapper glfw;
    GladWrapper glad;
    ImGuiWrapper imgui;

    if(!glfw.init("BlosSOM"))
    {
        std::cout << "GLFW initialization failed." << std::endl;
        return -1;
    }
    
    if(!glad.init()) 
    {
        std::cout << "GLAD initialization failed." << std::endl;
        glfw.destroy();
        return -1;
    }

    if(!imgui.init(glfw.window))
    {
        std::cout << "Dear ImGui initialization failed." << std::endl;
        glfw.destroy();
        return -1;
    }

    // OPENGL code
    float vertices[] = {
        -0.5f, -0.5f, 0.0f,
        0.5f, -0.5f, 0.0f,
        0.0f,  0.5f, 0.0f
    }; 

    unsigned int VAO;
    glGenVertexArrays(1, &VAO);  
    glBindVertexArray(VAO);

    unsigned int VBO;
    glGenBuffers(1, &VBO);  
    glBindBuffer(GL_ARRAY_BUFFER, VBO);  
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);  
    
    Shader scatter_shader(scatter_vs, scatter_fs);

    while (!glfw.window_should_close())
    {            
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        scatter_shader.use();
        glBindVertexArray(VAO);
        glDrawArrays(GL_POINTS, 0, 3);
        
        imgui.compose_frame();
        imgui.render();

        glfw.end_frame();
    }

    imgui.destroy();
    glfw.destroy();
    return 0;
}