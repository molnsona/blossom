#include "renderer.h"

#include <iostream>

#include "glad/gl.h"
#include "shaders.h"

Renderer::Renderer()
{}

bool Renderer::init()
{
    float vertices[] = {
        -0.5f, -0.5f, 0.0f,
        0.5f, -0.5f, 0.0f,
        0.0f,  0.5f, 0.0f
    }; 

    glGenVertexArrays(1, &VAO);  
    glBindVertexArray(VAO);
    unsigned int VBO;
    glGenBuffers(1, &VBO);  
    glBindBuffer(GL_ARRAY_BUFFER, VBO);  
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);  
    
    scatter_shader.build(scatter_vs, scatter_fs);

    return true;
}

void Renderer::render()
{
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    scatter_shader.use();
    glBindVertexArray(VAO);
    glDrawArrays(GL_POINTS, 0, 3);   
}