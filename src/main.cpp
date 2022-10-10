


#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "renderer.h"
#include "shader.h"
#include "shaders.h"
#include "state.h"
#include "timer.h"
#include "view.h"
#include "wrapper_glad.h"
#include "wrapper_glfw.h"
#include "wrapper_imgui.h"

#include <iostream>

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow *window);

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

// camera
glm::vec3 cameraPos   = glm::vec3(0.0f, 0.0f,  10.0f);
glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
glm::vec3 cameraUp    = glm::vec3(0.0f, 1.0f,  0.0f);

float fov   =  45.0f;

// timing
float deltaTime = 0.0f;	// time between current frame and last frame
float lastFrame = 0.0f;

int main()
{
    GlfwWrapper glfw;
    GladWrapper glad;
    ImGuiWrapper imgui;
    Renderer renderer;
    Timer timer;
    View view;
    State state;

    if (!glfw.init("BlosSOM")) {
        std::cout << "GLFW initialization failed." << std::endl;
        return -1;
    }

    // glfw: initialize and configure
    // ------------------------------
//     glfwInit();
//     glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
//     glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
//     glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

// #ifdef __APPLE__
//     glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
// #endif

//     // glfw window creation
//     // --------------------
//     GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
//     if (window == NULL)
//     {
//         std::cout << "Failed to create GLFW window" << std::endl;
//         glfwTerminate();
//         return -1;
//     }
//     glfwMakeContextCurrent(window);
//     glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
//     glfwSetScrollCallback(window, scroll_callback);

    if (!glad.init()) {
        std::cout << "GLAD initialization failed." << std::endl;
        glfw.destroy();
        return -1;
    }

    // // glad: load all OpenGL function pointers
    // // ---------------------------------------
    // if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    // {
    //     std::cout << "Failed to initialize GLAD" << std::endl;
    //     return -1;
    // }

    if (!imgui.init(glfw.window)) {
        std::cout << "Dear ImGui initialization failed." << std::endl;
        glfw.destroy();
        return -1;
    }

    if (!renderer.init()) {
        std::cout << "Renderer initialization failed." << std::endl;
        imgui.destroy();
        glfw.destroy();
        return -1;
    }
    //renderer.init();

    // configure global opengl state
    // -----------------------------
    //glEnable(GL_DEPTH_TEST);

    // build and compile our shader zprogram
    // ------------------------------------
    // Shader pointShader;
    // pointShader.build(graph_vs, graph_fs);

    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------ 
    // float point[] = {0.0f, 0.0f, 0.0f,
    //                 1.0f, 0.0f, 0.0f,
    //                 0.0f, 1.0f, 0.0f,
    //                 1.0f, 1.0f, 0.0f};
    // unsigned int VBO2, VAO2;
    // glGenVertexArrays(1, &VAO2);
    // glGenBuffers(1, &VBO2);

    // glBindVertexArray(VAO2);

    // glBindBuffer(GL_ARRAY_BUFFER, VBO2);
    // glBufferData(GL_ARRAY_BUFFER, sizeof(point), point, GL_STATIC_DRAW);

    // // position attribute
    // glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    // glEnableVertexAttribArray(0);

    // render loop
    // -----------
    while (!glfw.window_should_close())
    {
        // // per-frame time logic
        // // --------------------
        // float currentFrame = static_cast<float>(glfwGetTime());
        // deltaTime = currentFrame - lastFrame;
        // lastFrame = currentFrame;

        // input
        // -----
        //processInput(glfw.window);

        // render
        // ------
        // glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        // glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 

        // pass projection matrix to shader (as projection matrix rarely changes there's no need to do this per frame)
        // -----------------------------------------------------------------------------------------------------------
        // glm::mat4 projection = glm::perspective(glm::radians(fov), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
        
        // // camera/view transformation
        // glm::mat4 view_mat = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
        
        // pointShader.use();
        // pointShader.setMat4("model", glm::mat4(1.0f));
        // pointShader.setMat4("view", view_mat);
        // pointShader.setMat4("proj", projection);

        // glBindVertexArray(VAO2);        
        // glEnable(GL_PROGRAM_POINT_SIZE);
        // glDrawArrays(GL_POINTS, 0, 4);
        // glDisable(GL_PROGRAM_POINT_SIZE);
        
        timer.tick();
        
        view.update(timer.frametime, glfw.callbacks);
        state.update(timer.frametime);
        
        renderer.render(state, view);
        
        // glfw.callbacks.fb_width = SCR_WIDTH;
        // glfw.callbacks.fb_height = SCR_HEIGHT;
        imgui.render(glfw.callbacks, state);
        
        glfw.end_frame();
        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        // glfwSwapBuffers(glfw.window);
        // glfwPollEvents();
    }

    // // optional: de-allocate all resources once they've outlived their purpose:
    // // ------------------------------------------------------------------------
    // glDeleteVertexArrays(1, &VAO2);
    // glDeleteBuffers(1, &VBO2);

    // // glfw: terminate, clearing all previously allocated GLFW resources.
    // // ------------------------------------------------------------------
    // glfwTerminate();

    imgui.destroy();
    glfw.destroy();
    return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window)
{
    // if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    //     glfwSetWindowShouldClose(window, true);

    float cameraSpeed = static_cast<float>(2.5 * deltaTime);
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        //cameraPos += cameraSpeed * cameraFront;
        cameraPos += cameraUp * cameraSpeed;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        //cameraPos -= cameraSpeed * cameraFront;
        cameraPos -= cameraUp * cameraSpeed;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    // float cameraSpeed = static_cast<float>(2.5 * deltaTime);

    // if(yoffset > 0)
    //     cameraPos += cameraSpeed * cameraFront;
    // else if(yoffset < 0)
    //     cameraPos -= cameraSpeed * cameraFront;

    fov -= (float)yoffset;
    if (fov < 1.0f)
        fov = 1.0f;
    if (fov > 45.0f)
        fov = 45.0f;
}

// #include <iostream>

// #include "renderer.h"
// #include "state.h"
// #include "timer.h"
// #include "view.h"
// #include "wrapper_glad.h"
// #include "wrapper_glfw.h"
// #include "wrapper_imgui.h"

// int
// main()
// {
//     GlfwWrapper glfw;
//     GladWrapper glad;
//     ImGuiWrapper imgui;
//     Renderer renderer;

//     Timer timer;
//     View view;
//     State state;

//     if (!glfw.init("BlosSOM")) {
//         std::cout << "GLFW initialization failed." << std::endl;
//         return -1;
//     }

//     if (!glad.init()) {
//         std::cout << "GLAD initialization failed." << std::endl;
//         glfw.destroy();
//         return -1;
//     }

//     if (!imgui.init(glfw.window)) {
//         std::cout << "Dear ImGui initialization failed." << std::endl;
//         glfw.destroy();
//         return -1;
//     }

//     if (!renderer.init()) {
//         std::cout << "Renderer initialization failed." << std::endl;
//         imgui.destroy();
//         glfw.destroy();
//         return -1;
//     }

//     while (!glfw.window_should_close()) {
//         timer.tick();

//         view.update(timer.frametime);
//         state.update(timer.frametime);

//         renderer.render(state, view);

//         imgui.render(glfw.callbacks, state);

//         glfw.end_frame();
//     }

//     imgui.destroy();
//     glfw.destroy();
//     return 0;
// }