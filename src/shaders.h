#ifndef SHADERS_H
#define SHADERS_H

#include <string>

const std::string scatter_vs =
  "#version 330 core\n"
  "layout (location = 0) in vec2 aPos;\n"
  "layout (location = 1) in vec4 aCol;\n"
  "uniform mat4 model;\n"
  "uniform mat4 view;\n"
  "uniform mat4 proj;\n"
  "out vec4 outColor;\n"
  "void main()\n"
  "{\n"
  "   gl_Position = proj * view * model * vec4(aPos, 0.0, 1.0);\n"
  "   outColor = aCol;\n"
  "}\0";
const std::string scatter_fs = "#version 330 core\n"
                               "out vec4 FragColor;\n"
                               "in vec4 outColor;\n"
                               "void main()\n"
                               "{\n"
                               "   FragColor = outColor;\n"
                               "}\0";

const std::string graph_v_vs =
  "#version 330 core\n"
  "layout (location = 0) in vec2 aPos;\n"
  "uniform mat4 model;\n"
  "uniform mat4 view;\n"
  "uniform mat4 proj;\n"
  "void main()\n"
  "{\n"
  "   gl_PointSize = 10;\n"
  "   gl_Position = proj * view * model * vec4(aPos, 0.1, 1.0);\n"
  "}\0";
const std::string graph_v_fs = "#version 330 core\n"
                               "out vec4 FragColor;\n"
                               "void main()\n"
                               "{\n"
                               "   FragColor = vec4(0.4, 0.4, 0.4, 0.6);\n"
                               "}\0";

const std::string graph_e_vs =
  "#version 330 core\n"
  "layout (location = 0) in vec2 aPos;\n"
  "uniform mat4 model;\n"
  "uniform mat4 view;\n"
  "uniform mat4 proj;\n"
  "void main()\n"
  "{\n"  
  "   gl_Position = proj * view * model * vec4(aPos, 0.1, 1.0);\n"
  "}\0";
const std::string graph_e_fs = "#version 330 core\n"
                               "out vec4 FragColor;\n"
                               "void main()\n"
                               "{\n"
                               "   FragColor = vec4(1.0, 0.0, 0.0, 0.6);\n"
                               "}\0";                               

#endif // SHADERS_H