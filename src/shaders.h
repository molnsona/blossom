/* This file is part of BlosSOM.
 *
 * Copyright (C) 2021 Sona Molnarova
 *
 * BlosSOM is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option)
 * any later version.
 *
 * BlosSOM is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * BlosSOM. If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef SHADERS_H
#define SHADERS_H

#include <string>

const std::string scatter_vs =
  "#version 400 core\n"
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

const std::string scatter_fs = "#version 400 core\n"
                               "out vec4 FragColor;\n"
                               "in vec4 outColor;\n"
                               "void main()\n"
                               "{\n"
                               "   gl_FragColor = outColor;\n"
                               "}\0";

const std::string graph_v_vs =
  "#version 330 core\n"
  "layout (location = 0) in vec2 aPos;\n"
  "layout (location = 1) in vec4 aCol;\n"
  "uniform mat4 model;\n"
  "uniform mat4 view;\n"
  "uniform mat4 proj;\n"
  "out vec4 outColor;\n"
  "void main()\n"
  "{\n"
  "   gl_Position = proj * view * model * vec4(aPos, 0.1, 1.0);\n"
  "   outColor = aCol;\n"
  "}\0";
const std::string graph_v_fs = "#version 330 core\n"
                               "out vec4 FragColor;\n"
                               "in vec4 outColor;\n"
                               "void main()\n"
                               "{\n"
                               "   FragColor = outColor;\n"
                               "}\0";

const std::string graph_v_outline_vs =
  "#version 330 core\n"
  "layout (location = 0) in vec2 aPos;\n"
  "uniform mat4 model;\n"
  "uniform mat4 view;\n"
  "uniform mat4 proj;\n"
  "void main()\n"
  "{\n"
  "   gl_Position = proj * view * model * vec4(aPos, 0.1, 1.0);\n"
  "}\0";
const std::string graph_v_outline_fs = "#version 330 core\n"
                               "out vec4 FragColor;\n"
                               "in vec4 outColor;\n"
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

const std::string graph_r_vs =
  "#version 330 core\n"
  "layout (location = 0) in vec2 aPos;\n"
  "uniform mat4 model;\n"
  "uniform mat4 view;\n"
  "uniform mat4 proj;\n"
  "void main()\n"
  "{\n"
  "   gl_Position = proj * view * model * vec4(aPos, 0.2, 1.0);\n"
  "}\0";
const std::string graph_r_fs = "#version 330 core\n"
                               "out vec4 FragColor;\n"
                               "void main()\n"
                               "{\n"
                               "   FragColor = vec4(0.0, 0.0, 0.0, 0.3);\n"
                               "}\0";

#endif // SHADERS_H
