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

#ifndef SHADER_H
#define SHADER_H

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <string>

/**
 * @brief Abstracts working with shaders.
 *
 */
class Shader
{
public:
    /** Shader program id.*/
    unsigned int ID;

    Shader();

    /**
     * @brief Read and build the shader.
     *
     * @param vs Vertex shader string.
     * @param fs Fragment shader string.
     */
    void build(const std::string &vs, const std::string &fs);

    /**
     * @brief Activate built shader.
     *
     * Has to be called after Shader::build function.
     *
     */
    void use();

    /**
     * @brief Bind the bool variable to the shader.
     *
     * @param name Name of the variable in the shader.
     * @param value The set value of the variable.
     */
    void set_bool(const std::string &name, bool value) const;
    /**
     * @brief Bind the integer variable to the shader.
     *
     * @param name Name of the variable in the shader.
     * @param value The set value of the variable.
     */
    void set_int(const std::string &name, int value) const;
    /**
     * @brief Bind the float variable to the shader.
     *
     * @param name Name of the variable in the shader.
     * @param value The set value of the variable.
     */
    void set_float(const std::string &name, float value) const;
    /**
     * @brief Bind the matrix 4x4 variable to the shader.
     *
     * @param name Name of the variable in the shader.
     * @param value The set value of the variable.
     */
    void set_mat4(const std::string &name, glm::mat4 value) const;
};

#endif // SHADER_H
