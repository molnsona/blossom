#ifndef SHADER_H
#define SHADER_H

#include <glad/gl.h>
#include <glm/glm.hpp>
#include <string>

class Shader
{
public:
    // the program ID
    unsigned int ID;

    Shader();
    // constructor reads and builds the shader
    void build(const std::string &vertexPath, const std::string &fragmentPath);
    // use/activate the shader
    void use();
    // utility uniform functions
    void setBool(const std::string &name, bool value) const;
    void setInt(const std::string &name, int value) const;
    void setFloat(const std::string &name, float value) const;
    void setMat4(const std::string &name, glm::mat4 value) const;
};

#endif // SHADER_H