
#include <Magnum/MeshTools/Compile.h>
#include <Magnum/Primitives/Square.h>
#include <Magnum/GL/PixelFormat.h>
#include <Magnum/ImageView.h>
#include <Magnum/Trade/MeshData.h>

#include "../utils.hpp"

#include "canvas.h"

void PickableObject::draw(const Matrix4& transformationMatrix, SceneGraph::Camera3D& camera) {
    if(_changed) {
        std::vector<unsigned char> pixels(BYTES_PER_PIXEL * PLOT_WIDTH * PLOT_HEIGHT, 255);
        fill_pixels(pixels, _cell_cnt, _mean, _std_dev);
    // pixels = std::vector<unsigned char>(BYTES_PER_PIXEL * PLOT_WIDTH * PLOT_HEIGHT, 63);

        ImageView2D image{ GL::PixelFormat::RGBA, GL::PixelType::UnsignedByte, { PLOT_WIDTH, PLOT_HEIGHT },
            { pixels.data(), std::size_t(BYTES_PER_PIXEL * PLOT_WIDTH * PLOT_HEIGHT) } };

        //GL::Texture2D _texture;
        _texture.setWrapping(GL::SamplerWrapping::ClampToEdge)
            .setMagnificationFilter(GL::SamplerFilter::Linear)
            .setMinificationFilter(GL::SamplerFilter::Linear)
            .setStorage(1, GL::TextureFormat::RGBA8, image.size())
            .setSubImage(0, {}, image); 
    }
    
    _shader.setTransformationProjectionMatrix(camera.projectionMatrix() * transformationMatrix)
        .bindTexture(_texture)
        .draw(_mesh);
}

Canvas::Canvas(Object3D& parent, SceneGraph::DrawableGroup3D& drawables)
{
    _canvas_mesh = MeshTools::compile(Primitives::squareSolid(Primitives::SquareFlag::TextureCoordinates));    

    _canvas = new PickableObject{3, _textured_shader, _canvas_mesh, parent, drawables};
    (*_canvas).rotateX(-90.0_degf)        
                .scale(Vector3{PLOT_WIDTH / 2, 0.0f, PLOT_HEIGHT / 2});
}

void Canvas::draw_event(State* p_state)
{
    (*_canvas).setConfig(p_state->cell_cnt, p_state->mean, p_state->std_dev);
}