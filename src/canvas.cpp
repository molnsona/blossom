
#include <Magnum/GL/PixelFormat.h>
#include <Magnum/ImageView.h>
#include <Magnum/MeshTools/Compile.h>
#include <Magnum/Primitives/Square.h>
#include <Magnum/Trade/MeshData.h>

#include "canvas.h"
#include "utils.hpp"

void
DrawableObject::fill_texture(State *p_state)
{
    if (_changed) {

        // std::vector<unsigned char> pixels(BYTES_PER_PIXEL * PLOT_WIDTH *
        // PLOT_HEIGHT, DEFAULT_WHITE);
        std::fill(
          p_state->pixels.begin(), p_state->pixels.end(), DEFAULT_WHITE);

        fill_pixels(p_state->pixels, _cell_cnt, _mean, _std_dev);
        // pixels = std::vector<unsigned char>(BYTES_PER_PIXEL * PLOT_WIDTH *
        // PLOT_HEIGHT, 63);

        ImageView2D image{ GL::PixelFormat::RGBA,
                           GL::PixelType::UnsignedByte,
                           { PLOT_WIDTH, PLOT_HEIGHT },
                           { p_state->pixels.data(),
                             std::size_t(BYTES_PER_PIXEL * PLOT_WIDTH *
                                         PLOT_HEIGHT) } };

        // GL::Texture2D _texture;
        _texture.setWrapping(GL::SamplerWrapping::ClampToEdge)
          .setMagnificationFilter(GL::SamplerFilter::Linear)
          .setMinificationFilter(GL::SamplerFilter::Linear)
          .setStorage(1, GL::TextureFormat::RGBA8, image.size())
          .setSubImage(0, {}, image);
    }
}

void
DrawableObject::draw(const Matrix4 &transformationMatrix,
                     SceneGraph::Camera3D &camera)
{
    _shader
      .setTransformationProjectionMatrix(camera.projectionMatrix() *
                                         transformationMatrix)
      .bindTexture(_texture)
      .draw(_mesh);
}

Canvas::Canvas(Object3D &parent, SceneGraph::DrawableGroup3D &drawables)
{
    _canvas_mesh = MeshTools::compile(
      Primitives::squareSolid(Primitives::SquareFlag::TextureCoordinates));

    _canvas = new DrawableObject{
        0, _textured_shader, _canvas_mesh, parent, drawables
    };
    (*_canvas) //.translate({0.0f, 0.0f, -2000.0f})
      .scale(Vector3{
        PLOT_WIDTH / 2.0f,
        PLOT_HEIGHT / 2.0f,
        0.0f }); //.rotateX(-90.0_degf)
                 //.scale(Vector3{PLOT_WIDTH / 2.0f, PLOT_HEIGHT / 2.0f, 0.0f});
}

void
Canvas::draw_event(State *p_state)
{
    _canvas->setConfig(p_state->cell_cnt, p_state->mean, p_state->std_dev);
    _canvas->fill_texture(p_state);
}
