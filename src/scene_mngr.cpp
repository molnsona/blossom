#include <Magnum/GL/DefaultFramebuffer.h>

#include "imgui_config.h"

#include "scene_mngr.h"

SceneMngr::SceneMngr(State *p_state)
  : _canvas(_scene, _drawables)
  , _graph(p_state, _scene, _drawables)
{
    // _canvas = std::make_unique<Canvas>(_scene, _drawables);
    // _graph = std::make_unique<Graph>(p_state, _scene, _drawables);

    // /* Configure camera */
    // _cameraObject = new Object3D{&scene};
    // (*_cameraObject)
    //     .translate(Vector3::zAxis(40.0f))
    //     .rotateX(-90.0_degf);
    // p_state->camera_trans.z() = 40.0f;

    // _camera = new SceneGraph::Camera3D{*_cameraObject};
    // _camera->setAspectRatioPolicy(SceneGraph::AspectRatioPolicy::Extend)
    //     .setProjectionMatrix(Matrix4::orthographicProjection(p_state->zoom_depth,
    //     0.001f, 100.0f))
    //     .setViewport(GL::defaultFramebuffer.viewport().size());

    /* Configure camera */
    _cameraObject = new Object3D{ &_scene };
    _cameraObject->translate(Vector3::zAxis(PLOT_WIDTH + 200));
    //_cameraObject->translate(Vector3::zAxis(201.0f));
    _camera = new SceneGraph::Camera3D{ *_cameraObject };
    _camera->setAspectRatioPolicy(SceneGraph::AspectRatioPolicy::Extend)
      .setProjectionMatrix(Matrix4::orthographicProjection(
        { PLOT_WIDTH, PLOT_HEIGHT }, 0.001f, 3000.0f))
      .setViewport(GL::defaultFramebuffer.viewport().size());
}

void
SceneMngr::update(State *p_state)
{
    _graph.update(p_state);
}

void
SceneMngr::draw_event(State *p_state)
{
    _canvas.draw_event(p_state);
    _graph.draw_event(p_state, _scene, _drawables);

    _camera->draw(_drawables);
}

void
SceneMngr::viewport_event(Platform::Application::ViewportEvent &event)
{
    _camera->setViewport(event.windowSize());
}
