#include <Magnum/GL/DefaultFramebuffer.h>

#include "scene_mngr.h"

SceneMngr::SceneMngr(State* p_state, Scene3D& scene, SceneGraph::DrawableGroup3D& drawables)
{
    _p_canvas = std::make_unique<Canvas>(scene, drawables);
    _p_graph = std::make_unique<Graph>(p_state, scene, drawables);
    
    // /* Configure camera */
    // _cameraObject = new Object3D{&scene};
    // (*_cameraObject)
    //     .translate(Vector3::zAxis(40.0f))
    //     .rotateX(-90.0_degf);
    // p_state->camera_trans.z() = 40.0f;

    // _camera = new SceneGraph::Camera3D{*_cameraObject};
    // _camera->setAspectRatioPolicy(SceneGraph::AspectRatioPolicy::Extend)
    //     .setProjectionMatrix(Matrix4::orthographicProjection(p_state->zoom_depth, 0.001f, 100.0f))
    //     .setViewport(GL::defaultFramebuffer.viewport().size());

     /* Configure camera */
    _cameraObject = new Object3D{&scene};
    _cameraObject->translate(Vector3::zAxis(PLOT_WIDTH+200));
    //_cameraObject->translate(Vector3::zAxis(201.0f));
    _camera = new SceneGraph::Camera3D{*_cameraObject};
    _camera->setAspectRatioPolicy(SceneGraph::AspectRatioPolicy::Extend)
        .setProjectionMatrix(Matrix4::perspectiveProjection(35.0_degf, 4.0f/3.0f, 0.001f, 3000.0f))
        .setViewport(GL::defaultFramebuffer.viewport().size());
}

void SceneMngr::draw_event(State* p_state, SceneGraph::DrawableGroup3D& drawables)
{
    _p_canvas->draw_event(p_state);
    _p_graph->draw_event(p_state);

    _camera->draw(drawables);
}

void SceneMngr::viewport_event(Platform::Application::ViewportEvent& event)
{
    _camera->setViewport(event.windowSize());
}