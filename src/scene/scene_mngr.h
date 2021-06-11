#ifndef SCENE_MNGR_H
#define SCENE_MNGR_H

#include <Magnum/Platform/Sdl2Application.h>
#include <Magnum/SceneGraph/Camera.h>
#include <Magnum/SceneGraph/Scene.h>
#include <Magnum/SceneGraph/Drawable.h>

#include <memory>

#include "canvas.h"
#include "graph.h"
#include "../app/state.h"

typedef SceneGraph::Object<SceneGraph::MatrixTransformation3D> Object3D;
typedef SceneGraph::Scene<SceneGraph::MatrixTransformation3D> Scene3D;

class SceneMngr 
{
public:
    SceneMngr(State* p_state, Scene3D& scene, SceneGraph::DrawableGroup3D& drawables);
    
    void draw_event(State* p_state, SceneGraph::DrawableGroup3D& drawables);
    void viewport_event(Platform::Application::ViewportEvent& event);
private:
    std::unique_ptr<Canvas> _p_canvas;
    std::unique_ptr<Graph> _p_graph;

    SceneGraph::Camera3D* _camera;
    Object3D* _cameraObject;
};

#endif // #ifndef SCENE_MNGR_H