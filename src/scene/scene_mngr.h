#ifndef SCENE_MNGR_H
#define SCENE_MNGR_H

#include <Magnum/Platform/Sdl2Application.h>
#include <Magnum/SceneGraph/Camera.h>
#include <Magnum/SceneGraph/Scene.h>
#include <Magnum/SceneGraph/Drawable.h>
#include <Magnum/SceneGraph/MatrixTransformation3D.h>

#include <memory>

#include "canvas.h"
#include "graph.h"
#include "../app/state.h"

using namespace Magnum;
using namespace Math::Literals;

typedef SceneGraph::Object<SceneGraph::MatrixTransformation3D> Object3D;
typedef SceneGraph::Scene<SceneGraph::MatrixTransformation3D> Scene3D;

class SceneMngr 
{
public:
    SceneMngr() = delete;
    SceneMngr(State* p_state);

    void update(State* p_state);
    
    void draw_event(State* p_state);
    void viewport_event(Platform::Application::ViewportEvent& event);

    Matrix4 getInvViewProjMat() { return (_camera->projectionMatrix()*_camera->cameraMatrix()).inverted();}
private:
    Scene3D _scene;
    SceneGraph::DrawableGroup3D _drawables;

    Canvas _canvas;
    Graph _graph;

    SceneGraph::Camera3D* _camera;
    Object3D* _cameraObject;
};

#endif // #ifndef SCENE_MNGR_H