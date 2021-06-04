#include <Magnum/Math/Color.h>
#include <Magnum/MeshTools/Compile.h>
#include <Magnum/Primitives/Circle.h>
#include <Magnum/Trade/MeshData.h>

#include "graph.h"

void PickableObject::draw(const Matrix4& transformationMatrix, SceneGraph::Camera3D& camera) {    
    _shader.setTransformationProjectionMatrix(camera.projectionMatrix() * transformationMatrix)
        .setColor(0xdb4437_rgbf)
        .draw(_mesh);
}

Graph::Graph(Object3D& parent, SceneGraph::DrawableGroup3D& drawables)
{
    _circle_mesh = MeshTools::compile(Primitives::circle2DSolid(32));    

    _vertices.emplace_back(new PickableObject{1, _circle_mesh, parent, drawables});
    (*_vertices[0]).translate({0.0f, 0.0f, PLOT_WIDTH-1500})//.rotateX(-90.0_degf)        
                .scale(Vector3{10.0f, 10.0f, 1.0f});
}

void Graph::draw_event(State* p_state)
{
}