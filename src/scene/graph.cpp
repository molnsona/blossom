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
    for(std::size_t i = 0; i < 49; ++i)
    {
        _circle_meshes.emplace_back(MeshTools::compile(Primitives::circle2DSolid(32)));    
    }
    
    std::size_t index = 0;
    for (int y = 3; y > -4; --y)
    {
        int y_plot = y * 10.0f;
        for (int x = -3; x < 4; ++x)
        {
            int x_plot = x * 10.0f;
            _vertices.emplace_back(new PickableObject{index + 1, _circle_meshes[index], parent, drawables});
            (*_vertices[index]).translate({x_plot, y_plot, PLOT_WIDTH-1500})//.rotateX(-90.0_degf)        
                .scale(Vector3{10.0f, 10.0f, 1.0f});

            ++index;
        }
        
    }
    

}

void Graph::draw_event(State* p_state)
{
}