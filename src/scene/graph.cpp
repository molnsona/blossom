#include <Magnum/Math/Color.h>
#include <Magnum/MeshTools/Compile.h>
#include <Magnum/Primitives/Circle.h>
#include <Magnum/Primitives/Line.h>
#include <Magnum/Trade/MeshData.h>
#include <Magnum/Math/Vector.h>

#include "graph.h"

void PickableObject::draw(const Matrix4& transformationMatrix, SceneGraph::Camera3D& camera) {    
    _shader.setTransformationProjectionMatrix(camera.projectionMatrix() * transformationMatrix)
        .setColor(_color)
        .draw(_mesh);
}

Graph::Graph(State* p_state, Object3D& parent, SceneGraph::DrawableGroup3D& drawables)
{
    std::size_t vtx_count = 49;
    for(std::size_t i = 0; i < vtx_count; ++i)
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
            _vertices.emplace_back(new PickableObject{index + 1, _vert_color, _circle_meshes[index], parent, drawables});
            (*_vertices[index]).translate({x_plot, y_plot, PLOT_WIDTH-1500})//.rotateX(-90.0_degf)        
                .scale(Vector3{10.0f, 10.0f, 1.0f});                        
            ++index;
            p_state->vtx_pos.emplace_back(Vector2i{x_plot, y_plot});
        }
        
    }

    // Horizontal edges
    for (size_t i = 0; i < 42; i++)
    {
        _hor_line_meshes.emplace_back(MeshTools::compile(Primitives::line2D({0.0f,0.0f}, {1.0f,0.0f})));   
    }
    
    std::size_t edges_ind = 0;    
    for (int y = 3; y > -4; --y)
    {
        int y_plot = y * 100;
        for (int x = -3; x < 3; ++x)
        {
            int x_plot = x;
            _edges.emplace_back(new PickableObject{index + 1, _edge_color, _hor_line_meshes[edges_ind], parent, drawables});
            (*_edges[edges_ind]).translate({x_plot, y_plot, PLOT_WIDTH-1500})//.rotateX(-90.0_degf)        
                .scale(Vector3{100.0f, 1.0f, 1.0f});                    
            ++index;    
            ++edges_ind;
            p_state->edges.emplace_back(Vector2i(edges_ind, edges_ind + 1));
        }
        
    }
    
    // _line_meshes.emplace_back(MeshTools::compile(Primitives::line2D({0.0f,0.0f}, {1.0f,0.0f})));   
    // _edges.emplace_back(new PickableObject{index + 1, _edge_color, _line_meshes[0], parent, drawables});
    // (*_edges[0]).translate({-3.0f, 300.0f, PLOT_WIDTH-1500})//.rotateX(-90.0_degf)        
    //     .scale(Vector3{100.0f, 1.0f, 1.0f});                        
    
    // Vertical edges
    for (size_t i = 0; i < 42; i++)
    {
        _ver_line_meshes.emplace_back(MeshTools::compile(Primitives::line2D({0.0f,0.0f}, {0.0f,-1.0f})));   
    }
    
    std::size_t prev_edges_ind = edges_ind;
    edges_ind = 0;    
    for (int y = 3; y > -3; --y)
    {
        int y_plot = y;
        for (int x = -3; x < 4; ++x)
        {
            int x_plot = x * 100;
            _edges.emplace_back(new PickableObject{index + 1, _edge_color, _ver_line_meshes[edges_ind], parent, drawables});
            (*_edges[prev_edges_ind]).translate({x_plot, y_plot, PLOT_WIDTH-1500})//.rotateX(-90.0_degf)        
                .scale(Vector3{1.0f, 100.0f, 1.0f});                    
            ++index;    
            ++edges_ind;
            ++prev_edges_ind;
            p_state->edges.emplace_back(Vector2i(edges_ind, edges_ind + 6));
        }
        
    }

    // _edges.emplace_back(new PickableObject{index + 1, _edge_color, _ver_line_meshes[0], parent, drawables});
    // (*_edges[prev_edges_ind]).translate({100, 2, PLOT_WIDTH-1500})//.rotateX(-90.0_degf)        
    //     .scale(Vector3{1.0f, 100.0f, 1.0f});                    

}

void Graph::draw_event(State* p_state)
{
}