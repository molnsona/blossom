#include <Magnum/Math/Color.h>
#include <Magnum/Math/Vector.h>
#include <Magnum/MeshTools/Compile.h>
#include <Magnum/Primitives/Circle.h>
#include <Magnum/Primitives/Line.h>
#include <Magnum/Trade/MeshData.h>

#include <vector>

#include "graph.h"

Graph::Graph(State *p_state,
             Object3D &parent,
             SceneGraph::DrawableGroup3D &drawables)
{
    for (std::size_t i = 0; i < vtx_count; ++i) {
        _circle_meshes.emplace_back(
          MeshTools::compile(Primitives::circle2DSolid(32)));
    }

    std::size_t index = 0;
    for (int y = 3; y > -4; --y) {
        int y_plot = y * 100.0f;
        for (int x = -3; x < 4; ++x) {
            int x_plot = x * 100.0f;
            _vertices.emplace_back(
              std::make_unique<PickableObject>(index + 1,
                                               _vert_color,
                                               _circle_meshes[index],
                                               parent,
                                               drawables));
            (*_vertices[index])
              .scale(Vector3{ 10.0f, 10.0f, 1.0f })
              .translate({ x_plot,
                           y_plot,
                           1.0f /*PLOT_WIDTH-1500*/ }); //.rotateX(-90.0_degf)
            ++index;
            p_state->vtx_pos.emplace_back(Vector2i{ x_plot, y_plot });
        }
    }

    // Horizontal edges
    for (size_t i = 0; i < 42; i++) {
        //_hor_line_meshes.emplace_back(MeshTools::compile(Primitives::line2D({0.0f,0.0f},
        //{1.0f,0.0f})));
        _edge_meshes.emplace_back(MeshTools::compile(
          Primitives::line2D({ 0.0f, 0.0f }, { 100.0f, 0.0f })));
    }

    // Vertical edges
    for (size_t i = 0; i < 42; i++) {
        //_ver_line_meshes.emplace_back(MeshTools::compile(Primitives::line2D({0.0f,0.0f},
        //{0.0f,-1.0f})));
        _edge_meshes.emplace_back(MeshTools::compile(
          Primitives::line2D({ 0.0f, 0.0f }, { 0.0f, -100.0f })));
    }

    std::size_t edges_ind = 0;
    std::size_t ind = 0;
    std::size_t vert_ind = 0;
    for (int y = 3; y > -4; --y) {
        int y_plot = y * 100;
        for (int x = -3; x < 3; ++x) {
            int x_plot = x * 100;
            _edges.emplace_back(
              std::make_unique<PickableObject>(index + 1,
                                               _edge_color,
                                               _edge_meshes[edges_ind],
                                               parent,
                                               drawables));
            (*_edges[edges_ind])
              .translate({ x_plot,
                           y_plot,
                           0.5f /*PLOT_WIDTH-1500*/ }); //.rotateX(-90.0_degf)
            p_state->edges.emplace_back(Vector2i(ind, ind + 1));
            p_state->edges.emplace_back(Vector2i(vert_ind, vert_ind + 7));
            ++index;
            ++edges_ind;
            ++ind;
            ++vert_ind;
        }
        ++ind;
    }

    // _line_meshes.emplace_back(MeshTools::compile(Primitives::line2D({0.0f,0.0f},
    // {1.0f,0.0f}))); _edges.emplace_back(new PickableObject{index + 1,
    // _edge_color, _line_meshes[0], parent, drawables});
    // (*_edges[0]).translate({-3.0f, 300.0f,
    // PLOT_WIDTH-1500})//.rotateX(-90.0_degf)
    //     .scale(Vector3{100.0f, 1.0f, 1.0f});

    // std::size_t prev_edges_ind = edges_ind;
    // edges_ind = 0;
    ind = 0;
    for (int y = 3; y > -3; --y) {
        int y_plot = y * 100;
        for (int x = -3; x < 4; ++x) {
            int x_plot = x * 100;
            _edges.emplace_back(
              std::make_unique<PickableObject>(index + 1,
                                               _edge_color,
                                               _edge_meshes[edges_ind],
                                               parent,
                                               drawables));
            (*_edges[edges_ind])
              .translate({ x_plot,
                           y_plot,
                           0.5f /*PLOT_WIDTH-1500*/ }); //.rotateX(-90.0_degf)
            // p_state->edges.emplace_back(Vector2i(ind, ind + 7));
            ++index;
            ++edges_ind;
            ++ind;
            //++prev_edges_ind;
        }
        //++ind;
    }

    // _edges.emplace_back(new PickableObject{index + 1, _edge_color,
    // _ver_line_meshes[0], parent, drawables});
    // (*_edges[prev_edges_ind]).translate({100, 2,
    // PLOT_WIDTH-1500})//.rotateX(-90.0_degf)
    //     .scale(Vector3{1.0f, 100.0f, 1.0f});

    // p_state->_vertices = _vertices;

    edges_size = 84;
    p_state->lengths = std::vector(edges_size, 100);
    // meshes.resize(edges_size);
    //_edges.resize(edges_size);
}

void
Graph::update(State *p_state)
{
    if (p_state->vtx_selected || p_state->mouse_pressed) {
        p_state->time = 0;
        // Move vertices
        for (auto &&vtx : _vertices)
            vtx->setSelected(false);
        UnsignedInt id = p_state->vtx_ind;
        if (id > 0 && id < vtx_count + 1) {
            _vertices[id - 1]->setSelected(true);
            //_vertices[id - 1]->translate({p_state->mouse_pos.x()+ 10,
            // p_state->mouse_pos.y() + 10, 0});
            int x = 0, y = 0;
            // x = p_state->mouse_pos.x() > 0 ? 10 : -10;
            // y = p_state->mouse_pos.y() > 0 ? 10 : -10;
            p_state->vtx_pos[id - 1] = p_state->mouse_pos; /*+ Vector2{x,y};*/

            // (*_vertices[id - 1]).resetTransformation();
            // (*_vertices[id - 1]).scale(Vector3{10.0f, 10.0f, 1.0f})
            //     .translate({p_state->vtx_pos[id - 1].x(), p_state->vtx_pos[id
            //     - 1].y(), 1.0f});
        }
        p_state->vtx_selected = false;
    }
}

void
Graph::draw_event(State *p_state,
                  Object3D &parent,
                  SceneGraph::DrawableGroup3D &drawables)
{
    // Draw vertices
    for (size_t i = 0; i < p_state->vtx_pos.size(); ++i) {
        //_vertices[i] = std::make_unique<PickableObject>(i + 1, _vert_color,
        //_circle_meshes[i], parent, drawables);
        (*_vertices[i]).resetTransformation();
        (*_vertices[i])
          .scale(Vector3{ 10.0f, 10.0f, 1.0f })
          .translate(
            { p_state->vtx_pos[i].x(), p_state->vtx_pos[i].y(), 1.0f });
    }

    // Draw edges
    //_edges.clear();
    // meshes.clear();
    // std::vector<GL::Mesh> meshes;
    std::size_t index = vtx_count + 1;
    std::size_t mesh_ind = 0;
    for (auto &&edge : p_state->edges) {
        Vector2 v1_pos, v2_pos;
        v1_pos = Vector2(p_state->vtx_pos[edge.x()]);
        v2_pos = Vector2(p_state->vtx_pos[edge.y()]);
        _edge_meshes[mesh_ind] =
          MeshTools::compile(Primitives::line2D(v1_pos, v2_pos));
        // _edges[mesh_ind]->translate({v1_pos.x(), v1_pos.y(), 0.0f});

        _edges[mesh_ind]->resetTransformation();
        (*_edges[mesh_ind]).translate({ 0.0f, 0.0f, 0.5f });
        // _edges[mesh_ind] = std::make_unique<PickableObject>(index + 1,
        // _edge_color, _edge_meshes[mesh_ind], parent, drawables);
        // (*_edges[mesh_ind]).translate({0.0f, 0.0f,
        // 0.5f/*PLOT_WIDTH-1500*/});//.rotateX(-90.0_degf)
        // //     .scale(Vector3{100.0f, 1.0f, 1.0f});

        ++mesh_ind;
        ++index;

        // break;
    }

    // mesh_ind = 0;
    // for (auto&& edge: p_state->edges)
    // {
    //     _edges[mesh_ind] = new PickableObject{index, _edge_color,
    //     _edge_meshes[mesh_ind], parent, drawables};
    //     ++index;
    //     ++mesh_ind;
    // }
}