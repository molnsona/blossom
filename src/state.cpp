
#include <iostream>

#include "embedsom.h"
#include "fcs_parser.h"
#include "state.h"
#include "tsv_parser.h"

State::State()
  : trans(data.data, data.d, data.n)
{}

void
State::update(float actual_time)
{
    // avoid simulation explosions on long frames
    float time = std::min(actual_time, 0.05);

    if (ui.reset) {
        data = DataModel();
        trans.set_data(data.data, data.d, data.n);
        landmarks = LandmarkModel();
        scatter = ScatterModel();
        layout_data = GraphLayoutData();
        ui.reset_data();
    }

    if (ui.parser_data.parse) {
        std::string ext =
          std::filesystem::path(ui.parser_data.file_path).extension().string();
        Parser parse;

        if (ext == ".fcs") {
            ui.parser_data.reset_data();
            parse = FCSParser::parse;
        } else if (ext == ".tsv") {
            ui.parser_data.reset_data();
            parse = TSVParser::parse;
        }

        ui.reset_data();

        parse(ui.parser_data.file_path,
              1000,
              data.data,
              data.d,
              data.n,
              ui.trans_data.param_names);

        ui.trans_data.scale.clear();
        ui.trans_data.scale.resize(ui.trans_data.param_names.size());

        ui.trans_data.sliders.clear();
        ui.trans_data.sliders.resize(ui.trans_data.param_names.size());

        trans.set_data(data.data, data.d, data.n);

        landmarks.update_dim(trans.d);
        ui.parser_data.parse = false;
    }

    trans.update(ui.trans_data, data);

#if 0
    graph_layout_step(layout_data,
                      mouse,
                      landmarks.lodim_vertices,
                      landmarks.edges,
                      landmarks.edge_lengths,
                      time);
#endif

#if 0
    kmeans_landmark_step(
      kmeans_data,
      data,
      landmarks.n_landmarks(),
      landmarks.d,
      100, // TODO parametrize (now this is 100 iters per frame, there should be
           // fixed number of iters per actual elapsed time)
      0.0001,    // TODO parametrize, logarithmically between 1e-6 and ~0.5
      0,   // TODO parametrize as 0-1 multiple of ^^
      landmarks.edges,
      landmarks.hidim_vertices);
#endif

#if 1
    som_landmark_step(
      kmeans_data,
      data,
      landmarks.n_landmarks(),
      landmarks.d,
      100,
      ui.sliders_data.alpha,
      ui.sliders_data
        .sigma, // TODO this really needs to be slidable by the user
      landmarks.hidim_vertices,
      landmarks.lodim_vertices);
#endif

#if 0
    make_knn_edges(knn_data, landmarks, 3);
#endif

    if (scatter.points.size() != trans.n) {
        scatter.points.clear();
        scatter.points.resize(trans.n);
    }

#ifdef NO_CUDA
    // TODO check that data dimension matches landmark dimension and that
    // model sizes are matching (this is going to change dynamically)
    embedsom(data.n,
             landmarks.lodim_vertices.size(),
             data.d, // should be the same as landmarks.d
             2.0,
             10,
             0.2,
             data.data.data() /* <3 */,
             landmarks.hidim_vertices.data(),
             landmarks.lodim_vertices[0].data(),
             scatter.points[0].data());
#else
    // these methods should be called only once in the initialization and then
    // only when the data/paramters change
    esom_cuda.setDim(trans.d);
    esom_cuda.setK(10);
    esom_cuda.setPoints(trans.n, trans.get_data().data());
    esom_cuda.setLandmarks(landmarks.lodim_vertices.size(),
                           landmarks.hidim_vertices.data(),
                           landmarks.lodim_vertices[0].data());

    // this is the actual method that is called in every update
    esom_cuda.embedsom(
      2.0, 0.2, scatter.points[0].data()); // boost and adjust parameters are
                                           // now passed in every call, but we
                                           // might want to cache them iside?

    static std::size_t counter = 0;

    if (++counter >= 10) {
        std::cout << esom_cuda.getAvgPointsUploadTime() << "ms \t"
                  << esom_cuda.getAvgLandmarksUploadTime() << "ms \t"
                  << esom_cuda.getAvgProcessingTime() << "ms" << std::endl;
    }
    counter %= 10;
#endif
}
