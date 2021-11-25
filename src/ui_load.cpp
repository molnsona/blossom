
#include "application.h"
#include "parsers.h"
#include <exception>

uiLoader::uiLoader()
{
    opener.SetTitle("Open file");
    opener.SetTypeFilters(
      { ".fcs", ".tsv" }); // TODO take this from parsers, somehow
}

void
uiLoader::render(Application &app)
{
    opener.Display();

    if (opener.HasSelected()) {
        try {
            parse_generic(opener.GetSelected().string(), app.state.data);
        } catch (std::exception &e) {
            loading_error = e.what();
        }

        app.state.trans.reset(
          app.state.data); // TODO have State handler for this
        app.state.training_conf.reset_data();

        opener.ClearSelected();
    }

    if (!loading_error.empty()) {
        ImGui::Begin("Loading error", nullptr, 0);
        ImGui::Text(loading_error.c_str());
        if (ImGui::Button("OK"))
            loading_error = "";
        ImGui::End();
    }
}
