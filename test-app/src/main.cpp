#include "app/Application.h"
#include <spdlog/spdlog.h>
#include <cstddef>
#include <vector>

#include "ShoeboxAcousticsScene.h"

int main()
{
    spdlog::info("[TestApp] Starting");

    Application app;
    if (!app.Initialize())
    {
        spdlog::error("[TestApp] Application::Initialize() failed — aborting");
        return -1;
    }

    ShoeboxAcousticsScene shoeboxScene;

    bool shoeboxOk = shoeboxScene.Setup();

    if (!shoeboxOk)
        spdlog::error("[TestApp] ShoeboxAcousticsScene::Setup() failed");

    std::vector<Scene*> scenes;
    if (shoeboxOk) scenes.push_back(&shoeboxScene);

    for (std::size_t i = 0; i < scenes.size(); ++i)
        spdlog::info("[TestApp] Press {} for {}", i + 1, scenes[i]->GetName());

    if (!scenes.empty())
        app.Run(scenes, 0);
    else
        spdlog::error("[TestApp] No scenes were available to run");

    spdlog::info("[TestApp] Shutting down");
    return 0;
}
