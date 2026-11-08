#pragma once
#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include "acoustics/AcousticSimulation.h"
#include "acoustics/AudioEngine.h"
#include "acoustics/RoomMetrics.h"
#include "core/IBLDebugMode.h"

class Scene;
class MeshBuffer;
class Renderer;
class MouseInput;
class ShaderProgram;
struct RendererDebugStats;
struct FrameSubmission;

/// Sidebar tab identifiers used by the Inspector.
enum class UITab : int {
    Scene = 0,
    Lights = 1,
    Materials = 2,
    Shadow = 3,
    PostFX = 4,
    Stats = 5,
    Terrain = 6,  ///< Only shown when the active scene returns HasTerrainTab() == true
    Acoustics = 7,
};

class RendererUI {
public:
    RendererUI();
    ~RendererUI();

    /// Apply the global ImGui theme. Call once after ImGui::CreateContext(),
    /// or whenever you want to re-apply (e.g. after a style reset).
    /// Must NOT be called inside a NewFrame/Render block.
    static void ApplyTheme();

    /// Draw the full UI for the current frame (call between NewFrame / Render).
    void Draw(int fbWidth, int fbHeight,
              Scene &scene,
              Renderer &renderer,
              MouseInput *mouse,
              const std::vector<Scene *> &scenes,
              std::size_t &activeSceneIndex);

    /// Draw the last simulation's debug geometry into the current scene framebuffer.
    void DrawAcousticDebug(const Scene &scene);

    /// Returns the viewport rectangle [x, y, w, h] in framebuffer pixels.
    /// Forward this to Renderer::Resize() / SetViewport() every frame.
    void GetViewportRect(int &x, int &y, int &w, int &h) const {
        x = m_vpX;
        y = m_vpY;
        w = m_vpW;
        h = m_vpH;
    }

    // Public toggles
    bool wireframeOverride = false;
    bool normalMapOverride = true;
    bool skyboxOverride = true;
    // After: bool skyboxOverride = true;
    bool  tonemapEnabled        = true;
    int   tonemapOperator       = 1;   // 0=Reinhard, 1=ACES, 2=Uncharted2
    float exposure              = 1.0f;
    bool  bloomEnabled          = true;
    float bloomStrength         = 0.25f;
    float bloomThreshold        = 1.0f;
    bool  bloomSoftThreshold    = true;
    float bloomSoftKnee         = 0.0f;
    float bloomRadius           = 0.1f;
    int   bloomBlurIterations   = 4;
    int   postFxDebugView       = 0;   // 0=Final, 1=HDR only, 2=Bright-pass, 3=Blurred bloom, 4=No bloom
    IBLDebugMode iblDebugMode = kDefaultIBLDebugMode;
    float iblDebugPrefilteredMip = 0.0f;
    float ambientFloorStrength = 0.18f;
    float maxShadowOcclusion = 0.75f;
    bool frustumCullingEnabled = true;
    bool showHelpWindow = false;
    bool showSidebar = true;

private:
    // Viewport rect in framebuffer pixels (updated every Draw call).
    int m_vpX = 0, m_vpY = 0, m_vpW = 0, m_vpH = 0;
    int m_prevVpW = -1, m_prevVpH = -1;

    UITab m_activeTab = UITab::Scene;

    struct AcousticRun {
        acoustics::Receiver listener;
        acoustics::Echogram echogram;
        acoustics::TraceStats stats;
        std::size_t triangleCount = 0;
        acoustics::AirConditions air;
        float speedOfSound = 0.0f;
        acoustics::RoomMetricsResult metrics;
        float enclosedVolume = 0.0f;
        std::array<float, OctaveBands::Count> sabineRt60{};
        std::array<float, OctaveBands::Count> eyringRt60{};
        int plottedEchoBand = -1;
        std::vector<float> leftEchoPeaks;
        std::vector<float> rightEchoPeaks;
        std::vector<acoustics::RayDebugSegment> debugSegments;
        std::unique_ptr<MeshBuffer> debugRayMesh;
    };

    const Scene *m_acousticScene = nullptr;
    acoustics::Settings m_acousticSettings;
    acoustics::AirConditions m_acousticAir;
    acoustics::Receiver m_acousticReceiver;
    std::optional<AcousticRun> m_acousticRun;
    std::string m_acousticError;
    std::array<char, 1024> m_acousticDryPath{};
    char m_acousticOutputPath[1024] = "auralized.wav";
    std::optional<acoustics::StereoAudio> m_acousticWet;
    std::string m_acousticAudioMessage;
    int m_acousticGraphBand = 3;
    int m_acousticGraphKind = 0;
    int m_acousticMetricEar = 0;
    int m_acousticRt60Fit = 0;
    bool m_showAcousticRays = false;
    bool m_showAcousticReceiver = false;
    uint32_t m_acousticDisplayedBounces = 5;
    std::unique_ptr<ShaderProgram> m_acousticDebugShader;
    std::unique_ptr<MeshBuffer> m_acousticDebugSphere;

    // ── Draw helpers ──────────────────────────────────────────────────────────
    void DrawTopbar(int fbWidth,
                    Scene &scene,
                    const RendererDebugStats &stats,
                    const std::vector<Scene *> &scenes,
                    std::size_t &activeSceneIndex,
                    bool lookMode);

    // `frame` is the submission already built by Draw() — do not rebuild inside.
    void DrawSidebar(int fbHeight,
                     Scene &scene,
                     const RendererDebugStats &stats,
                     const FrameSubmission &frame,
                     bool lookMode);

    /// Viewport overlay: HUD pills, gizmo row, camera-mode badge.
    // `frame` is the submission already built by Draw() — do not rebuild inside.
    void DrawViewport(int fbWidth, int fbHeight,
                      const RendererDebugStats &stats,
                      Scene &scene,
                      const FrameSubmission &frame,
                      bool lookMode);

    // ── Tab content ───────────────────────────────────────────────────────────
    // All tabs receive the pre-built FrameSubmission so they never call
    // scene.BuildSubmission() themselves.
    void DrawTabScene(Scene &scene, const RendererDebugStats &stats,
                      const FrameSubmission &frame);

    void DrawTabLights(Scene &scene, const RendererDebugStats &stats,
                       const FrameSubmission &frame);

    void DrawTabMaterials(Scene &scene, const RendererDebugStats &stats,
                          const FrameSubmission &frame);

    void DrawTabShadow(Scene &scene, const RendererDebugStats &stats);

    void DrawTabPostFX(Scene &scene, const RendererDebugStats &stats);

    void DrawTabStats(Scene &scene, const RendererDebugStats &stats);

    void DrawTabTerrain(Scene &scene);

    void DrawTabAcoustics(Scene &scene, const FrameSubmission &frame);
    void DrawAcousticGraphs(AcousticRun &run);
    void DrawAcousticMetrics(const AcousticRun &run);
    void ResetAcousticAudio();
    void PlayAcousticResponse(const acoustics::MonoAudio &dry);

    void DrawHelpWindow(int fbWidth, int fbHeight);
};
