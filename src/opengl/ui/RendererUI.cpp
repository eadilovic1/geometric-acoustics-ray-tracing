#include "ui/RendererUI.h"
#include "ui/UITheme.h"

#include "acoustics/AcousticSimulation.h"
#include "acoustics/AcousticGeometry.h"
#include "acoustics/BVH.h"
#include "acoustics/AcousticMaterial.h"
#include "core/Camera.h"
#include "core/Primitives.h"
#include "core/Renderer.h"
#include "core/MouseInput.h"
#include "core/ShaderProgram.h"
#include "scene/Scene.h"
#include "scene/LightEnvironment.h"
#include "scene/FrameSubmission.h"
#include "core/Material.h"
#include <imgui.h>
#include <imgui_internal.h>
#include <glm/gtc/matrix_transform.hpp>
#include <cstdarg>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <exception>
#include <utility>

// Pal colours and SectionHeader are now in UITheme.h (shared with scene tabs).

RendererUI::RendererUI() = default;
RendererUI::~RendererUI() = default;

namespace {
struct AcousticDebugGLState {
    GLboolean depthTest = glIsEnabled(GL_DEPTH_TEST);
    GLboolean blend = glIsEnabled(GL_BLEND);
    GLboolean cull = glIsEnabled(GL_CULL_FACE);
    GLboolean depthWrite;
    GLint program, vao, arrayBuffer;
    GLint depthFunc, cullMode;
    GLint blendSrcRgb, blendDstRgb, blendSrcAlpha, blendDstAlpha, blendEqRgb, blendEqAlpha;

    AcousticDebugGLState() {
        glGetBooleanv(GL_DEPTH_WRITEMASK, &depthWrite);
        glGetIntegerv(GL_CURRENT_PROGRAM, &program);
        glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &vao);
        glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &arrayBuffer);
        glGetIntegerv(GL_DEPTH_FUNC, &depthFunc);
        glGetIntegerv(GL_CULL_FACE_MODE, &cullMode);
        glGetIntegerv(GL_BLEND_SRC_RGB, &blendSrcRgb);
        glGetIntegerv(GL_BLEND_DST_RGB, &blendDstRgb);
        glGetIntegerv(GL_BLEND_SRC_ALPHA, &blendSrcAlpha);
        glGetIntegerv(GL_BLEND_DST_ALPHA, &blendDstAlpha);
        glGetIntegerv(GL_BLEND_EQUATION_RGB, &blendEqRgb);
        glGetIntegerv(GL_BLEND_EQUATION_ALPHA, &blendEqAlpha);
    }

    ~AcousticDebugGLState() {
        glUseProgram(program);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, arrayBuffer);
        glDepthMask(depthWrite);
        glDepthFunc(depthFunc);
        glCullFace(cullMode);
        glBlendFuncSeparate(blendSrcRgb, blendDstRgb, blendSrcAlpha, blendDstAlpha);
        glBlendEquationSeparate(blendEqRgb, blendEqAlpha);
        (depthTest ? glEnable : glDisable)(GL_DEPTH_TEST);
        (blend ? glEnable : glDisable)(GL_BLEND);
        (cull ? glEnable : glDisable)(GL_CULL_FACE);
    }
};
} // namespace

static constexpr float kPanelPadding = 12.0f;
static constexpr float kRounding = 10.0f;
static constexpr float kTopbarHeight = 44.0f;
static constexpr float kSidebarWidth = 360.0f;

struct InspectorTab {
    UITab id;
    const char *label;
};

static constexpr InspectorTab kInspectorTabs[] = {
    {UITab::Scene, "Scene"},
    {UITab::Lights, "Lights"},
    {UITab::Materials, "Mat"},
    {UITab::Shadow, "Shadow"},
    {UITab::PostFX, "Post"},
    {UITab::Stats, "Stats"},
    {UITab::Acoustics, "Acoustics"},
    {UITab::Terrain, "Terr"},
};

// ─────────────────────────────────────────────────────────────────────────────
// ApplyTheme — call once after ImGui::CreateContext(), NOT inside a frame.
// ─────────────────────────────────────────────────────────────────────────────
void RendererUI::ApplyTheme() {
    ImGuiStyle &style = ImGui::GetStyle();
    style.WindowRounding = kRounding;
    style.ChildRounding = kRounding * 0.5f;
    style.PopupRounding = kRounding * 0.5f;
    style.FrameRounding = 4.0f;
    style.GrabRounding = 4.0f;
    style.WindowBorderSize = 1.0f;
    style.PopupBorderSize = 1.0f;
    style.FrameBorderSize = 0.0f;
    style.WindowPadding = ImVec2(12, 12);
    style.DisplaySafeAreaPadding = ImVec2(8, 8);
    style.ItemSpacing = ImVec2(8, 8);
    style.Colors[ImGuiCol_WindowBg] = Pal::Bg1;
    style.Colors[ImGuiCol_Border] = Pal::Border;
}

// ─────────────────────────────────────────────────────────────────────────────
// Small widget helpers
// ─────────────────────────────────────────────────────────────────────────────
static std::string FormatCompact(uint64_t v) {
    if (v >= 1'000'000ull)
        return std::to_string(v / 1'000'000ull) + "." +
               std::to_string((v % 1'000'000ull) / 100'000ull) + "M";
    if (v >= 1'000ull)
        return std::to_string(v / 1'000ull) + "." +
               std::to_string((v % 1'000ull) / 100ull) + "k";
    return std::to_string(v);
}

static bool ProjectWorldToViewport(const Camera &camera,
                                   const glm::vec3 &world,
                                   float viewportWidth,
                                   float viewportHeight,
                                   ImVec2 &outScreen) {
    const glm::vec4 clip = camera.GetViewProjection() * glm::vec4(world, 1.0f);
    if (clip.w <= 0.0001f) return false;

    const glm::vec3 ndc = glm::vec3(clip) / clip.w;
    if (ndc.z < -1.0f || ndc.z > 1.0f) return false;

    outScreen.x = (ndc.x * 0.5f + 0.5f) * viewportWidth;
    outScreen.y = (1.0f - (ndc.y * 0.5f + 0.5f)) * viewportHeight;
    return true;
}

static void DrawCenteredOverlayText(ImDrawList *drawList,
                                    const ImVec2 &pos,
                                    ImU32 color,
                                    const char *text) {
    const ImVec2 size = ImGui::CalcTextSize(text);
    const ImVec2 min(pos.x - size.x * 0.5f - 8.0f, pos.y - size.y * 0.5f - 4.0f);
    const ImVec2 max(pos.x + size.x * 0.5f + 8.0f, pos.y + size.y * 0.5f + 4.0f);
    drawList->AddRectFilled(min, max, IM_COL32(12, 14, 18, 190), 6.0f);
    drawList->AddRect(min, max, IM_COL32(255, 255, 255, 28), 6.0f);
    drawList->AddText(ImVec2(pos.x - size.x * 0.5f, pos.y - size.y * 0.5f), color, text);
}

static bool MiniBadgeButton(const char *label, bool danger = false, bool add = false) {
    if (danger) {
        ImGui::PushStyleColor(ImGuiCol_Button, Pal::RedBg);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, {0.30f, 0.08f, 0.08f, 1.f});
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, {0.15f, 0.04f, 0.04f, 1.f});
        ImGui::PushStyleColor(ImGuiCol_Text, Pal::Red);
    } else if (add) {
        ImGui::PushStyleColor(ImGuiCol_Button, {0.05f, 0.18f, 0.10f, 1.f});
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, {0.08f, 0.26f, 0.14f, 1.f});
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, {0.04f, 0.14f, 0.08f, 1.f});
        ImGui::PushStyleColor(ImGuiCol_Text, {0.30f, 0.62f, 0.43f, 1.f});
    } else {
        ImGui::PushStyleColor(ImGuiCol_Button, Pal::Bg3);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, {0.15f, 0.15f, 0.19f, 1.f});
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, Pal::Bg2);
        ImGui::PushStyleColor(ImGuiCol_Text, Pal::TextDim);
    }
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(6, 2));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 3.0f);
    const bool clicked = ImGui::Button(label);
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor(4);
    return clicked;
}

// SectionHeader is now in UITheme.h.

// ─────────────────────────────────────────────────────────────────────────────
// Light editors
// ─────────────────────────────────────────────────────────────────────────────
static void DrawDirectionalLight(DirectionalLight &light) {
    ImGui::PushStyleColor(ImGuiCol_Text, Pal::TextMid);
    ImGui::DragFloat("Intensity##dir", &light.intensity, 0.05f, 0.0f, 1000.0f, "%.3f");
    ImGui::ColorEdit3("Color##dir", &light.color.x);

    const glm::vec3 &d = light.direction;
    float elev = glm::degrees(std::asin(glm::clamp(d.y, -1.0f, 1.0f)));
    float azim = glm::degrees(std::atan2(d.z, d.x));
    bool changed = false;
    changed |= ImGui::DragFloat("Azimuth (°)##dir", &azim, 0.5f, -180.f, 180.f, "%.1f");
    changed |= ImGui::DragFloat("Elevation (°)##dir", &elev, 0.5f, -90.f, 90.f, "%.1f");
    if (changed) {
        const float pr = glm::radians(elev), yr = glm::radians(azim);
        light.direction = {
            std::cos(pr) * std::cos(yr),
            std::sin(pr),
            std::cos(pr) * std::sin(yr)
        };
    }
    ImGui::PushStyleColor(ImGuiCol_Text, Pal::TextFaint);
    ImGui::Text("XYZ: %.3f, %.3f, %.3f", d.x, d.y, d.z);
    ImGui::PopStyleColor(2);
}

// Returns true → caller should remove this light.
static bool DrawPointLight(PointLight &light, int idx, const glm::vec3 &camPos) {
    // Use ImGui's ID stack for uniqueness — no per-frame string allocation needed.
    ImGui::PushID(idx);

    ImGui::PushStyleColor(ImGuiCol_ChildBg, Pal::Bg3);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 4.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8, 6));

    ImGui::BeginChild("##plc", ImVec2(0, 0),
                      ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_Border);

    // Header row — display name, then remove button flush-right.
    char nameBuf[64];
    if (light.name.empty())
        std::snprintf(nameBuf, sizeof(nameBuf), "Light %d", idx + 1);
    else
        std::snprintf(nameBuf, sizeof(nameBuf), "%s", light.name.c_str());

    ImGui::TextUnformatted(nameBuf);
    ImGui::SameLine(ImGui::GetContentRegionAvail().x - 26);
    const bool remove = MiniBadgeButton("X##rm", /*danger=*/true);

    ImGui::Separator();
    ImGui::PushStyleColor(ImGuiCol_Text, Pal::TextMid);

    // All widgets share the same ID scope (PushID above), so plain labels are fine.
    ImGui::DragFloat("Intensity", &light.intensity, 0.05f, 0.f, 1000.f, "%.3f");
    ImGui::ColorEdit3("Color", &light.color.x);
    ImGui::DragFloat3("Position", &light.position.x, 0.05f);
    ImGui::DragFloat("Radius", &light.radius, 0.5f, 0.1f, 500.f, "%.1f");

    if (MiniBadgeButton("Move to camera"))
        light.position = camPos;
    ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_Text, Pal::TextFaint);
    ImGui::Text("(%.1f, %.1f, %.1f)", camPos.x, camPos.y, camPos.z);
    ImGui::PopStyleColor(2);

    ImGui::EndChild();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor();
    ImGui::Spacing();

    ImGui::PopID();
    return remove;
}

static void DrawShadowParams(LightShadowParams &s, const char *farLabel,
                             const char *note, bool dirNote) {
    ImGui::PushStyleColor(ImGuiCol_Text, Pal::TextMid);
    ImGui::Checkbox("Cast shadow", &s.castShadow);

    // Clamp before presenting so downstream consumers never see a bad value.
    s.nearPlane = std::max(s.nearPlane, 0.001f);
    if (s.farPlane <= s.nearPlane) s.farPlane = s.nearPlane + 0.01f;

    ImGui::DragFloat("Depth bias", &s.depthBias, 0.0001f, 0.f, 1.f, "%.5f");
    ImGui::DragFloat("Normal bias", &s.normalBias, 0.0005f, 0.f, 10.f, "%.4f");
    ImGui::DragFloat("Slope bias", &s.slopeBias, 0.050f, 0.f, 10.f, "%.3f");

    int res[2] = {s.shadowMapWidth, s.shadowMapHeight};
    if (ImGui::DragInt2("Resolution", res, 4.f, 16, 8192)) {
        s.shadowMapWidth = std::max(16, res[0]);
        s.shadowMapHeight = std::max(16, res[1]);
    }

    ImGui::DragFloat("Near plane", &s.nearPlane, 0.01f, 0.001f, 1000.f, "%.3f");
    ImGui::DragFloat(farLabel, &s.farPlane, 0.05f, 0.01f, 10000.f, "%.3f");

    // Re-clamp after drag in case user pulled near past far.
    s.nearPlane = std::max(s.nearPlane, 0.001f);
    if (s.farPlane <= s.nearPlane) s.farPlane = s.nearPlane + 0.01f;

    ImGui::SliderInt("PCF radius", &s.pcfRadius, 0, 4);
    s.pcfRadius = std::max(0, s.pcfRadius);

    const int k = s.pcfRadius * 2 + 1;
    ImGui::PushStyleColor(ImGuiCol_Text, Pal::TextFaint);
    ImGui::Text("PCF kernel: %dx%d (%d taps)", k, k, k * k);
    ImGui::TextWrapped("%s", note);
    if (dirNote) ImGui::TextWrapped("Directional cast/bias packed in light UBO.");
    ImGui::PopStyleColor(2);
}

// ─────────────────────────────────────────────────────────────────────────────
// Top-level Draw — builds the shared FrameSubmission once, then dispatches.
// ─────────────────────────────────────────────────────────────────────────────
void RendererUI::Draw(int fbW, int fbH,
                      Scene &scene, Renderer &renderer, MouseInput *mouse,
                      const std::vector<Scene *> &scenes,
                      std::size_t &activeSceneIndex) {
    const bool lookMode = mouse && mouse->IsCaptured();
    const RendererDebugStats &stats = renderer.GetDebugStats();

    if (m_acousticScene != &scene) {
        m_acousticScene = &scene;
        m_acousticRun.reset();
        m_acousticError.clear();
        ResetAcousticAudio();
        m_bvhCachedScene = nullptr;
        m_cachedAcousticGeometry.reset();
        m_cachedAcousticBvh.reset();
    }

    // ── Viewport fills the entire window ─────────────────────────────────────
    m_vpX = 0;
    m_vpY = 0;
    m_vpW = fbW;
    m_vpH = fbH;
    if (m_vpW != m_prevVpW || m_vpH != m_prevVpH) {
        renderer.Resize(m_vpW, m_vpH);
        m_prevVpW = m_vpW;
        m_prevVpH = m_vpH;
    }

    // ── Build scene submission exactly once for the whole frame ───────────────
    FrameSubmission frame;
    scene.BuildSubmission(frame);

    if (frame.camera && m_acousticFollowCamera) {
        m_acousticReceiver.FollowCamera(*frame.camera);
    }

    // ── Draw regions ──────────────────────────────────────────────────────────
    DrawTopbar(fbW, scene, stats, scenes, activeSceneIndex, lookMode);

    if (showSidebar)
        DrawSidebar(fbH, scene, stats, frame, lookMode);

    DrawViewport(fbW, fbH, stats, scene, frame, lookMode);

    if (showHelpWindow)
        DrawHelpWindow(fbW, fbH);
}

// ─────────────────────────────────────────────────────────────────────────────
// Topbar — centered Dynamic Island pill
// ─────────────────────────────────────────────────────────────────────────────
void RendererUI::DrawTopbar(int fbW,
                            Scene &scene,
                            const RendererDebugStats &stats,
                            const std::vector<Scene *> &scenes,
                            std::size_t &activeSceneIndex,
                            bool lookMode) {
    const float pillWidth = 520.0f;
    const float pillHeight = kTopbarHeight;
    const float pillX = (static_cast<float>(fbW) - pillWidth) * 0.5f;
    const float pillY = kPanelPadding;

    ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration |
                             ImGuiWindowFlags_NoMove |
                             ImGuiWindowFlags_NoScrollbar |
                             ImGuiWindowFlags_NoSavedSettings;
    if (lookMode) flags |= ImGuiWindowFlags_NoMouseInputs;

    ImGui::SetNextWindowPos({pillX, pillY}, ImGuiCond_Always);
    ImGui::SetNextWindowSize({pillWidth, pillHeight}, ImGuiCond_Always);

    ImGui::PushStyleColor(ImGuiCol_WindowBg, Pal::Bg1);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, pillHeight * 0.5f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(16, 0));

    if (ImGui::Begin("##DynamicIsland", nullptr, flags)) {
        // Helper: transparent menu-bar button that highlights when active.
        auto MenuButton = [&](const char *label, bool isActive,
                              const char *hotkey = nullptr) -> bool {
            ImGui::PushStyleColor(ImGuiCol_Button, isActive
                                                       ? Pal::AccentDim
                                                       : ImVec4(0, 0, 0, 0));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, {1.0f, 1.0f, 1.0f, 0.05f});
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, Pal::AccentDim);
            ImGui::PushStyleColor(ImGuiCol_Text, isActive
                                                     ? Pal::Accent
                                                     : Pal::TextMid);
            ImGui::SetCursorPosY((pillHeight - ImGui::GetFrameHeight()) * 0.5f);
            const bool clicked = ImGui::Button(label);
            if (ImGui::IsItemHovered() && hotkey) {
                ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8, 8));
                ImGui::SetTooltip("Hotkey: %s", hotkey);
                ImGui::PopStyleVar();
            }
            ImGui::PopStyleColor(4);
            return clicked;
        };

        ImGui::SameLine(0, 20);
        ImGui::SetCursorPosY(0);

        // --- Scenes ---
        MenuButton("Scenes", false);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12, 12));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8, 8));
        if (ImGui::BeginPopupContextItem("##scenesPopup", ImGuiPopupFlags_MouseButtonLeft)) {
            for (std::size_t i = 0; i < scenes.size(); ++i) {
                if (!scenes[i]) continue;
                char shortcut[8] = "";
                if (i < 9)
                    std::snprintf(shortcut, sizeof(shortcut), "%zu", i + 1);
                if (ImGui::MenuItem(scenes[i]->GetName().c_str(), i < 9 ? shortcut : nullptr,
                                    i == activeSceneIndex))
                    activeSceneIndex = i;
            }
            ImGui::EndPopup();
        }
        ImGui::PopStyleVar(2);

        ImGui::SameLine(0, 8);

        // --- View ---
        if (MenuButton("View", wireframeOverride || showSidebar || !skyboxOverride, "V"))
            ImGui::OpenPopup("##viewPopup");

        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12, 12));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8, 8));
        if (ImGui::BeginPopup("##viewPopup")) {
            if (ImGui::MenuItem("Inspector", "S", showSidebar))
                showSidebar = !showSidebar;
            ImGui::Separator();
            if (ImGui::MenuItem("Skybox", nullptr, skyboxOverride)) {
                skyboxOverride = !skyboxOverride;
                scene.SetSkyboxVisible(skyboxOverride);
            }
            if (ImGui::MenuItem("Wireframe", "Z", wireframeOverride))
                wireframeOverride = !wireframeOverride;
            if (ImGui::MenuItem("Shaded", nullptr, !wireframeOverride))
                wireframeOverride = false;
            ImGui::EndPopup();
        }
        ImGui::PopStyleVar(2);

        ImGui::SameLine(0, 8);

        // --- PostFX debug / integration menu ---
        if (MenuButton("PostFX", tonemapEnabled || bloomEnabled || postFxDebugView != 0, nullptr))
            ImGui::OpenPopup("##postFxPopup");

        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12, 12));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8, 8));
        if (ImGui::BeginPopup("##postFxPopup")) {
            ImGui::TextColored(Pal::TextDim, "Intermediate View");
            static const char *kDebugViews[] = {
                "Final", "HDR Only", "Bright-pass", "Blurred Bloom", "No Bloom"
            };
            for (int i = 0; i < IM_ARRAYSIZE(kDebugViews); ++i) {
                if (ImGui::MenuItem(kDebugViews[i], nullptr, postFxDebugView == i))
                    postFxDebugView = i;
            }

            ImGui::Separator();
            ImGui::MenuItem("Tone Mapping", nullptr, &tonemapEnabled);
            ImGui::MenuItem("Bloom", nullptr, &bloomEnabled);
            ImGui::SliderFloat("Exposure", &exposure, 0.1f, 5.0f, "%.2f");
            ImGui::SliderFloat("Bloom Strength", &bloomStrength, 0.0f, 3.0f, "%.2f");
            ImGui::SliderFloat("Bloom Threshold", &bloomThreshold, 0.0f, 2.0f, "%.2f");
            ImGui::EndPopup();
        }
        ImGui::PopStyleVar(2);

        ImGui::SameLine(0, 8);

        // --- Help ---
        if (MenuButton("Help", showHelpWindow, "H"))
            showHelpWindow = !showHelpWindow;

        // --- Mini FPS pill ---
        ImGui::SetCursorPos({
            pillWidth - 80.0f,
            (pillHeight - ImGui::GetTextLineHeight()) * 0.5f
        });
        ImGui::TextColored(Pal::Green, "%.0f", stats.fps);
        ImGui::SameLine(0, 4);
        ImGui::TextColored(Pal::TextDim, "FPS");
    }
    ImGui::End();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor();
}

// ─────────────────────────────────────────────────────────────────────────────
// Sidebar — floating Inspector panel
// ─────────────────────────────────────────────────────────────────────────────
void RendererUI::DrawSidebar(int fbH,
                             Scene &scene,
                             const RendererDebugStats &stats,
                             const FrameSubmission &frame,
                             bool lookMode) {
    const float sbH = static_cast<float>(fbH) - (kPanelPadding * 2.0f);

    ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration |
                             ImGuiWindowFlags_NoMove |
                             ImGuiWindowFlags_NoSavedSettings;
    if (lookMode) flags |= ImGuiWindowFlags_NoMouseInputs;

    ImGui::SetNextWindowPos({kPanelPadding, kPanelPadding}, ImGuiCond_Always);
    ImGui::SetNextWindowSize({kSidebarWidth, sbH}, ImGuiCond_Always);

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, kRounding);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, Pal::Bg1);

    if (ImGui::Begin("##Inspector", nullptr, flags)) {
        // --- Pill-style tab bar ---
        const bool hasTerrain = scene.HasTerrainTab();
        if (!hasTerrain && m_activeTab == UITab::Terrain)
            m_activeTab = UITab::Scene;

        constexpr int tabsPerRow = 4;
        constexpr float tabHeight = 30.0f;
        constexpr float tabBarWidth = kSidebarWidth - 32.0f;
        constexpr float tabWidth = tabBarWidth / tabsPerRow;
        ImGui::SetCursorPos({16, 16});
        ImGui::BeginChild("##tabBar", ImVec2(tabBarWidth, 2 * tabHeight + 6),
                          false, ImGuiWindowFlags_NoScrollbar);

        int visibleIndex = 0;
        for (const auto &tab : kInspectorTabs) {
            if (tab.id == UITab::Terrain && !hasTerrain)
                continue;
            if (visibleIndex % tabsPerRow != 0)
                ImGui::SameLine(0, 0);
            ++visibleIndex;

            const bool active = m_activeTab == tab.id;
            ImGui::PushStyleColor(ImGuiCol_Button, active
                                                       ? Pal::Accent
                                                       : ImVec4(0, 0, 0, 0));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, active
                                                              ? Pal::Accent
                                                              : ImVec4(1, 1, 1, 0.05f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, Pal::Accent);
            ImGui::PushStyleColor(ImGuiCol_Text, active
                                                     ? Pal::TextHi
                                                     : Pal::TextDim);
            ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 18.0f);
            ImGui::PushStyleVar(ImGuiStyleVar_ButtonTextAlign, ImVec2(0.5f, 0.5f));

            if (ImGui::Button(tab.label, ImVec2(tabWidth, tabHeight)))
                m_activeTab = tab.id;

            ImGui::PopStyleVar(2);
            ImGui::PopStyleColor(4);
        }
        ImGui::EndChild();

        ImGui::Separator();

        // --- Scrollable content area ---
        // For the terrain tab we reserve a fixed strip at the bottom for the
        // sticky Regenerate bar so it is always visible without scrolling.
        const bool hasTerrainFooter = (m_activeTab == UITab::Terrain) && hasTerrain;
        constexpr float kFooterH = 68.0f;

        ImGui::SetCursorPosX(0);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(16, 16));
        ImGui::BeginChild("##content", ImVec2(0, hasTerrainFooter ? -kFooterH : 0.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8, 12));

        switch (m_activeTab) {
            case UITab::Scene:     DrawTabScene(scene, stats, frame);    break;
            case UITab::Lights:    DrawTabLights(scene, stats, frame);   break;
            case UITab::Materials: DrawTabMaterials(scene, stats, frame); break;
            case UITab::Shadow:    DrawTabShadow(scene, stats);          break;
            case UITab::PostFX:    DrawTabPostFX(scene, stats);          break;
            case UITab::Stats:     DrawTabStats(scene, stats);           break;
            case UITab::Terrain:   DrawTabTerrain(scene);                break;
            case UITab::Acoustics: DrawTabAcoustics(scene, frame);       break;
        }

        if (lookMode) {
            ImGui::Spacing();
            ImGui::PushStyleColor(ImGuiCol_Text, Pal::TextFaint);
            ImGui::Separator();
            ImGui::TextUnformatted("TAB — release mouse for UI");
            ImGui::PopStyleColor();
        }

        ImGui::PopStyleVar(2);
        ImGui::EndChild();

        // --- Sticky terrain footer (always visible at the bottom of the sidebar) ---
        if (hasTerrainFooter) {
            ImGui::PushStyleColor(ImGuiCol_Separator, ImVec4(1, 1, 1, 0.07f));
            ImGui::Separator();
            ImGui::PopStyleColor();
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(6, 6));
            ImGui::SetCursorPosX(12);
            scene.OnTerrainTabFooter();
            ImGui::PopStyleVar();
        }
    }
    ImGui::End();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(3);
}

// ─────────────────────────────────────────────────────────────────────────────
// Viewport overlay — HUD; transparent, sits on top of GL content
// ─────────────────────────────────────────────────────────────────────────────
void RendererUI::DrawViewport(int fbW, int fbH,
                              const RendererDebugStats &stats,
                              Scene &scene,
                              const FrameSubmission &frame,
                              bool lookMode) {
    const float vpW = static_cast<float>(fbW);
    const float vpH = static_cast<float>(fbH);

    ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration |
                             ImGuiWindowFlags_NoMove |
                             ImGuiWindowFlags_NoSavedSettings |
                             ImGuiWindowFlags_NoNav |
                             ImGuiWindowFlags_NoScrollbar |
                             ImGuiWindowFlags_NoBringToFrontOnFocus;
    if (lookMode) flags |= ImGuiWindowFlags_NoMouseInputs;

    ImGui::SetNextWindowPos({0, 0}, ImGuiCond_Always);
    ImGui::SetNextWindowSize({vpW, vpH}, ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.0f);

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(kPanelPadding, kPanelPadding));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0, 0, 0, 0));

    if (ImGui::Begin("##VP_Overlay", nullptr, flags)) {
        // ── Camera mode string — use the pre-built submission ─────────────────
        const char *modeStr = "FreeFly";
        if (frame.camera) {
            switch (frame.camera->GetMode()) {
                case CameraMode::FirstPerson: modeStr = "First Person";
                    break;
                case CameraMode::ThirdPerson: modeStr = "Third Person";
                    break;
                default: break;
            }
        }

        // ── Bottom-right HUD pills ─────────────────────────────────────────────
        char resBuf[32], dcBuf[32];
        std::snprintf(resBuf, sizeof(resBuf), "%d × %d", fbW, fbH);
        std::snprintf(dcBuf, sizeof(dcBuf), "%u draw calls", stats.drawCallCount);

        const float pillH = 26.0f;
        const float pillGap = 6.0f;
        float py = vpH - pillH - kPanelPadding;

        // Returns true if the pill was clicked.
        auto HudPill = [&](const char *txt, bool accent,
                           const char *hotkey = nullptr) -> bool {
            const float tw = ImGui::CalcTextSize(txt).x;
            const float pw = tw + 20.0f;
            ImGui::SetCursorPos({vpW - pw - kPanelPadding, py});

            ImGui::PushStyleColor(ImGuiCol_Button, {0.07f, 0.07f, 0.08f, 0.7f});
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, {0.10f, 0.10f, 0.12f, 0.8f});
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, {0.05f, 0.05f, 0.06f, 0.9f});
            ImGui::PushStyleColor(ImGuiCol_Text, accent ? Pal::Accent : Pal::TextDim);
            ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, pillH * 0.5f);

            const bool clicked = ImGui::Button(txt, ImVec2(pw, pillH));

            if (ImGui::IsItemHovered() && hotkey) {
                ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8, 8));
                ImGui::SetTooltip("Hotkey: %s", hotkey);
                ImGui::PopStyleVar();
            }
            ImGui::PopStyleVar();
            ImGui::PopStyleColor(4);
            py -= (pillH + pillGap);
            return clicked;
        };

        HudPill(dcBuf, /*accent=*/false);
        HudPill(resBuf, /*accent=*/false);
        if (HudPill(modeStr, /*accent=*/true, "C (switch)")) {
            Camera &cam = scene.GetCamera();
            CameraMode m = static_cast<CameraMode>(
                (static_cast<int>(cam.GetMode()) + 1) % 3);
            cam.SetMode(m);
        }

        if (scene.GetName() == "PBR Validation" && frame.camera) {
            ImDrawList *drawList = ImGui::GetWindowDrawList();

            ImVec2 roughLeft, roughRight, roughLabel;
            ImVec2 metalBottom, metalTop, metalLabel;

            const bool haveRoughLeft = ProjectWorldToViewport(*frame.camera, {-3.04f, -1.15f, 0.95f}, vpW, vpH, roughLeft);
            const bool haveRoughRight = ProjectWorldToViewport(*frame.camera, {3.04f, -1.15f, 0.95f}, vpW, vpH, roughRight);
            const bool haveRoughLabel = ProjectWorldToViewport(*frame.camera, {0.0f, -1.65f, 1.10f}, vpW, vpH, roughLabel);
            const bool haveMetalBottom = ProjectWorldToViewport(*frame.camera, {-4.35f, -1.09f, 0.25f}, vpW, vpH, metalBottom);
            const bool haveMetalTop = ProjectWorldToViewport(*frame.camera, {-4.35f, 4.99f, 0.25f}, vpW, vpH, metalTop);
            const bool haveMetalLabel = ProjectWorldToViewport(*frame.camera, {-5.55f, 1.95f, 0.60f}, vpW, vpH, metalLabel);

            if (haveRoughLeft && haveRoughRight) {
                drawList->AddLine(roughLeft, roughRight, IM_COL32(155, 178, 255, 180), 2.0f);
                drawList->AddCircleFilled(roughLeft, 4.0f, IM_COL32(180, 200, 255, 235));
                drawList->AddCircleFilled(roughRight, 4.0f, IM_COL32(255, 255, 255, 235));
                DrawCenteredOverlayText(drawList, ImVec2(roughLeft.x, roughLeft.y + 18.0f), IM_COL32(225, 232, 255, 255), "0.0");
                DrawCenteredOverlayText(drawList, ImVec2(roughRight.x, roughRight.y + 18.0f), IM_COL32(225, 232, 255, 255), "1.0");
            }

            if (haveMetalBottom && haveMetalTop) {
                drawList->AddLine(metalBottom, metalTop, IM_COL32(255, 198, 132, 180), 2.0f);
                drawList->AddCircleFilled(metalBottom, 4.0f, IM_COL32(255, 224, 190, 235));
                drawList->AddCircleFilled(metalTop, 4.0f, IM_COL32(255, 244, 224, 235));
                DrawCenteredOverlayText(drawList, ImVec2(metalBottom.x - 28.0f, metalBottom.y), IM_COL32(255, 238, 214, 255), "0.0");
                DrawCenteredOverlayText(drawList, ImVec2(metalTop.x - 28.0f, metalTop.y), IM_COL32(255, 238, 214, 255), "1.0");
            }

            if (haveRoughLabel)
                DrawCenteredOverlayText(drawList, roughLabel, IM_COL32(225, 232, 255, 255), "Roughness");

            if (haveMetalLabel)
                DrawCenteredOverlayText(drawList, metalLabel, IM_COL32(255, 238, 214, 255), "Metallic");
        }

        // ── Bottom-left gizmo row ──────────────────────────────────────────────
        {
            const float bsz = 32.0f;
            const float gx = kPanelPadding + (showSidebar ? kSidebarWidth + 8.0f : 0.0f);
            const float gy = vpH - bsz - kPanelPadding;
            ImGui::SetCursorPos({gx, gy});

            // Action-driven tool button — no string comparison, no allocation.
            auto ToolBtn = [&](const char *lbl, bool active,
                               const char *hotkey, const char *tooltip,
                               auto action) {
                ImGui::PushStyleColor(ImGuiCol_Button, active
                                                           ? Pal::Accent
                                                           : ImVec4(0.07f, 0.07f, 0.08f, 0.7f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, active
                                                                  ? Pal::Accent
                                                                  : ImVec4(1, 1, 1, 0.1f));
                ImGui::PushStyleColor(ImGuiCol_ButtonActive, Pal::Accent);
                ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 6.0f);

                if (ImGui::Button(lbl, ImVec2(bsz, bsz)))
                    action();

                if (ImGui::IsItemHovered()) {
                    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8, 8));
                    ImGui::SetTooltip("%s (Hotkey: %s)", tooltip, hotkey);
                    ImGui::PopStyleVar();
                }
                ImGui::PopStyleVar();
                ImGui::PopStyleColor(3);
                ImGui::SameLine(0, 8);
            };

            ToolBtn(showSidebar ? "«" : "»", showSidebar, "X", "Toggle Inspector",
                    [&] { showSidebar = !showSidebar; });
            ToolBtn("W", wireframeOverride, "Z", "Toggle Wireframe",
                    [&] { wireframeOverride = !wireframeOverride; });
            ToolBtn("N", normalMapOverride, "N", "Toggle Normal Maps",
                    [&] { normalMapOverride = !normalMapOverride; });
            ToolBtn("S", skyboxOverride, "K", "Toggle Skybox",
                    [&] {
                        skyboxOverride = !skyboxOverride;
                        scene.SetSkyboxVisible(skyboxOverride);
                    });
        }
    }
    ImGui::End();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar();
}

// ─────────────────────────────────────────────────────────────────────────────
// Tab: Scene
// ─────────────────────────────────────────────────────────────────────────────
void RendererUI::DrawTabScene(Scene &scene, const RendererDebugStats &stats,
                              const FrameSubmission &frame) {
    if (SectionHeader("Camera")) {
        ImGui::PushStyleColor(ImGuiCol_Text, Pal::TextMid);

        // Label/value row helper — aligns values at column 90.
        auto Row = [](const char *lbl, const char *valFmt, ...) {
            ImGui::PushStyleColor(ImGuiCol_Text, Pal::TextDim);
            ImGui::TextUnformatted(lbl);
            ImGui::PopStyleColor();
            ImGui::SameLine(90);
            va_list ap;
            va_start(ap, valFmt);
            ImGui::TextV(valFmt, ap);
            va_end(ap);
        };

        if (frame.camera) {
            const glm::vec3 p = frame.camera->GetPosition();
            Row("Position", "%.2f, %.2f, %.2f", p.x, p.y, p.z);
            Row("Yaw/Pitch", "%.1f / %.1f",
                frame.camera->GetYaw(), frame.camera->GetPitch());

            const char *modeName = "FreeFly";
            switch (frame.camera->GetMode()) {
                case CameraMode::FirstPerson: modeName = "FirstPerson";
                    break;
                case CameraMode::ThirdPerson: modeName = "ThirdPerson";
                    break;
                default: break;
            }
            ImGui::PushStyleColor(ImGuiCol_Text, Pal::TextDim);
            ImGui::TextUnformatted("Mode");
            ImGui::PopStyleColor();
            ImGui::SameLine(90);
            ImGui::TextColored(Pal::Green, "%s", modeName);

            Row("Speed", "%.1f m/s", scene.GetCurrentCameraSpeed());
        } else {
            ImGui::TextColored(Pal::TextFaint, "No camera");
        }

        ImGui::PopStyleColor();
        ImGui::Spacing();
    }

    if (SectionHeader("Render Stats")) {
        ImGui::PushStyleColor(ImGuiCol_Text, Pal::TextMid);

        auto SR = [](const char *l, const char *v) {
            ImGui::PushStyleColor(ImGuiCol_Text, Pal::TextDim);
            ImGui::TextUnformatted(l);
            ImGui::PopStyleColor();
            ImGui::SameLine(110);
            ImGui::TextUnformatted(v);
        };
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%u", stats.drawCallCount);
        SR("Draw calls", buf);
        std::snprintf(buf, sizeof(buf), "%s", FormatCompact(stats.approxTriangleCount).c_str());
        SR("Approx tris", buf);
        std::snprintf(buf, sizeof(buf), "%u", stats.submittedRenderItemCount);
        SR("Submitted", buf);
        std::snprintf(buf, sizeof(buf), "%u", stats.frustumCulledRenderItemCount);
        SR("Culled", buf);
        std::snprintf(buf, sizeof(buf), "%u", stats.visibleRenderItemCount);
        SR("Visible", buf);

        ImGui::PopStyleColor();
        ImGui::Spacing();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tab: Lights
// ─────────────────────────────────────────────────────────────────────────────
void RendererUI::DrawTabLights(Scene &scene, const RendererDebugStats & /*stats*/,
                               const FrameSubmission &frame) {
    LightEnvironment &lights = scene.GetLights();

    if (SectionHeader("Global Ambient")) {
        ImGui::ColorEdit3("Ambient Color", &lights.ambientColor.x);
        ImGui::SliderFloat("Ambient Intensity", &lights.ambientIntensity, 0.0f, 5.0f);
        ImGui::SliderFloat("Ambient Floor", &ambientFloorStrength, 0.0f, 1.0f, "%.3f");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Minimum diffuse ambient added to back-facing/unlit triangles.");
        ImGui::SliderFloat("Max Shadow Occlusion", &maxShadowOcclusion, 0.0f, 1.0f, "%.2f");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Caps directional shadow darkness. Lower = brighter shadows, higher = darker shadows.");
        ImGui::Spacing();
    }

    if (SectionHeader("Directional Light")) {
        if (lights.HasDirectionalLight())
            DrawDirectionalLight(lights.GetDirectionalLight());
        else
            ImGui::TextColored(Pal::TextFaint, "No directional light in scene.");
        ImGui::Spacing();
    }

    if (SectionHeader("Point Lights")) {
        // Camera position from the pre-built submission — no extra BuildSubmission.
        const glm::vec3 cam = frame.camera ? frame.camera->GetPosition() : glm::vec3(0.f);
        auto &pls = lights.GetPointLights();

        const bool atMax = (static_cast<int>(pls.size()) >= kMaxPointLights);
        if (atMax) ImGui::BeginDisabled();
        if (ImGui::Button("+ Add Point Light", ImVec2(-1, 0))) {
            PointLight nl;
            nl.position = cam;
            nl.color = {1, 1, 1};
            nl.intensity = 3.f;
            nl.radius = 20.f;
            nl.name = "Light " + std::to_string(pls.size() + 1);
            lights.AddPointLight(nl);
        }
        if (atMax) {
            ImGui::EndDisabled();
            ImGui::TextColored(Pal::Orange, "Max lights reached.");
        }

        ImGui::Spacing();

        if (pls.empty()) {
            ImGui::TextColored(Pal::TextFaint, "No point lights in scene.");
        } else {
            int rm = -1;
            for (std::size_t i = 0; i < pls.size(); ++i)
                if (DrawPointLight(pls[i], static_cast<int>(i), cam))
                    rm = static_cast<int>(i);

            if (rm >= 0) pls.erase(pls.begin() + rm);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tab: Materials
// ─────────────────────────────────────────────────────────────────────────────
void RendererUI::DrawTabMaterials(Scene &scene, const RendererDebugStats &stats,
                                  const FrameSubmission &frame) {

    // ── Skybox ────────────────────────────────────────────────────────────────
    if (auto* sb = scene.GetSkybox()) {
        if (SectionHeader("Skybox")) {
            ImGui::PushStyleColor(ImGuiCol_Text, Pal::TextMid);

            glm::vec3 tint = sb->GetTint();
            if (ImGui::ColorEdit3("Tint", &tint.x))
                sb->SetTint(tint);

            float exposure = sb->GetExposure();
            if (ImGui::SliderFloat("Exposure##sky", &exposure, 0.0f, 5.0f, "%.2f"))
                sb->SetExposure(exposure);

            glm::vec3 emissive = sb->GetEmissiveColor();
            if (ImGui::ColorEdit3("Emissive Color##sky", &emissive.x))
                sb->SetEmissiveColor(emissive);

            float emissiveStr = sb->GetEmissiveStrength();
            if (ImGui::SliderFloat("Emissive Strength##sky", &emissiveStr, 0.0f, 10.0f, "%.2f"))
                sb->SetEmissiveStrength(emissiveStr);

            ImGui::Spacing();
            if (ImGui::Button("Reset Skybox##sky")) {
                sb->SetTint({1.0f, 1.0f, 1.0f});
                sb->SetExposure(1.0f);
                sb->SetEmissiveColor({0.0f, 0.0f, 0.0f});
                sb->SetEmissiveStrength(0.0f);
            }

            ImGui::PopStyleColor();
            ImGui::Spacing();
        }
    }

    if (frame.objects.empty()) {
        ImGui::TextColored(Pal::TextDim, "No renderable objects in current frame.");
        return;
    }

    // Identify unique MaterialInstances in the submission
    std::vector<const MaterialInstance *> uniqueMats;
    for (const auto &item : frame.objects) {
        if (item.material) {
            bool found = false;
            for (auto m : uniqueMats) {
                if (m == item.material) {
                    found = true;
                    break;
                }
            }
            if (!found) uniqueMats.push_back(item.material);
        }
    }

    if (uniqueMats.empty()) {
        ImGui::TextColored(Pal::TextDim, "No material instances found in objects.");
        return;
    }

    ImGui::Text("   %zu Material Instances found", uniqueMats.size());
    ImGui::Separator();
    ImGui::Spacing();

    for (size_t i = 0; i < uniqueMats.size(); ++i) {
        MaterialInstance *inst = const_cast<MaterialInstance *>(uniqueMats[i]);

        char label[256];
        snprintf(label, sizeof(label), "%s##%p", inst->GetName().c_str(), (void*)inst);

        if (SectionHeader(label, /*defaultOpen=*/false)) {
            ImGui::PushID(static_cast<int>(i));

            ImGui::Indent();

            ImGui::Separator();

            // Albedo
            glm::vec3 albedo = inst->GetVec3("u_AlbedoColor", glm::vec3(1.0f));
            if (ImGui::ColorEdit3("Albedo Color", &albedo.x, ImGuiColorEditFlags_NoInputs))
                inst->SetVec3("u_AlbedoColor", albedo);

            // Metallic
            float metallic = inst->GetFloat("u_MetallicValue", 0.0f);
            if (ImGui::SliderFloat("Metallic", &metallic, 0.0f, 1.0f))
                inst->SetFloat("u_MetallicValue", metallic);

            // Roughness
            float roughness = inst->GetFloat("u_RoughnessValue", 0.5f);
            if (ImGui::SliderFloat("Roughness", &roughness, 0.0f, 1.0f))
                inst->SetFloat("u_RoughnessValue", roughness);

            // AO
            float aoStrength = inst->GetFloat("u_AoStrength", 1.0f);
            if (ImGui::SliderFloat("AO Strength", &aoStrength, 0.0f, 1.0f))
                inst->SetFloat("u_AoStrength", aoStrength);

            // Emissive
            glm::vec3 emissive = inst->GetVec3("u_EmissiveColor", glm::vec3(0.0f));
            if (ImGui::ColorEdit3("Emissive Color", &emissive.x, ImGuiColorEditFlags_NoInputs))
                inst->SetVec3("u_EmissiveColor", emissive);
            
            float emissiveStr = inst->GetFloat("u_EmissiveStrength", 1.0f);
            if (ImGui::SliderFloat("Emissive Strength", &emissiveStr, 0.0f, 20.0f))
                inst->SetFloat("u_EmissiveStrength", emissiveStr);

            ImGui::Separator();

            // Normal Mapping
            bool useNormalMap = inst->GetUseNormalMap();
            if (ImGui::Checkbox("Use Normal Map", &useNormalMap))
                inst->SetUseNormalMap(useNormalMap);

            if (useNormalMap) {
                float normalScale = inst->GetFloat("u_NormalScale", 1.0f);
                if (ImGui::SliderFloat("Normal Scale", &normalScale, 0.0f, 2.0f))
                    inst->SetFloat("u_NormalScale", normalScale);
            }

            ImGui::Spacing();
            if (ImGui::Button("Reset to Defaults")) {
                // Clear overrides in the instance by setting them to parent's values or nullopt
                // Since we don't have a 'Clear' API yet, we can just set them to standard defaults
                inst->SetVec3("u_AlbedoColor", {1, 1, 1});
                inst->SetFloat("u_MetallicValue", 0.0f);
                inst->SetFloat("u_RoughnessValue", 0.5f);
                inst->SetVec3("u_EmissiveColor", {0, 0, 0});
                inst->SetFloat("u_EmissiveStrength", 1.0f);
                inst->SetFloat("u_AoStrength", 1.0f);
                inst->SetFloat("u_NormalScale", 1.0f);
                inst->SetUseNormalMap(true);
            }

            ImGui::Unindent();
            ImGui::PopID();
            ImGui::Spacing();
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tab: Shadow
// ─────────────────────────────────────────────────────────────────────────────
void RendererUI::DrawTabShadow(Scene &scene, const RendererDebugStats &stats) {
    LightEnvironment &lights = scene.GetLights();

    const int dirSh = (lights.HasDirectionalLight() &&
                       lights.GetDirectionalLight().shadow.castShadow)
                          ? 1
                          : 0;
    int ptSh = 0;
    for (const auto &pl: lights.GetPointLights())
        if (pl.shadow.castShadow) ++ptSh;

    if (SectionHeader("Summary")) {
        ImGui::PushStyleColor(ImGuiCol_Text, Pal::TextMid);
        ImGui::Text("Dir shadow lights : %d", dirSh);
        ImGui::Text("Point shadow lights: %d", ptSh);
        ImGui::Text("Shadow draws      : %u", stats.shadowPassObjectCount);
        ImGui::PopStyleColor();
        ImGui::Spacing();
    }

    if (lights.HasDirectionalLight()) {
        if (SectionHeader("Directional Shadow", /*defaultOpen=*/false)) {
            DrawShadowParams(lights.GetDirectionalLight().shadow,
                             "Far / extent",
                             "Resolution is live for directional shadow-map generation.",
                             /*dirNote=*/true);
            ImGui::Spacing();
        }
    }

    auto &pls = lights.GetPointLights();
    for (std::size_t i = 0; i < pls.size(); ++i) {
        ImGui::PushID(static_cast<int>(i));
        char hdr[48];
        std::snprintf(hdr, sizeof(hdr), "Point Shadow %zu##psh", i + 1);
        if (SectionHeader(hdr, /*defaultOpen=*/false)) {
            DrawShadowParams(pls[i].shadow, "Far plane",
                             "Point-light shadow params.", /*dirNote=*/false);
            ImGui::Spacing();
        }
        ImGui::PopID();
    }

    if (SectionHeader("Cascade Preview", /*defaultOpen=*/false)) {
        if (stats.shadowMapPreviewAvailable) {
            ImGui::Text("Res/cascade: %u x %u",
                        stats.shadowMapWidth, stats.shadowMapHeight);

            const float pw = (kSidebarWidth - 40.0f) * 0.5f;
            for (std::size_t i = 0; i < stats.cascadePreviewTextureIds.size(); ++i) {
                const uint32_t tex = stats.cascadePreviewTextureIds[i];
                const float sn = (i == 0) ? 0.f : stats.cascadeSplitDistances[i - 1];
                const float sf = stats.cascadeSplitDistances[i];

                ImGui::BeginGroup();
                ImGui::TextColored(Pal::TextDim, "C%zu [%.0f–%.0f]", i, sn, sf);
                if (tex)
                    ImGui::Image(reinterpret_cast<ImTextureID>(
                                     static_cast<intptr_t>(tex)),
                                 ImVec2(pw, pw), {0, 1}, {1, 0});
                else {
                    ImGui::Dummy({pw, pw});
                    ImGui::TextColored(Pal::TextFaint, "no view");
                }
                ImGui::EndGroup();
                if ((i % 2) == 0) ImGui::SameLine(0, 4);
            }
        } else {
            ImGui::TextColored(Pal::TextFaint, "Cascade preview not available.");
        }
        ImGui::Spacing();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tab: Post-FX
// ─────────────────────────────────────────────────────────────────────────────
void RendererUI::DrawTabPostFX(Scene & /*scene*/, const RendererDebugStats &stats) {
    if (SectionHeader("Tone Mapping")) {
        ImGui::PushStyleColor(ImGuiCol_Text, Pal::TextMid);

        ImGui::Checkbox("Enabled##tonemap", &tonemapEnabled);

        if (tonemapEnabled) {
            static const char *kOps[] = {"Reinhard", "ACES", "Uncharted2"};
            ImGui::Combo("Operator", &tonemapOperator, kOps, IM_ARRAYSIZE(kOps));

            ImGui::SliderFloat("Exposure", &exposure, 0.1f, 5.0f, "%.2f");
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Multiplier applied to HDR color before tone mapping");
        }

        ImGui::PopStyleColor();
        ImGui::Spacing();
    }

    if (SectionHeader("Bloom")) {
        ImGui::PushStyleColor(ImGuiCol_Text, Pal::TextMid);

        ImGui::Checkbox("Enabled##bloom", &bloomEnabled);
        ImGui::SliderFloat("Strength##bloom", &bloomStrength, 0.0f, 3.0f, "%.2f");
        ImGui::SliderFloat("Threshold##bloom", &bloomThreshold, 0.0f, 2.0f, "%.2f");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Luminance threshold for bloom extraction.");

        ImGui::Checkbox("Soft threshold##bloom", &bloomSoftThreshold);
        ImGui::BeginDisabled(!bloomSoftThreshold);
        ImGui::SliderFloat("Soft Knee##bloom", &bloomSoftKnee, 0.0f, 1.0f, "%.2f");
        ImGui::EndDisabled();

        ImGui::SliderFloat("Radius##bloom", &bloomRadius, 0.1f, 5.0f, "%.2f");
        ImGui::SliderInt("Blur iterations##bloom", &bloomBlurIterations, 1, 10);

        const char *kDebugViews[] = {"Final", "HDR Only", "Bright-pass", "Blurred Bloom", "No Bloom"};
        ImGui::Combo("Debug View##postfx", &postFxDebugView, kDebugViews, IM_ARRAYSIZE(kDebugViews));

        ImGui::PopStyleColor();
        ImGui::Spacing();
    }

    if (SectionHeader("IBL Debug", /*defaultOpen=*/false)) {
        ImGui::PushStyleColor(ImGuiCol_Text, Pal::TextMid);
        ImGui::Text("IBL active (stats): %s", stats.iblAvailable ? "yes" : "no");

        int iblModeIndex = std::clamp(ToUniformValue(iblDebugMode),
                                      0,
                                      static_cast<int>(kIBLDebugModeLabels.size()) - 1);
        if (ImGui::Combo("Debug Mode##ibl",
                         &iblModeIndex,
                         kIBLDebugModeLabels.data(),
                         static_cast<int>(kIBLDebugModeLabels.size()))) {
            iblDebugMode = static_cast<IBLDebugMode>(iblModeIndex);
        }

        const float maxMip = stats.iblPrefilteredMipCount > 0
                                 ? static_cast<float>(stats.iblPrefilteredMipCount - 1)
                                 : 0.0f;
        iblDebugPrefilteredMip = std::clamp(iblDebugPrefilteredMip, 0.0f, maxMip);
        ImGui::BeginDisabled(stats.iblPrefilteredMipCount <= 1);
        ImGui::SliderFloat("Prefiltered Mip##ibl", &iblDebugPrefilteredMip, 0.0f, maxMip, "%.1f");
        ImGui::EndDisabled();

        ImGui::Separator();
        ImGui::Text("IBL intensity    : %.3f", stats.iblIntensity);
        ImGui::Text("Selected mip     : %.1f / %.1f", iblDebugPrefilteredMip, maxMip);

        auto DrawResourceInfo = [](const char *label,
                                   uint32_t textureId,
                                   uint32_t width,
                                   uint32_t height,
                                   uint32_t mipCount = 0) {
            const bool ready = textureId != 0;
            ImGui::Text("%s", label);
            ImGui::SameLine(165.0f);
            ImGui::TextColored(ready ? Pal::Green : Pal::TextFaint,
                               "%s", ready ? "ready" : "missing");
            ImGui::TextColored(Pal::TextFaint, "  id: %u", textureId);
            if (ready && width > 0 && height > 0)
                ImGui::TextColored(Pal::TextFaint, "  size: %u x %u", width, height);
            else
                ImGui::TextColored(Pal::TextFaint, "  size: unknown");
            if (mipCount > 0)
                ImGui::TextColored(Pal::TextFaint, "  mips: %u", mipCount);
        };

        auto DrawCubemapPreview = [](const char *label,
                                     const auto& faceTextureIds,
                                     bool available) {
            ImGui::Spacing();
            ImGui::TextColored(Pal::TextDim, "%s", label);
            if (!available) {
                ImGui::TextColored(Pal::TextFaint, "  preview unavailable");
                return;
            }

            static const char *kFaceLabels[] = {"+X", "-X", "+Y", "-Y", "+Z", "-Z"};
            const float gap = 4.0f;
            const float width = ImGui::GetContentRegionAvail().x;
            const float thumb = std::max(48.0f, (width - gap * 2.0f) / 3.0f);

            for (std::size_t face = 0; face < faceTextureIds.size(); ++face) {
                ImGui::BeginGroup();
                ImGui::TextColored(Pal::TextFaint, "%s", kFaceLabels[face]);
                const uint32_t previewTexture = faceTextureIds[face];
                if (previewTexture != 0) {
                    ImGui::Image(reinterpret_cast<ImTextureID>(static_cast<intptr_t>(previewTexture)),
                                 ImVec2(thumb, thumb),
                                 ImVec2(0, 1),
                                 ImVec2(1, 0));
                } else {
                    ImGui::Dummy(ImVec2(thumb, thumb));
                }
                ImGui::EndGroup();
                if ((face % 3) != 2)
                    ImGui::SameLine(0, gap);
            }
        };

        DrawResourceInfo("Source environment",
                         stats.iblSourceTextureId,
                         stats.iblSourceWidth,
                         stats.iblSourceHeight);
        DrawResourceInfo("Irradiance cubemap",
                         stats.iblIrradianceTextureId,
                         stats.iblIrradianceWidth,
                         stats.iblIrradianceHeight);
        DrawResourceInfo("Prefiltered cubemap",
                         stats.iblPrefilteredTextureId,
                         stats.iblPrefilteredWidth,
                         stats.iblPrefilteredHeight,
                         stats.iblPrefilteredMipCount);
        DrawResourceInfo("BRDF LUT",
                         stats.iblBrdfLutTextureId,
                         stats.iblBrdfLutWidth,
                         stats.iblBrdfLutHeight);

        ImGui::Separator();
        DrawCubemapPreview("Original environment cubemap",
                           stats.iblSourcePreviewTextureIds,
                           stats.iblSourceTextureId != 0);
        DrawCubemapPreview("Irradiance cubemap",
                           stats.iblIrradiancePreviewTextureIds,
                           stats.iblIrradianceTextureId != 0);
        DrawCubemapPreview("Prefiltered cubemap (selected mip)",
                           stats.iblPrefilteredPreviewTextureIds,
                           stats.iblPrefilteredTextureId != 0);

        if (stats.iblBrdfLutTextureId != 0) {
            ImGui::Spacing();
            ImGui::TextWrapped("BRDF integration LUT (RG = scale, bias).");
            const float pw = std::min(256.0f, kSidebarWidth - 40.0f);
            ImGui::Image(reinterpret_cast<ImTextureID>(static_cast<intptr_t>(stats.iblBrdfLutTextureId)),
                         ImVec2(pw, pw), ImVec2(0, 1), ImVec2(1, 0));
        } else {
            ImGui::TextColored(Pal::TextFaint, "No BRDF LUT in active probe.");
        }
        ImGui::PopStyleColor();
        ImGui::Spacing();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tab: Stats
// ─────────────────────────────────────────────────────────────────────────────
void RendererUI::DrawTabStats(Scene & /*scene*/, const RendererDebugStats &stats) {
    auto Row = [](const char *label, const char *fmt, ...) {
        ImGui::PushStyleColor(ImGuiCol_Text, Pal::TextDim);
        ImGui::TextUnformatted(label);
        ImGui::PopStyleColor();
        ImGui::SameLine(150);

        va_list args;
        va_start(args, fmt);
        ImGui::TextV(fmt, args);
        va_end(args);
    };

    if (SectionHeader("Frame / Timing")) {
        ImGui::Checkbox("Frustum culling", &frustumCullingEnabled);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Skip forward-pass draws for objects outside the camera frustum.\nShadow casters are unaffected.");

        ImGui::PushStyleColor(ImGuiCol_Text, Pal::TextMid);
        Row("FPS", "%.1f", stats.fps);
        Row("Frame time", "%.2f ms", stats.frameTimeMs);
        if (stats.currentCameraAvailable) {
            Row("Camera", "%.1f FOV, yaw %.1f", stats.cameraFov, stats.cameraYaw);
            Row("Camera pos", "%.1f, %.1f, %.1f",
                stats.cameraPositionX, stats.cameraPositionY, stats.cameraPositionZ);
        } else {
            Row("Camera", "none");
        }
        ImGui::PopStyleColor();
        ImGui::Spacing();
    }

    if (SectionHeader("Render Pass Timings")) {
        ImGui::PushStyleColor(ImGuiCol_Text, Pal::TextMid);
        bool anyTiming = false;
        for (const RendererPassTiming &timing : stats.passTimings) {
            if (!timing.available)
                continue;
            Row(timing.name, "%.2f ms", timing.milliseconds);
            anyTiming = true;
        }
        if (!anyTiming)
            Row("Passes", "not available");
        ImGui::PopStyleColor();
        ImGui::Spacing();
    }

    if (SectionHeader("Culling")) {
        ImGui::PushStyleColor(ImGuiCol_Text, Pal::TextMid);
        Row("Status", "%s", stats.cullingEnabled ? "enabled" : "disabled");
        Row("Submitted", "%u", stats.submittedRenderItemCount);
        Row("Visible", "%u", stats.visibleRenderItemCount);
        Row("Culled", "%u", stats.frustumCulledRenderItemCount);
        Row("Drawn", "%u", stats.processedRenderItemCount);
        Row("Queued", "%u", stats.queuedRenderItemCount);
        if (!stats.cullingDataAvailable)
            Row("Culling data", "no camera");
        ImGui::PopStyleColor();
        ImGui::Spacing();
    }

    if (SectionHeader("Draw / State Changes")) {
        ImGui::PushStyleColor(ImGuiCol_Text, Pal::TextMid);
        Row("Draw calls", "%u", stats.drawCallCount);
        Row("Approx tris", "%s", FormatCompact(stats.approxTriangleCount).c_str());
        Row("Program changes", "%u", stats.shaderProgramChangeCount);
        Row("Material changes", "%u", stats.materialChangeCount);
        Row("Texture binds", "%u", stats.textureBindingCount);
        ImGui::PopStyleColor();
        ImGui::Spacing();
    }

    if (SectionHeader("Resources")) {
        ImGui::PushStyleColor(ImGuiCol_Text, Pal::TextMid);
        Row("Scene meshes", "%u", stats.meshCount);
        Row("Scene materials", "%u", stats.materialCount);
        Row("Scene shaders", "%u", stats.shaderProgramCount);
        Row("Lights", "%u total", stats.totalLightCount);
        Row("Directional", "%u", stats.directionalLightCount);
        Row("Point / Spot", "%u / %u", stats.pointLightCount, stats.spotLightCount);
        Row("Active cameras", "%u", stats.activeCameraCount);
        ImGui::Separator();
        Row("Cached shaders", "%u", stats.cachedShaderProgramCount);
        Row("Cached meshes", "%u", stats.cachedMeshCount);
        Row("Cached materials", "%u", stats.cachedMaterialCount);
        Row("Cached textures", "%u", stats.textureCount);
        Row("2D / cubemap", "%u / %u", stats.cachedTexture2DCount, stats.cachedCubemapCount);
        ImGui::PopStyleColor();
        ImGui::Spacing();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Terrain tab — delegates to scene so RendererUI stays scene-agnostic
// ─────────────────────────────────────────────────────────────────────────────
void RendererUI::DrawTabTerrain(Scene &scene) {
    scene.OnTerrainTabUI();
}

static std::vector<float> EchogramPeaks(const std::vector<OctaveBands> &bins, std::size_t band) {
    constexpr std::size_t maxColumns = 256;
    const std::size_t columns = std::min(bins.size(), maxColumns);
    std::vector<float> peaks(columns, 0.0f);
    for (std::size_t i = 0; i < columns; ++i) {
        const std::size_t first = i * bins.size() / columns;
        const std::size_t end = (i + 1) * bins.size() / columns;
        for (std::size_t bin = first; bin < end; ++bin)
            peaks[i] = std::max(peaks[i], bins[bin][band]);
    }
    return peaks;
}

void RendererUI::DrawAcousticGraphs(AcousticRun &run) {
    ImGui::Indent();
    const float width = ImGui::GetContentRegionAvail().x - ImGui::GetStyle().IndentSpacing;
    ImGui::PushItemWidth(width * 0.5f);
    const char *graphKinds[] = {"Echogram", "Schroeder decay"};
    ImGui::Combo("Graph", &m_acousticGraphKind, graphKinds, IM_ARRAYSIZE(graphKinds));

    char bandLabel[16];
    std::snprintf(bandLabel, sizeof(bandLabel), "%.0f Hz", OctaveBands::Frequencies[m_acousticGraphBand]);
    if (ImGui::BeginCombo("Octave band", bandLabel)) {
        for (int i = 0; i < static_cast<int>(OctaveBands::Count); ++i) {
            std::snprintf(bandLabel, sizeof(bandLabel), "%.0f Hz", OctaveBands::Frequencies[i]);
            if (ImGui::Selectable(bandLabel, m_acousticGraphBand == i))
                m_acousticGraphBand = i;
        }
        ImGui::EndCombo();
    }
    ImGui::PopItemWidth();

    const auto band = static_cast<std::size_t>(m_acousticGraphBand);
    if (m_acousticGraphKind == 0) {
        if (run.plottedEchoBand != m_acousticGraphBand) {
            run.leftEchoPeaks = EchogramPeaks(run.echogram.left, band);
            run.rightEchoPeaks = EchogramPeaks(run.echogram.right, band);
            run.plottedEchoBand = m_acousticGraphBand;
        }
        float peak = 0.0f;
        for (const auto *peaks : {&run.leftEchoPeaks, &run.rightEchoPeaks})
            for (const float energy : *peaks)
                peak = std::max(peak, energy);
        if (peak > 0.0f) {
            ImGui::TextWrapped("Energy per bin. Shared scale 0 to %.3g.", peak);
            ImGui::TextDisabled("Time from emission (s)");
            const auto plot = [&](const char *ear, const char *id, const std::vector<float> &peaks) {
                ImGui::Text("%s    0 to %.3f s", ear, run.echogram.DurationSec());
                ImGui::PlotHistogram(id, peaks.data(), static_cast<int>(peaks.size()), 0, nullptr,
                                     0.0f, peak, ImVec2(width, 84));
            };
            plot("Left ear", "##leftEcho", run.leftEchoPeaks);
            plot("Right ear", "##rightEcho", run.rightEchoPeaks);
        } else {
            ImGui::TextDisabled("No energy in this band.");
        }
    } else {
        ImGui::TextWrapped("Level from 0 to -80 dB. Time starts at each ear's first arrival.");
        const auto plot = [&](const char *ear, const char *id, acoustics::BandDecay &decay) {
            if (!decay.IsValid() || decay.levelDb.size() < 2) {
                ImGui::TextDisabled("%s    No decay curve", ear);
                return;
            }
            const double seconds = static_cast<double>(decay.levelDb.size() - 1) / run.metrics.sampleRate;
            ImGui::Text("%s    0 to %.3f s", ear, seconds);
            const auto level = [](void *data, int index) {
                const auto &values = *static_cast<const std::vector<double> *>(data);
                return static_cast<float>(std::clamp(values[index], -80.0, 0.0));
            };
            ImGui::PlotLines(id, level, &decay.levelDb, static_cast<int>(decay.levelDb.size()), 0, nullptr,
                             -80.0f, 0.0f, ImVec2(width, 84));
        };
        plot("Left ear", "##leftDecay", run.metrics.left.bands[band].decay);
        plot("Right ear", "##rightDecay", run.metrics.right.bands[band].decay);
    }
    ImGui::Unindent();
    ImGui::Spacing();
}

void RendererUI::DrawAcousticMetrics(const AcousticRun &run) {
    ImGui::Indent();
    const float width = ImGui::GetContentRegionAvail().x - ImGui::GetStyle().IndentSpacing;
    const char *ears[] = {"Left ear", "Right ear"};
    const char *fits[] = {"T30 (-5 to -35 dB)", "T20 (-5 to -25 dB)"};
    ImGui::PushItemWidth(width * 0.65f);
    ImGui::Combo("Ear", &m_acousticMetricEar, ears, IM_ARRAYSIZE(ears));
    ImGui::Combo("RT60 fit", &m_acousticRt60Fit, fits, IM_ARRAYSIZE(fits));
    ImGui::PopItemWidth();

    ImGui::TextWrapped("RT60 and EDT (s), C80 and C50 (dB), D50 (%%).");
    const auto cell = [](const acoustics::ScalarMetric &metric, const char *format) {
        ImGui::TableNextColumn();
        if (metric.IsValid()) {
            ImGui::Text(format, metric.value);
        } else {
            ImGui::TextDisabled("--");
            ImGui::SetItemTooltip("%s", acoustics::MetricStatusName(metric.status));
        }
    };
    const auto &channel = m_acousticMetricEar == 0 ? run.metrics.left : run.metrics.right;
    constexpr ImGuiTableFlags flags = ImGuiTableFlags_SizingStretchSame | ImGuiTableFlags_RowBg |
                                      ImGuiTableFlags_BordersInnerH;
    ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(4.0f, 4.0f));
    if (ImGui::BeginTable("##roomMetrics", 6, flags, ImVec2(width, 0))) {
        for (const char *label : {"Hz", "RT60", "EDT", "C80", "C50", "D50"})
            ImGui::TableSetupColumn(label);
        ImGui::TableHeadersRow();
        for (std::size_t band = 0; band < OctaveBands::Count; ++band) {
            const auto &values = channel.bands[band];
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%.0f", OctaveBands::Frequencies[band]);
            cell(m_acousticRt60Fit == 0 ? values.rt60FromT30 : values.rt60FromT20, "%.2f");
            cell(values.edt, "%.2f");
            cell(values.c80Db, "%.1f");
            cell(values.c50Db, "%.1f");
            cell(values.d50Percent, "%.1f");
        }
        ImGui::EndTable();
    }
    ImGui::Spacing();
    if (run.enclosedVolume > 0.0f) {
        ImGui::TextWrapped("Reference RT60 (s) from surface absorption for %.1f m³.", run.enclosedVolume);
        if (ImGui::BeginTable("##rt60References", 3, flags, ImVec2(width, 0))) {
            ImGui::TableSetupColumn("Hz");
            ImGui::TableSetupColumn("Sabine");
            ImGui::TableSetupColumn("Eyring");
            ImGui::TableHeadersRow();
            for (std::size_t band = 0; band < OctaveBands::Count; ++band) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%.0f", OctaveBands::Frequencies[band]);
                for (float seconds : {run.sabineRt60[band], run.eyringRt60[band]}) {
                    ImGui::TableNextColumn();
                    if (seconds > 0.0f && std::isfinite(seconds))
                        ImGui::Text("%.3f", seconds);
                    else
                        ImGui::TextDisabled("--");
                }
            }
            ImGui::EndTable();
        }
    } else {
        ImGui::TextWrapped("Sabine and Eyring references require a known enclosed room volume.");
    }
    ImGui::PopStyleVar();
    ImGui::Unindent();
    ImGui::Spacing();
}

void RendererUI::ResetAcousticAudio() {
    acoustics::AudioEngine::StopPlayback();
    m_acousticWet.reset();
    m_acousticAudioMessage.clear();
    m_acousticLoadedDry.reset();
}

bool RendererUI::EnsureDryAudioLoaded() {
    if (m_acousticLoadedDry)
        return true;
    if (m_acousticDryPath.front() == '\0')
        return false;
    try {
        acoustics::MonoAudio dry;
        const auto path = std::filesystem::path(reinterpret_cast<const char8_t *>(m_acousticDryPath.data()));
        if (std::filesystem::exists(std::filesystem::symlink_status(path))) {
            const auto loaded = acoustics::AudioEngine::ReadMonoWav(path, dry);
            if (loaded.ok) {
                m_acousticLoadedDry = std::move(dry);
                return true;
            } else {
                m_acousticAudioMessage = loaded.message;
            }
        }
    } catch (const std::exception &error) {
        m_acousticAudioMessage = error.what();
    }
    return false;
}

void RendererUI::ExecuteAcousticSimulation(Scene &scene, const FrameSubmission &frame, bool updateAudio) {
    if (!frame.camera) {
        m_acousticError = "The scene has no listener camera.";
        return;
    }
    const int enabledLights = frame.lights.ActiveLightCount();
    if (enabledLights == 0) {
        m_acousticError = "Enable a scene light to use it as a sound source.";
        return;
    }

    if (m_bvhCachedScene != &scene || !m_cachedAcousticBvh) {
        m_cachedAcousticGeometry = std::make_unique<AcousticSceneGeometry>();
        m_cachedAcousticGeometry->ExtractFromScene(scene);
        if (m_cachedAcousticGeometry->IsEmpty()) {
            m_acousticError = "This scene has no CPU mesh triangles for acoustic simulation.";
            return;
        }
        m_cachedAcousticBvh = std::make_unique<BVH>();
        m_cachedAcousticBvh->Build(*m_cachedAcousticGeometry);
        m_bvhCachedScene = &scene;
    }

    const AABB &bounds = m_cachedAcousticGeometry->GetBounds();
    const glm::vec3 center = (bounds.min + bounds.max) * 0.5f;
    const float radius = glm::length(bounds.max - bounds.min) * 0.5f;
    auto sources = acoustics::FromLights(frame.lights, center, radius);
    std::erase_if(sources, [](const acoustics::Source &source) {
        const float power = source.power.Average();
        return !std::isfinite(power) || power <= 0.0f;
    });
    if (sources.empty()) {
        m_acousticError = "Set a positive intensity on an enabled scene light.";
        return;
    }

    for (auto &source : sources) {
        if (source.pattern == acoustics::Source::Pattern::Parallel)
            source.position -= source.direction * (source.emitterRadius + m_acousticSettings.rayOffset);
    }

    const auto &materials = AcousticMaterialLibrary::Instance();
    const acoustics::Simulator simulator(*m_cachedAcousticBvh, materials);
    auto runSettings = m_acousticSettings;
    runSettings.SetAirConditions(m_acousticAir);

    AcousticRun run{m_acousticReceiver, acoustics::Echogram(runSettings.maxDuration), {},
                    m_cachedAcousticGeometry->GetTriangleCount(), m_acousticAir, runSettings.speedOfSound};
    run.stats = simulator.Run(sources, run.listener, run.echogram, runSettings, &run.debugSegments);
    run.metrics = acoustics::RoomMetrics::Compute(run.echogram);
    run.enclosedVolume = scene.GetEnclosedAcousticVolume();
    if (run.enclosedVolume > 0.0f) {
        m_cachedAcousticGeometry->SetExplicitVolume(run.enclosedVolume);
        for (std::size_t band = 0; band < OctaveBands::Count; ++band) {
            run.sabineRt60[band] = m_cachedAcousticGeometry->CalculateSabineRT60(band, materials);
            run.eyringRt60[band] = m_cachedAcousticGeometry->CalculateEyringRT60(band, materials);
        }
    }

    m_acousticRun = std::move(run);
    m_acousticError.clear();

    if (updateAudio) {
        EnsureDryAudioLoaded();
        if (m_acousticLoadedDry)
            PlayAcousticResponse(*m_acousticLoadedDry);
    }
}

void RendererUI::PlayAcousticResponse(const acoustics::MonoAudio &dry) {
    if (!m_acousticRun) {
        m_acousticAudioMessage = m_acousticError;
        return;
    }
    if (m_acousticRun->stats.earHits == 0) {
        m_acousticAudioMessage = "No rays reached either ear. Increase the ray count or adjust the listener.";
        return;
    }
    const auto impulse = acoustics::AudioEngine::SynthesizeImpulseResponse(m_acousticRun->echogram);
    if (!impulse.ok) {
        m_acousticAudioMessage = impulse.message;
        return;
    }
    auto wet = acoustics::AudioEngine::Convolve(dry, impulse.audio);
    if (!wet.ok) {
        m_acousticAudioMessage = wet.message;
        return;
    }
    m_acousticWet = std::move(wet.audio);
    m_acousticAudioMessage = acoustics::AudioEngine::PlayStereo(*m_acousticWet).message;
}

void RendererUI::DrawAcousticDebug(const Scene &scene) {
    if (m_acousticScene != &scene)
        return;

    const bool drawRays = m_showAcousticRays && m_acousticRun && !m_acousticRun->debugSegments.empty();
    if (!drawRays && !m_showAcousticReceiver)
        return;

    if (!m_acousticDebugShader) {
        m_acousticDebugShader = std::make_unique<ShaderProgram>(
            "assets/shaders/basic.vert", "assets/shaders/acoustic_debug.frag");
    }
    if (!m_acousticDebugShader->IsValid())
        return;

    const AcousticDebugGLState previousState;
    if (drawRays && !m_acousticRun->debugRayMesh) {
        constexpr glm::vec3 direct{0.10f, 1.0f, 0.25f};
        constexpr glm::vec3 firstReflection{1.0f, 0.83f, 0.06f};
        constexpr glm::vec3 laterReflection{1.0f, 0.32f, 0.05f};
        PrimitiveMeshData rays;
        rays.vertices.reserve(2 * m_acousticRun->debugSegments.size());
        for (const auto &segment : m_acousticRun->debugSegments) {
            if (segment.bounceCount > m_acousticDisplayedBounces)
                continue;
            const float order = static_cast<float>(std::min(segment.bounceCount, 5u));
            const glm::vec3 color = segment.bounceCount == 0 ? direct :
                glm::mix(firstReflection, laterReflection, (order - 1.0f) / 4.0f);
            rays.vertices.push_back({segment.start, color});
            rays.vertices.push_back({segment.end, color});
        }
        m_acousticRun->debugRayMesh = std::make_unique<MeshBuffer>(rays.CreateMeshBuffer());
    }
    if (m_showAcousticReceiver && !m_acousticDebugSphere) {
        const VertexArray uploadVao;
        uploadVao.Bind();
        m_acousticDebugSphere = std::make_unique<MeshBuffer>(
            GenerateSphere(1.0f, 12, {.colorMode = ColorMode::Solid}).CreateMeshBuffer());
    }

    auto &shader = *m_acousticDebugShader;
    shader.Bind();
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glDisable(GL_BLEND);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    const auto drawSphere = [&](const glm::vec3 &center, float radius, const glm::vec4 &color) {
        const auto model = glm::scale(glm::translate(glm::mat4(1.0f), center), glm::vec3(radius));
        shader.SetUniform("u_Model", model);
        shader.SetUniform("u_Tint", color);
        m_acousticDebugSphere->Draw();
    };

    const auto &listener = m_acousticFollowCamera ? m_acousticReceiver :
                           (m_acousticRun ? m_acousticRun->listener : m_acousticReceiver);

    if (m_showAcousticReceiver) {
        glDepthMask(GL_TRUE);
        drawSphere(listener.LeftEar(), listener.earRadius, {0.12f, 0.37f, 1.0f, 1.0f});
        drawSphere(listener.RightEar(), listener.earRadius, {1.0f, 0.08f, 0.12f, 1.0f});
    }
    glDepthMask(GL_FALSE);
    if (drawRays && m_acousticRun && m_acousticRun->debugRayMesh) {
        shader.SetUniform("u_Model", glm::mat4(1.0f));
        shader.SetUniform("u_Tint", glm::vec4(1.0f));
        m_acousticRun->debugRayMesh->Draw(GL_LINES);
    }
    if (m_showAcousticReceiver) {
        glEnable(GL_BLEND);
        glBlendEquation(GL_FUNC_ADD);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        drawSphere(listener.headCenter, listener.headRadius, {0.80f, 0.87f, 1.0f, 0.32f});
    }
}

void RendererUI::DrawTabAcoustics(Scene &scene, const FrameSubmission &frame) {
    constexpr float centimetersPerMeter = 100.0f;
    constexpr ImGuiSliderFlags clamp = ImGuiSliderFlags_AlwaysClamp;
    const int enabledLights = frame.lights.ActiveLightCount();
    auto &listener = m_acousticReceiver;
    if (frame.camera && m_acousticFollowCamera)
        listener.FollowCamera(*frame.camera);

    if (SectionHeader("Scene Inputs")) {
        ImGui::Indent();
        ImGui::PushStyleColor(ImGuiCol_Text, Pal::TextMid);
        ImGui::TextWrapped("Scene: %s", scene.GetName().c_str());
        ImGui::Text("Enabled lights: %d", enabledLights);
        if (frame.camera) {
            const auto &position = listener.headCenter;
            ImGui::Text("Listener: %.2f, %.2f, %.2f m", position.x, position.y, position.z);
            ImGui::Text("Facing:   %.2f, %.2f, %.2f", listener.forward.x, listener.forward.y, listener.forward.z);
        } else {
            ImGui::TextColored(Pal::TextDim, "No camera for the listener.");
        }
        ImGui::PopStyleColor();
        ImGui::Unindent();
        ImGui::Spacing();
    }

    if (SectionHeader("Binaural Receiver")) {
        ImGui::Indent();
        const float sectionWidth = ImGui::GetContentRegionAvail().x - ImGui::GetStyle().IndentSpacing;
        ImGui::PushItemWidth(sectionWidth * 0.5f);

        ImGui::Checkbox("Follow camera (Live tracking)", &m_acousticFollowCamera);
        if (!m_acousticFollowCamera && frame.camera) {
            ImGui::SameLine();
            if (ImGui::Button("Snap to Camera"))
                listener.FollowCamera(*frame.camera);
        }

        float spacingCm = listener.earSpacing * centimetersPerMeter;
        const bool spacingChanged = ImGui::SliderFloat("Ear spacing (cm)", &spacingCm, 10.0f, 25.0f,
                                                      "%.1f", clamp);
        const float maxRadiusCm = spacingCm * 0.25f;
        float radiusCm = std::min(listener.earRadius * centimetersPerMeter, maxRadiusCm);
        const bool radiusChanged = ImGui::SliderFloat("Ear radius (cm)", &radiusCm, 0.5f, maxRadiusCm,
                                                     "%.2f", clamp);
        if (spacingChanged || radiusChanged) {
            listener.earSpacing = spacingCm / centimetersPerMeter;
            listener.earRadius = std::min(radiusCm / centimetersPerMeter, listener.earSpacing * 0.25f);
            listener.headRadius = listener.earSpacing * 0.5f - listener.earRadius;
        }

        ImGui::PopItemWidth();
        ImGui::Checkbox("Head shadow (ILD)", &listener.headShadow);
        ImGui::TextColored(Pal::TextMid, "Head radius: %.2f cm", listener.headRadius * centimetersPerMeter);
        ImGui::TextColored(Pal::Accent, "Live Listener: (%.2f, %.2f, %.2f)",
                           listener.headCenter.x, listener.headCenter.y, listener.headCenter.z);
        ImGui::Unindent();
        ImGui::Spacing();
    }

    auto &settings = m_acousticSettings;

    if (SectionHeader("Simulation")) {
        ImGui::Indent();
        const float buttonWidth = ImGui::GetContentRegionAvail().x - ImGui::GetStyle().IndentSpacing;
        constexpr uint32_t minRays = 1000;
        constexpr uint32_t maxRays = 250000;
        ImGui::PushItemWidth(buttonWidth * 0.5f);
        ImGui::SliderScalar("Rays per source", ImGuiDataType_U32, &settings.raysPerSource,
                            &minRays, &maxRays, "%u", clamp | ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Max duration (s)", &settings.maxDuration, 0.1f, 5.0f, "%.2f", clamp);
        float thresholdDb = 10.0f * std::log10(settings.minEnergyFraction);
        if (ImGui::SliderFloat("Energy threshold (dB)", &thresholdDb, -90.0f, -20.0f, "%.0f", clamp)) {
            settings.minEnergyFraction = std::pow(10.0f, thresholdDb / 10.0f);
        }
        ImGui::SliderFloat("Temperature (C)", &m_acousticAir.temperatureC,
                           acoustics::AirConditions::MinTemperatureC, acoustics::AirConditions::MaxTemperatureC,
                           "%.1f", clamp);
        ImGui::SliderFloat("Relative humidity (%)", &m_acousticAir.relativeHumidityPercent,
                           acoustics::AirConditions::MinRelativeHumidityPercent,
                           acoustics::AirConditions::MaxRelativeHumidityPercent, "%.0f", clamp);
        ImGui::PopItemWidth();
        ImGui::TextColored(Pal::TextMid, "Air pressure: 101.325 kPa (fixed)");

        if (ImGui::Button("Run Simulation", ImVec2(buttonWidth, 0))) {
            try {
                ExecuteAcousticSimulation(scene, frame, false);
            } catch (const std::exception &error) {
                m_acousticError = error.what();
            }
        }
        if (!m_acousticError.empty()) {
            ImGui::PushStyleColor(ImGuiCol_Text, Pal::Orange);
            ImGui::TextWrapped("%s", m_acousticError.c_str());
            ImGui::PopStyleColor();
        }
        ImGui::Unindent();
        ImGui::Spacing();
    }

    if (SectionHeader("3D Debug View")) {
        ImGui::Indent();
        ImGui::Checkbox("Show ray paths", &m_showAcousticRays);
        ImGui::SameLine();
        ImGui::Checkbox("Show listener", &m_showAcousticReceiver);
        if (m_acousticRun) {
            constexpr uint32_t directOnly = 0;
            ImGui::BeginDisabled(!m_showAcousticRays);
            ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x * 0.5f);
            if (ImGui::SliderScalar("Displayed reflections", ImGuiDataType_U32, &m_acousticDisplayedBounces,
                                    &directOnly, &settings.maxBounces, "%u", clamp))
                m_acousticRun->debugRayMesh.reset();
            ImGui::EndDisabled();
            ImGui::TextWrapped("Green: direct segments. Yellow to orange: reflected segments.");
            ImGui::TextWrapped("Blue sphere: Left ear | Red sphere: Right ear | Center: Head");
            ImGui::TextColored(Pal::TextMid, "Tip: Press C or F3 for Third-Person camera to orbit and inspect the moving listener!");
        } else {
            ImGui::TextWrapped("Blue sphere: Left ear | Red sphere: Right ear");
            ImGui::TextColored(Pal::TextMid, "Tip: Press C or F3 for Third-Person camera to inspect the listener!");
        }
        ImGui::Unindent();
        ImGui::Spacing();
    }

    if (SectionHeader("Auralization")) {
        ImGui::Indent();
        const float width = ImGui::GetContentRegionAvail().x - ImGui::GetStyle().IndentSpacing;
        ImGui::TextUnformatted("Dry WAV path (mono PCM16 or float32)");
        ImGui::SetNextItemWidth(width);
        if (ImGui::InputText("##dryWavPath", m_acousticDryPath.data(), m_acousticDryPath.size()))
            ResetAcousticAudio();
        ImGui::BeginDisabled(m_acousticDryPath.front() == '\0');
        const bool pressedP = ImGui::IsKeyPressed(ImGuiKey_P, false) && !ImGui::GetIO().WantTextInput;
        if (ImGui::Button("Simulate & Play (P)") || pressedP) {
            try {
                if (EnsureDryAudioLoaded() && m_acousticLoadedDry) {
                    ExecuteAcousticSimulation(scene, frame, false);
                    PlayAcousticResponse(*m_acousticLoadedDry);
                }
            } catch (const std::exception &error) {
                m_acousticAudioMessage = error.what();
            }
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::BeginDisabled(!m_acousticLoadedDry && !m_acousticWet);
        if (ImGui::Button("Replay (Current Pos)")) {
            try {
                if (EnsureDryAudioLoaded() && m_acousticLoadedDry) {
                    ExecuteAcousticSimulation(scene, frame, false);
                    PlayAcousticResponse(*m_acousticLoadedDry);
                } else if (m_acousticWet) {
                    m_acousticAudioMessage = acoustics::AudioEngine::PlayStereo(*m_acousticWet).message;
                }
            } catch (const std::exception &error) {
                m_acousticAudioMessage = error.what();
            }
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        if (ImGui::Button("Stop Preview"))
            acoustics::AudioEngine::StopPlayback();

        ImGui::TextUnformatted("Stereo WAV path (save or play)");
        ImGui::SetNextItemWidth(width);
        ImGui::InputText("##stereoWavPath", m_acousticStereoPath, sizeof(m_acousticStereoPath));
        const bool hasStereoPath = m_acousticStereoPath[0] != '\0';
        ImGui::BeginDisabled(!m_acousticWet || !hasStereoPath);
        if (ImGui::Button("Save to WAV")) {
            try {
                const auto path = std::filesystem::path(reinterpret_cast<const char8_t *>(m_acousticStereoPath));
                if (std::filesystem::exists(std::filesystem::symlink_status(path))) {
                    m_acousticAudioMessage = "This path already exists. Choose a new output file.";
                } else {
                    m_acousticAudioMessage = acoustics::AudioEngine::WriteStereoWav(path, *m_acousticWet).message;
                }
            } catch (const std::exception &error) {
                m_acousticAudioMessage = error.what();
            }
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::BeginDisabled(!hasStereoPath);
        if (ImGui::Button("Play Saved WAV")) {
            try {
                const auto path = std::filesystem::path(reinterpret_cast<const char8_t *>(m_acousticStereoPath));
                m_acousticAudioMessage = acoustics::AudioEngine::PlayWavFile(path).message;
            } catch (const std::exception &error) {
                m_acousticAudioMessage = error.what();
            }
        }
        ImGui::EndDisabled();
        if (!m_acousticAudioMessage.empty())
            ImGui::TextWrapped("%s", m_acousticAudioMessage.c_str());
        ImGui::Unindent();
        ImGui::Spacing();
    }

    if (m_acousticRun && SectionHeader("Acoustic Graphs"))
        DrawAcousticGraphs(*m_acousticRun);

    if (m_acousticRun && SectionHeader("Room Metrics"))
        DrawAcousticMetrics(*m_acousticRun);

    if (m_acousticRun && SectionHeader("Last Simulation")) {
        const AcousticRun &run = *m_acousticRun;
        const auto &position = run.listener.headCenter;
        ImGui::Indent();
        ImGui::PushStyleColor(ImGuiCol_Text, Pal::TextMid);
        ImGui::Text("Triangles: %s", FormatCompact(run.triangleCount).c_str());
        ImGui::Text("Rays: %s    Ear hits: %s",
                    FormatCompact(run.stats.emitted).c_str(), FormatCompact(run.stats.earHits).c_str());
        ImGui::Text("Segments: %s", FormatCompact(run.stats.segments).c_str());
        ImGui::Text("Echogram: %s bins at %.0f Hz",
                    FormatCompact(run.echogram.BinCount()).c_str(), run.echogram.sampleRate);
        ImGui::Text("Listener at run: %.2f, %.2f, %.2f m", position.x, position.y, position.z);
        ImGui::Text("Ear spacing / radius: %.1f / %.2f cm",
                    run.listener.earSpacing * centimetersPerMeter, run.listener.earRadius * centimetersPerMeter);
        ImGui::Text("Head shadow (ILD): %s", run.listener.headShadow ? "On" : "Off");
        ImGui::Text("Air at run: %.1f C, %.0f%% RH", run.air.temperatureC, run.air.relativeHumidityPercent);
        ImGui::Text("Sound speed: %.1f m/s", run.speedOfSound);
        ImGui::PopStyleColor();
        ImGui::Unindent();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
void RendererUI::DrawHelpWindow(int fbW, int fbH) {
    constexpr float w = 380.0f;
    constexpr float h = 500.0f;
    ImGui::SetNextWindowPos({(fbW - w) * 0.5f, (fbH - h) * 0.5f}, ImGuiCond_Always);
    ImGui::SetNextWindowSize({w, h}, ImGuiCond_Always);

    // NoScrollbar removed — let ImGui show the bar only when content overflows,
    // matching the sidebar content child behaviour.
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoCollapse |
                             ImGuiWindowFlags_NoResize |
                             ImGuiWindowFlags_NoSavedSettings |
                             ImGuiWindowFlags_NoScrollbar |
                             ImGuiWindowFlags_NoScrollWithMouse; // scroll via keyboard/drag only

    ImGui::PushStyleColor(ImGuiCol_TitleBgActive, Pal::Accent);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, Pal::Bg1); // ← translucent, matches sidebar
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(20, 20));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, kRounding); // ← rounded corners


    if (ImGui::Begin("Keyboard Shortcuts", &showHelpWindow, flags)) {
        auto KeyRow = [](const char *key, const char *desc) {
            ImGui::TextColored(Pal::Accent, "%-12s", key);
            ImGui::SameLine(100);
            ImGui::TextColored(Pal::TextMid, "%s", desc);
            ImGui::Spacing();
        };

        ImGui::TextColored(Pal::TextDim, "MOVEMENT");
        ImGui::Separator();
        KeyRow("W/A/S/D", "Move");
        KeyRow("SPACE/CTRL", "Move up/down");
        KeyRow("SHIFT", "Increase speed");

        ImGui::Spacing();
        ImGui::Spacing();

        ImGui::TextColored(Pal::TextDim, "CAMERA");
        ImGui::Separator();
        KeyRow("TAB", "Toggle rotation mode");
        KeyRow("C", "Cycle camera modes");
        KeyRow("F1", "Free-fly");
        KeyRow("F2", "First-person");
        KeyRow("F3", "Third-person");

        ImGui::Spacing();
        ImGui::Spacing();

        ImGui::TextColored(Pal::TextDim, "INTERFACE");
        ImGui::Separator();
        KeyRow("X", "Toggle inspector");
        KeyRow("1-9", "Switch scenes");
        KeyRow("Z", "Wireframe mode");
        KeyRow("N", "Toggle normal map");
        KeyRow("K", "Toggle skybox");
        KeyRow("H", "Help window");

        ImGui::Dummy(ImVec2(0.0f, ImGui::GetContentRegionAvail().y - 30.0f));
        if (ImGui::Button("Close", ImVec2(-1, 30)))
            showHelpWindow = false;
    }
    ImGui::End();
    ImGui::PopStyleVar(2); // ← WindowRounding + WindowPadding
    ImGui::PopStyleColor(2); // ← WindowBg + TitleBgActive
}
