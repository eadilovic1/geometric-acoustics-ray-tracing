#include "ShoeboxAcousticsScene.h"

#include "acoustics/AcousticGeometry.h"
#include "assets/AssetImporter.h"
#include "core/Material.h"
#include "core/Mesh.h"
#include "core/Primitives.h"
#include "core/ShaderProgram.h"
#include "scene/LightBuilder.h"

#include <algorithm>
#include <cmath>
#include <utility>
#include <spdlog/spdlog.h>

namespace
{
    constexpr glm::vec3 kRoomDimensions{10.0f, 3.0f, 8.0f};
    constexpr glm::vec3 kRoomCenter{0.0f, kRoomDimensions.y * 0.5f, 0.0f};
    constexpr glm::vec3 kListenerPosition{0.0f, 1.5f, 0.5f};
    constexpr char kSourceName[] = "Shoebox Source";
    constexpr glm::vec3 kSourceColor{1.0f, 0.72f, 0.22f};

    struct SurfaceStyle
    {
        const char* name;
        glm::vec3 color;
    };

    constexpr std::array<SurfaceStyle, 3> kSurfaces{{
        {"Floor", {0.32f, 0.23f, 0.17f}},
        {"Ceiling", {0.72f, 0.73f, 0.70f}},
        {"Walls", {0.37f, 0.49f, 0.54f}},
    }};

    MeshData BuildRoomMesh()
    {
        AcousticSceneGeometry room;
        // The temporary IDs select the visual style for each generated surface.
        room.AddShoeboxRoom(kRoomDimensions, 0, 1, 2, kRoomCenter);

        MeshData data;
        data.name = "Shoebox Room";
        for (uint32_t surface = 0; surface < kSurfaces.size(); ++surface)
        {
            SubMesh part;
            part.name = kSurfaces[surface].name;
            part.materialIndex = surface;
            part.indexByteOffset = static_cast<uint32_t>(data.indices.size() * sizeof(uint32_t));

            for (const auto& triangle : room.GetTriangles())
            {
                if (triangle.materialId != surface)
                    continue;

                const glm::vec3 tangent = std::abs(triangle.normal.x) > 0.5f
                    ? glm::vec3(0.0f, 0.0f, 1.0f)
                    : glm::vec3(1.0f, 0.0f, 0.0f);
                const glm::vec3 bitangent = glm::cross(triangle.normal, tangent);
                for (const auto& position : {triangle.v0, triangle.v1, triangle.v2})
                {
                    data.indices.push_back(static_cast<uint32_t>(data.vertices.size()));
                    data.vertices.push_back({
                        position,
                        triangle.normal,
                        {glm::dot(position, tangent), glm::dot(position, bitangent)},
                        glm::vec4(tangent, 1.0f),
                    });
                    ++part.indexCount;
                }
            }
            data.submeshes.push_back(part);
        }
        return data;
    }
}

bool ShoeboxAcousticsScene::Setup()
{
    if (m_roomMesh)
        return true;

    auto roomShader = AssetImporter::LoadShader("assets/shaders/mesh.vert", "assets/shaders/mesh.frag");
    auto markerShader = AssetImporter::LoadShader("assets/shaders/basic.vert", "assets/shaders/basic.frag");
    if (!roomShader || !roomShader->IsValid() || !markerShader || !markerShader->IsValid())
    {
        spdlog::error("[ShoeboxAcousticsScene] Room or source marker shader failed to load");
        return false;
    }

    const uint32_t uniformMaterial = AcousticMaterialLibrary::Instance().GetMaterialId("Uniform02");
    SetSceneName("Shoebox Acoustics");
    SetSkyboxVisible(false);
    SetAmbientLight({0.55f, 0.59f, 0.62f}, 0.7f);
    SetFirstPersonEyeHeight(kListenerPosition.y);
    m_cameraOrbitRadius = 2.5f;

    m_roomMesh = std::make_shared<Mesh>(BuildRoomMesh());
    auto baseMaterial = Material::CreateDefaultMaterial(roomShader);
    baseMaterial->SetFloat("u_RoughnessValue", 0.85f);
    for (uint32_t surface = 0; surface < kSurfaces.size(); ++surface)
    {
        auto& material = m_roomMaterials[surface];
        material = std::make_shared<MaterialInstance>(baseMaterial);
        material->SetName(kSurfaces[surface].name);
        material->SetVec3("u_AlbedoColor", kSurfaces[surface].color);

        RenderItem item;
        item.meshMulti = m_roomMesh.get();
        item.subMeshIndex = surface;
        item.material = material.get();
        item.acousticMaterialId = uniformMaterial;
        AddObject(item);
    }

    m_sourceMarkerShader = std::move(markerShader);
    m_sourceMarker = std::make_shared<MeshBuffer>(
        GenerateSphere(0.075f, 12, {.colorMode = ColorMode::Solid, .baseColor = kSourceColor})
            .CreateMeshBuffer());
    // The marker uses the visual primitive path so it contributes no acoustic triangles.
    RenderItem marker;
    marker.mesh = m_sourceMarker.get();
    marker.shader = m_sourceMarkerShader.get();
    marker.flags.castShadow = false;
    marker.flags.receiveShadow = false;
    m_sourceMarkerIndex = AddObject(marker);

    SetValidationPreset(ValidationPreset::FrontNear);
    return true;
}

float ShoeboxAcousticsScene::GetEnclosedAcousticVolume() const
{
    return kRoomDimensions.x * kRoomDimensions.y * kRoomDimensions.z;
}

void ShoeboxAcousticsScene::OnUpdate(float deltaTime, IInputProvider& input)
{
    glm::vec3 moveDirXZ{0.0f};
    UpdateStandardCameraAndPlayer(deltaTime, input, m_playerPosition, moveDirXZ);

    const auto& lights = GetLights().GetPointLights();
    const auto source = std::find_if(lights.begin(), lights.end(), [](const PointLight& light) {
        return light.name == kSourceName;
    });
    auto& marker = GetObject(m_sourceMarkerIndex);
    marker.flags.visible = source != lights.end() && source->enabled;
    if (marker.flags.visible)
        marker.transform.SetTranslation(source->position);
}

void ShoeboxAcousticsScene::SetValidationPreset(ValidationPreset preset)
{
    glm::vec3 offset{0.0f, 0.0f, -2.0f};
    switch (preset)
    {
        case ValidationPreset::FrontNear: break;
        case ValidationPreset::FrontFar:  offset.z = -4.0f; break;
        case ValidationPreset::Right:     offset = { 2.0f, 0.0f, 0.0f}; break;
        case ValidationPreset::Left:      offset = {-2.0f, 0.0f, 0.0f}; break;
    }
    const glm::vec3 sourcePosition = kListenerPosition + offset;

    Camera camera;
    camera.SetPosition(kListenerPosition);
    camera.SetOrientation(-90.0f, 0.0f);
    SetCamera(camera);
    m_playerPosition = {kListenerPosition.x, 0.0f, kListenerPosition.z};

    auto& lights = GetLights();
    lights.Clear();
    lights.AddPointLight(PointLightBuilder()
        .Position(sourcePosition)
        .Color(kSourceColor)
        .Intensity(3.0f)
        .Radius(18.0f)
        .Name(kSourceName)
        .Build());

    auto& marker = GetObject(m_sourceMarkerIndex);
    marker.flags.visible = true;
    marker.transform.SetTranslation(sourcePosition);
}
