#pragma once

#include "scene/Scene.h"

#include <array>
#include <cstddef>
#include <memory>

class MaterialInstance;
class Mesh;
class MeshBuffer;
class ShaderProgram;

/// A rectangular 10 m × 8 m × 3 m room for repeatable acoustic measurements.
class ShoeboxAcousticsScene final : public Scene
{
public:
    enum class ValidationPreset
    {
        FrontNear,
        FrontFar,
        Right,
        Left,
    };

    bool Setup();
    void OnUpdate(float deltaTime, IInputProvider& input) override;

    /// After setup, reset the listener pose and move the source to a known position.
    void SetValidationPreset(ValidationPreset preset);

private:
    std::shared_ptr<Mesh> m_roomMesh;
    std::array<std::shared_ptr<MaterialInstance>, 3> m_roomMaterials;
    std::shared_ptr<MeshBuffer> m_sourceMarker;
    std::shared_ptr<ShaderProgram> m_sourceMarkerShader;
    std::size_t m_sourceMarkerIndex = 0;
    glm::vec3 m_playerPosition{0.0f};
};
