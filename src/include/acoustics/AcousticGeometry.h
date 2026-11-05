#pragma once

#include "core/AABB.h"
#include "acoustics/AcousticMaterial.h"
#include <glm/glm.hpp>
#include <vector>
#include <cstdint>

class Mesh;
class Scene;

/// A single triangle in world space for acoustic ray tracing.
struct AcousticTriangle {
    glm::vec3 v0{0.0f};
    glm::vec3 v1{0.0f};
    glm::vec3 v2{0.0f};
    glm::vec3 normal{0.0f, 1.0f, 0.0f};
    float area = 0.0f;
    uint32_t materialId = 0;

    AcousticTriangle() = default;
    AcousticTriangle(const glm::vec3& p0, const glm::vec3& p1, const glm::vec3& p2, uint32_t matId);

    [[nodiscard]] AABB GetBounds() const;
    [[nodiscard]] glm::vec3 GetCentroid() const;
};

/// CPU-side collection of triangles representing scene geometry for acoustic ray tracing.
class AcousticSceneGeometry {
public:
    AcousticSceneGeometry() = default;

    /// Clears all triangles and reset bounds/metrics.
    void Clear();

    /// Adds a single triangle to the acoustic scene.
    void AddTriangle(const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2, uint32_t materialId);

    /// Adds a quad as two triangles (v0-v1-v2, v0-v2-v3).
    void AddQuad(const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2, const glm::vec3& v3, uint32_t materialId);

    /// Adds an axis-aligned box (e.g. shoebox room). If insideFacing is true, normals face inwards.
    void AddBox(const glm::vec3& min, const glm::vec3& max, uint32_t materialId, bool insideFacing = true);

    /// Adds a rectangular shoebox room with individual materials for floor, ceiling, and walls.
    void AddShoeboxRoom(const glm::vec3& dimensions,
                        uint32_t floorMatId,
                        uint32_t ceilingMatId,
                        uint32_t wallsMatId,
                        const glm::vec3& center = glm::vec3(0.0f));

    /// Adds raw mesh geometry with a world-space transformation matrix.
    void AddMesh(const std::vector<glm::vec3>& positions,
                 const std::vector<uint32_t>& indices,
                 const glm::mat4& transform,
                 uint32_t materialId);

    /// Adds an engine Mesh instance using its CPU geometry and world transform.
    bool AddMesh(const Mesh& mesh, const glm::mat4& transform, uint32_t defaultMaterialId = 0);

    /// Automatically extracts acoustic geometry from all RenderItems in the given Scene.
    void ExtractFromScene(const Scene& scene, uint32_t defaultMaterialId = 0);

    /// Recomputes bounds, total surface area, and estimated volume.
    void Finalize();

    [[nodiscard]] const std::vector<AcousticTriangle>& GetTriangles() const { return m_triangles; }
    [[nodiscard]] const AABB& GetBounds() const { return m_bounds; }
    [[nodiscard]] float GetTotalSurfaceArea() const { return m_totalSurfaceArea; }
    [[nodiscard]] float GetEstimatedVolume() const { return m_estimatedVolume; }
    [[nodiscard]] size_t GetTriangleCount() const { return m_triangles.size(); }
    [[nodiscard]] bool IsEmpty() const { return m_triangles.empty(); }

    /// Sets explicit room volume (m³) when known analytically.
    void SetExplicitVolume(float volume) { m_estimatedVolume = volume; }

    // --- Analytical Room Acoustics (Sabine and Eyring equations) ---

    /// Average surface-weighted absorption coefficient α_avg = (Σ S_i * α_i) / Σ S_i.
    [[nodiscard]] float CalculateAverageAbsorption(size_t bandIndex, const AcousticMaterialLibrary& materials) const;

    /// Total metric absorption area A = Σ S_i * α_i (metric Sabins, m²).
    [[nodiscard]] float CalculateTotalAbsorptionArea(size_t bandIndex, const AcousticMaterialLibrary& materials) const;

    /// Sabine reverberation time: RT60 ≈ 0.161 * V / A.
    [[nodiscard]] float CalculateSabineRT60(size_t bandIndex, const AcousticMaterialLibrary& materials) const;

    /// Eyring reverberation time: RT60 ≈ 0.161 * V / (-S * ln(1 - α_avg)).
    [[nodiscard]] float CalculateEyringRT60(size_t bandIndex, const AcousticMaterialLibrary& materials) const;

private:
    std::vector<AcousticTriangle> m_triangles;
    AABB m_bounds = AABB::Empty();
    float m_totalSurfaceArea = 0.0f;
    float m_estimatedVolume = 0.0f;
};
