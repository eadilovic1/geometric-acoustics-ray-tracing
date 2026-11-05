#include "acoustics/AcousticGeometry.h"
#include "core/Mesh.h"
#include "scene/Scene.h"
#include "scene/RenderItem.h"
#include <glm/gtc/matrix_transform.hpp>
#include <cmath>
#include <algorithm>
#include <spdlog/spdlog.h>

// ---------------------------------------------------------------------------
//  AcousticTriangle
// ---------------------------------------------------------------------------

AcousticTriangle::AcousticTriangle(const glm::vec3& p0, const glm::vec3& p1, const glm::vec3& p2, uint32_t matId)
    : v0(p0), v1(p1), v2(p2), materialId(matId)
{
    const glm::vec3 edge1 = v1 - v0;
    const glm::vec3 edge2 = v2 - v0;
    const glm::vec3 crossProduct = glm::cross(edge1, edge2);
    const float len = glm::length(crossProduct);

    area = 0.5f * len;
    normal = (len > 1e-7f) ? (crossProduct / len) : glm::vec3(0.0f, 1.0f, 0.0f);
}

AABB AcousticTriangle::GetBounds() const {
    AABB box = AABB::Empty();
    box.Expand(v0);
    box.Expand(v1);
    box.Expand(v2);
    return box;
}

glm::vec3 AcousticTriangle::GetCentroid() const {
    return (v0 + v1 + v2) * (1.0f / 3.0f);
}

// ---------------------------------------------------------------------------
//  AcousticSceneGeometry
// ---------------------------------------------------------------------------

void AcousticSceneGeometry::Clear() {
    m_triangles.clear();
    m_bounds = AABB::Empty();
    m_totalSurfaceArea = 0.0f;
    m_estimatedVolume = 0.0f;
}

void AcousticSceneGeometry::AddTriangle(const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2, uint32_t materialId) {
    m_triangles.emplace_back(v0, v1, v2, materialId);
}

void AcousticSceneGeometry::AddQuad(const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2, const glm::vec3& v3, uint32_t materialId) {
    AddTriangle(v0, v1, v2, materialId);
    AddTriangle(v0, v2, v3, materialId);
}

void AcousticSceneGeometry::AddBox(const glm::vec3& min, const glm::vec3& max, uint32_t materialId, bool insideFacing) {
    // 8 vertices of the box
    const glm::vec3 p000(min.x, min.y, min.z);
    const glm::vec3 p100(max.x, min.y, min.z);
    const glm::vec3 p110(max.x, max.y, min.z);
    const glm::vec3 p010(min.x, max.y, min.z);
    const glm::vec3 p001(min.x, min.y, max.z);
    const glm::vec3 p101(max.x, min.y, max.z);
    const glm::vec3 p111(max.x, max.y, max.z);
    const glm::vec3 p011(min.x, max.y, max.z);

    if (insideFacing) {
        // Normals point towards the interior of the room
        // Floor (-Y, normal +Y)
        AddQuad(p000, p001, p101, p100, materialId);
        // Ceiling (+Y, normal -Y)
        AddQuad(p010, p110, p111, p011, materialId);
        // Front wall (+Z, normal -Z)
        AddQuad(p001, p011, p111, p101, materialId);
        // Back wall (-Z, normal +Z)
        AddQuad(p100, p110, p010, p000, materialId);
        // Left wall (-X, normal +X)
        AddQuad(p000, p010, p011, p001, materialId);
        // Right wall (+X, normal -X)
        AddQuad(p101, p111, p110, p100, materialId);
    } else {
        // Outside facing (exterior box)
        // Bottom (-Y)
        AddQuad(p000, p100, p101, p001, materialId);
        // Top (+Y)
        AddQuad(p010, p011, p111, p110, materialId);
        // Front (+Z)
        AddQuad(p001, p101, p111, p011, materialId);
        // Back (-Z)
        AddQuad(p100, p000, p010, p110, materialId);
        // Left (-X)
        AddQuad(p000, p001, p011, p010, materialId);
        // Right (+X)
        AddQuad(p101, p100, p110, p111, materialId);
    }
}

void AcousticSceneGeometry::AddShoeboxRoom(const glm::vec3& dimensions,
                                          uint32_t floorMatId,
                                          uint32_t ceilingMatId,
                                          uint32_t wallsMatId,
                                          const glm::vec3& center)
{
    const glm::vec3 min = center - dimensions * 0.5f;
    const glm::vec3 max = center + dimensions * 0.5f;

    const glm::vec3 p000(min.x, min.y, min.z);
    const glm::vec3 p100(max.x, min.y, min.z);
    const glm::vec3 p110(max.x, max.y, min.z);
    const glm::vec3 p010(min.x, max.y, min.z);
    const glm::vec3 p001(min.x, min.y, max.z);
    const glm::vec3 p101(max.x, min.y, max.z);
    const glm::vec3 p111(max.x, max.y, max.z);
    const glm::vec3 p011(min.x, max.y, max.z);

    // Floor (-Y, inwards normal +Y)
    AddQuad(p000, p001, p101, p100, floorMatId);

    // Ceiling (+Y, inwards normal -Y)
    AddQuad(p010, p110, p111, p011, ceilingMatId);

    // 4 Walls (inwards normals)
    // Front wall (+Z, normal -Z)
    AddQuad(p001, p011, p111, p101, wallsMatId);
    // Back wall (-Z, normal +Z)
    AddQuad(p100, p110, p010, p000, wallsMatId);
    // Left wall (-X, normal +X)
    AddQuad(p000, p010, p011, p001, wallsMatId);
    // Right wall (+X, normal -X)
    AddQuad(p101, p111, p110, p100, wallsMatId);

    SetExplicitVolume(dimensions.x * dimensions.y * dimensions.z);
    Finalize();
}

void AcousticSceneGeometry::AddMesh(const std::vector<glm::vec3>& positions,
                                    const std::vector<uint32_t>& indices,
                                    const glm::mat4& transform,
                                    uint32_t materialId)
{
    if (positions.empty() || indices.empty()) return;

    std::vector<glm::vec3> worldPositions(positions.size());
    for (size_t i = 0; i < positions.size(); ++i) {
        worldPositions[i] = glm::vec3(transform * glm::vec4(positions[i], 1.0f));
    }

    const size_t triCount = indices.size() / 3;
    m_triangles.reserve(m_triangles.size() + triCount);

    for (size_t i = 0; i + 2 < indices.size(); i += 3) {
        const uint32_t idx0 = indices[i];
        const uint32_t idx1 = indices[i + 1];
        const uint32_t idx2 = indices[i + 2];

        if (idx0 < worldPositions.size() && idx1 < worldPositions.size() && idx2 < worldPositions.size()) {
            AddTriangle(worldPositions[idx0], worldPositions[idx1], worldPositions[idx2], materialId);
        }
    }
}

bool AcousticSceneGeometry::AddMesh(const Mesh& mesh, const glm::mat4& transform, uint32_t defaultMaterialId) {
    if (!mesh.HasCpuGeometry()) {
        spdlog::warn("[AcousticGeometry] Mesh '{}' has no CPU geometry stored", mesh.GetName());
        return false;
    }

    const auto& vertices = mesh.GetCpuVertices();
    const auto& indices = mesh.GetCpuIndices();

    std::vector<glm::vec3> positions;
    positions.reserve(vertices.size());
    for (const auto& v : vertices) {
        positions.push_back(v.position);
    }

    AddMesh(positions, indices, transform, defaultMaterialId);
    return true;
}

void AcousticSceneGeometry::ExtractFromScene(const Scene& scene, uint32_t defaultMaterialId) {
    Clear();
    const auto& objects = scene.GetObjects();
    spdlog::info("[AcousticGeometry] Extracting geometry from scene: {} objects", objects.size());

    for (const auto& item : objects) {
        glm::mat4 modelMat = item.transform.GetModelMatrix();
        const glm::mat4 offsetMat = glm::translate(glm::mat4(1.0f), item.translationOffset) * glm::mat4_cast(item.rotationOffset);
        const glm::mat4 worldMat = modelMat * offsetMat;

        const uint32_t matId = (item.acousticMaterialId != 0) ? item.acousticMaterialId : defaultMaterialId;

        if (item.meshMulti) {
            AddMesh(*item.meshMulti, worldMat, matId);
        }
    }

    Finalize();
    spdlog::info("[AcousticGeometry] Extraction complete: {} triangles, total area = {:.2f} m², volume = {:.2f} m³",
                 m_triangles.size(), m_totalSurfaceArea, m_estimatedVolume);
}

void AcousticSceneGeometry::Finalize() {
    m_bounds = AABB::Empty();
    m_totalSurfaceArea = 0.0f;

    for (const auto& tri : m_triangles) {
        m_bounds.Expand(tri.v0);
        m_bounds.Expand(tri.v1);
        m_bounds.Expand(tri.v2);
        m_totalSurfaceArea += tri.area;
    }

    if (m_estimatedVolume <= 0.0f && m_bounds.IsValid()) {
        const glm::vec3 extents = m_bounds.max - m_bounds.min;
        m_estimatedVolume = extents.x * extents.y * extents.z;
    }
}

float AcousticSceneGeometry::CalculateTotalAbsorptionArea(size_t bandIndex, const AcousticMaterialLibrary& materials) const {
    if (bandIndex >= OctaveBands::Count) return 0.0f;

    float totalA = 0.0f;
    for (const auto& tri : m_triangles) {
        const AcousticMaterial& mat = materials.GetMaterial(tri.materialId);
        totalA += tri.area * mat.absorption[bandIndex];
    }
    return totalA;
}

float AcousticSceneGeometry::CalculateAverageAbsorption(size_t bandIndex, const AcousticMaterialLibrary& materials) const {
    if (m_totalSurfaceArea <= 0.0f) return 0.0f;
    const float totalA = CalculateTotalAbsorptionArea(bandIndex, materials);
    return std::clamp(totalA / m_totalSurfaceArea, 0.0f, 1.0f);
}

float AcousticSceneGeometry::CalculateSabineRT60(size_t bandIndex, const AcousticMaterialLibrary& materials) const {
    const float totalA = CalculateTotalAbsorptionArea(bandIndex, materials);
    if (totalA <= 1e-5f || m_estimatedVolume <= 0.0f) return 0.0f;

    // Sabine equation: RT60 ≈ 0.161 * V / A
    return 0.161f * (m_estimatedVolume / totalA);
}

float AcousticSceneGeometry::CalculateEyringRT60(size_t bandIndex, const AcousticMaterialLibrary& materials) const {
    if (m_totalSurfaceArea <= 0.0f || m_estimatedVolume <= 0.0f) return 0.0f;

    const float alphaAvg = CalculateAverageAbsorption(bandIndex, materials);
    // Eyring formula: RT60 ≈ 0.161 * V / (-S * ln(1 - α_avg))
    const float clampedAlpha = std::min(alphaAvg, 0.999f);
    const float denominator = -m_totalSurfaceArea * std::log(1.0f - clampedAlpha);

    if (denominator <= 1e-5f) return 0.0f;
    return 0.161f * (m_estimatedVolume / denominator);
}
