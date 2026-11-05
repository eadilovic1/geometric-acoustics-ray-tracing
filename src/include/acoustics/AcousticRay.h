#pragma once

#include "core/AABB.h"
#include "acoustics/AcousticMaterial.h"
#include "acoustics/AcousticGeometry.h"
#include <glm/glm.hpp>
#include <cmath>
#include <algorithm>
#include <limits>
#include <cstdint>

/// Acoustic ray carrying energy across 7 ISO octave bands.
/// Represents a discrete packet of acoustic energy propagating through 3D space.
struct AcousticRay {
    glm::vec3 origin{0.0f};
    glm::vec3 direction{0.0f, 0.0f, 1.0f};
    glm::vec3 invDirection{0.0f, 0.0f, 1.0f}; // 1.0f / direction (cached for branchless slab test)
    OctaveBands energy{1.0f};                  // Energy per octave band (Watts or relative energy)
    float distanceTraveled = 0.0f;             // Total path length traversed (meters)
    uint32_t bounceCount = 0;                  // Number of surface reflections

    AcousticRay() = default;

    AcousticRay(const glm::vec3& rayOrigin,
                const glm::vec3& rayDirection,
                const OctaveBands& initialEnergy = OctaveBands(1.0f),
                float initialDistance = 0.0f,
                uint32_t bounces = 0)
        : origin(rayOrigin)
        , direction(glm::normalize(rayDirection))
        , energy(initialEnergy)
        , distanceTraveled(initialDistance)
        , bounceCount(bounces)
    {
        UpdateInvDirection();
    }

    /// Sets a new normalized direction and updates cached inverse direction.
    void SetDirection(const glm::vec3& newDir) {
        direction = glm::normalize(newDir);
        UpdateInvDirection();
    }

    /// Recomputes cached inverse direction components with protection against division by zero.
    void UpdateInvDirection() {
        constexpr float kEpsilon = 1e-9f;
        constexpr float kLarge = 1e9f;
        invDirection.x = (std::abs(direction.x) > kEpsilon) ? (1.0f / direction.x) : (direction.x >= 0.0f ? kLarge : -kLarge);
        invDirection.y = (std::abs(direction.y) > kEpsilon) ? (1.0f / direction.y) : (direction.y >= 0.0f ? kLarge : -kLarge);
        invDirection.z = (std::abs(direction.z) > kEpsilon) ? (1.0f / direction.z) : (direction.z >= 0.0f ? kLarge : -kLarge);
    }

    /// Returns point at distance t along the ray.
    [[nodiscard]] glm::vec3 PointAt(float t) const {
        return origin + direction * t;
    }

    /// Advances the ray by distance dist: moves origin and increments distanceTraveled.
    void Advance(float dist) {
        origin = PointAt(dist);
        distanceTraveled += dist;
    }

    /// Returns total energy across all bands (average).
    [[nodiscard]] float AverageEnergy() const {
        return energy.Average();
    }
};

/// Result of a ray intersection test with acoustic scene geometry.
struct AcousticHit {
    float distance = std::numeric_limits<float>::max();
    glm::vec3 point{0.0f};
    glm::vec3 normal{0.0f};
    float u = 0.0f;
    float v = 0.0f;
    uint32_t triangleIndex = std::numeric_limits<uint32_t>::max();
    uint32_t materialId = 0;

    [[nodiscard]] bool HasHit() const {
        return distance < std::numeric_limits<float>::max();
    }
};

// ---------------------------------------------------------------------------
//  Intersection Algorithms
// ---------------------------------------------------------------------------

/// Branchless Ray-AABB Slab intersection test.
/// Uses precomputed invDirection to avoid divisions and branch mispredictions.
[[nodiscard]] inline bool IntersectAABB(const AcousticRay& ray,
                                        const AABB& box,
                                        float tMinLimit,
                                        float tMaxLimit,
                                        float& outTNear)
{
    const glm::vec3 t0 = (box.min - ray.origin) * ray.invDirection;
    const glm::vec3 t1 = (box.max - ray.origin) * ray.invDirection;
    const glm::vec3 tNear = glm::min(t0, t1);
    const glm::vec3 tFar  = glm::max(t0, t1);

    const float enter = std::max({ tMinLimit, tNear.x, tNear.y, tNear.z });
    const float exit  = std::min({ tMaxLimit, tFar.x,  tFar.y,  tFar.z  });

    outTNear = enter;
    return enter <= exit;
}

/// Möller–Trumbore ray–triangle intersection algorithm.
/// Numerically stable, exact, double-sided (handles front and back faces).
[[nodiscard]] inline bool IntersectTriangle(const AcousticRay& ray,
                                            const AcousticTriangle& tri,
                                            float tMin,
                                            float tMax,
                                            float& outT,
                                            float& outU,
                                            float& outV)
{
    constexpr float kEpsilon = 1e-8f;
    const glm::vec3 edge1 = tri.v1 - tri.v0;
    const glm::vec3 edge2 = tri.v2 - tri.v0;
    const glm::vec3 pvec  = glm::cross(ray.direction, edge2);
    const float det       = glm::dot(edge1, pvec);

    // Reject ray if it is nearly parallel to triangle plane
    if (std::abs(det) < kEpsilon) return false;

    const float invDet   = 1.0f / det;
    const glm::vec3 tvec = ray.origin - tri.v0;
    const float u        = glm::dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return false;

    const glm::vec3 qvec = glm::cross(tvec, edge1);
    const float v        = glm::dot(ray.direction, qvec) * invDet;
    if (v < 0.0f || (u + v) > 1.0f) return false;

    const float t        = glm::dot(edge2, qvec) * invDet;
    if (t < tMin || t > tMax) return false;

    outT = t;
    outU = u;
    outV = v;
    return true;
}

/// Ray–sphere intersection test.
/// Used for receiver ear spheres (binaural detection) and central head sphere (ILD shadowing).
/// Returns true if ray intersects sphere within [tMin, tMax], computing chord segment [outT1, outT2].
[[nodiscard]] inline bool IntersectSphere(const AcousticRay& ray,
                                          const glm::vec3& center,
                                          float radius,
                                          float tMin,
                                          float tMax,
                                          float& outT1,
                                          float& outT2)
{
    const glm::vec3 oc = ray.origin - center;
    const float a = glm::dot(ray.direction, ray.direction);
    const float b = 2.0f * glm::dot(oc, ray.direction);
    const float c = glm::dot(oc, oc) - radius * radius;
    const float discriminant = b * b - 4.0f * a * c;

    if (discriminant < 0.0f) return false;

    const float sqrtD = std::sqrt(discriminant);
    const float inv2a = 0.5f / a;
    float t1 = (-b - sqrtD) * inv2a;
    float t2 = (-b + sqrtD) * inv2a;

    if (t1 > t2) std::swap(t1, t2);

    if (t2 < tMin || t1 > tMax) return false;

    outT1 = std::max(t1, tMin);
    outT2 = std::min(t2, tMax);
    return outT1 <= outT2;
}
