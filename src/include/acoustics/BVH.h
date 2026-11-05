#pragma once

#include "core/AABB.h"
#include "acoustics/AcousticRay.h"
#include "acoustics/AcousticGeometry.h"
#include <vector>
#include <cstdint>
#include <limits>

/// Node in the linear BVH acceleration structure.
/// Aligned to 32 bytes for CPU cache line efficiency (exactly 2 nodes per 64-byte cache line).
struct alignas(32) BVHNode {
    AABB bounds; // 24 bytes
    union {
        uint32_t primitivesOffset;   // Leaf: offset into triangle array
        uint32_t secondChildOffset;  // Interior: index of right child (left child is always nodeIndex + 1)
    };                               // 4 bytes
    uint16_t primitiveCount = 0;     // 2 bytes (0 for interior node, >0 for leaf)
    uint8_t axis = 0;                // 1 byte (split axis: 0=X, 1=Y, 2=Z)
    uint8_t padding = 0;             // 1 byte

    [[nodiscard]] bool IsLeaf() const { return primitiveCount > 0; }
};
static_assert(sizeof(BVHNode) == 32, "BVHNode must be exactly 32 bytes for optimal CPU cache utilization");

/// High-performance Binned SAH (Surface Area Heuristic) Bounding Volume Hierarchy.
/// Provides O(log N) ray-scene intersection tests for acoustic ray tracing.
class BVH {
public:
    BVH() = default;
    ~BVH() = default;

    /// Builds the BVH from an AcousticSceneGeometry instance.
    void Build(const AcousticSceneGeometry& sceneGeometry);

    /// Builds the BVH from a vector of AcousticTriangles.
    void Build(std::vector<AcousticTriangle> triangles);

    /// Clears the BVH tree and triangle storage.
    void Clear();

    /// Finds the closest triangle intersection along the ray between tMin and tMax.
    /// Traversal is non-recursive (iterative with fixed local stack of 64 nodes) and branch-pruned.
    [[nodiscard]] AcousticHit Intersect(const AcousticRay& ray,
                                        float tMin = 0.001f,
                                        float tMax = 1e6f) const;

    /// Tests if any triangle intersects the ray between tMin and tMax (shadow / occlusion test).
    /// Returns true immediately upon first confirmed hit without finding closest distance.
    [[nodiscard]] bool IntersectAny(const AcousticRay& ray,
                                    float tMin = 0.001f,
                                    float tMax = 1e6f) const;

    /// Returns true if the BVH is built and has valid nodes.
    [[nodiscard]] bool IsValid() const { return !m_nodes.empty() && !m_triangles.empty(); }

    /// Returns overall scene bounding box (root node bounds).
    [[nodiscard]] AABB GetBounds() const { return m_nodes.empty() ? AABB::Empty() : m_nodes[0].bounds; }

    /// Statistics
    [[nodiscard]] size_t GetNodeCount() const { return m_nodes.size(); }
    [[nodiscard]] size_t GetTriangleCount() const { return m_triangles.size(); }
    [[nodiscard]] size_t GetLeafCount() const;
    [[nodiscard]] uint32_t GetMaxDepth() const;

    [[nodiscard]] const std::vector<BVHNode>& GetNodes() const { return m_nodes; }
    [[nodiscard]] const std::vector<AcousticTriangle>& GetTriangles() const { return m_triangles; }

private:
    struct BuildPrimitive {
        uint32_t triangleIndex = 0;
        AABB bounds;
        glm::vec3 centroid{0.0f};
    };

    struct Bin {
        uint32_t count = 0;
        AABB bounds = AABB::Empty();
    };

    static constexpr uint32_t kNumBins = 16;
    static constexpr uint32_t kMaxLeafPrimitives = 4;
    static constexpr float kTraversalCost = 1.0f;
    static constexpr float kIntersectionCost = 1.5f;

    uint32_t RecursiveBuild(std::vector<BuildPrimitive>& prims,
                           uint32_t start,
                           uint32_t end,
                           const std::vector<AcousticTriangle>& sourceTriangles,
                           std::vector<AcousticTriangle>& orderedTriangles,
                           uint32_t currentDepth);

    std::vector<BVHNode> m_nodes;
    std::vector<AcousticTriangle> m_triangles;
};
