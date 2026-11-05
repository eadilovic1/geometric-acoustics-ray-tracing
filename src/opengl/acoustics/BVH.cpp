#include "acoustics/BVH.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>
#include <limits>

namespace {

[[nodiscard]] inline float ComputeAABBSurfaceArea(const AABB& box) {
    if (!box.IsValid()) return 0.0f;
    const glm::vec3 d = box.max - box.min;
    return 2.0f * (d.x * d.y + d.y * d.z + d.z * d.x);
}

} // namespace

void BVH::Clear() {
    m_nodes.clear();
    m_triangles.clear();
}

void BVH::Build(const AcousticSceneGeometry& sceneGeometry) {
    Build(sceneGeometry.GetTriangles());
}

void BVH::Build(std::vector<AcousticTriangle> triangles) {
    Clear();
    if (triangles.empty()) {
        spdlog::warn("[BVH] Build called with 0 triangles");
        return;
    }

    const size_t triCount = triangles.size();
    std::vector<BuildPrimitive> prims(triCount);
    for (size_t i = 0; i < triCount; ++i) {
        prims[i].triangleIndex = static_cast<uint32_t>(i);
        prims[i].bounds = triangles[i].GetBounds();
        prims[i].centroid = triangles[i].GetCentroid();
    }

    m_nodes.reserve(2 * triCount + 1);
    std::vector<AcousticTriangle> orderedTriangles;
    orderedTriangles.reserve(triCount);

    RecursiveBuild(prims, 0, static_cast<uint32_t>(triCount), triangles, orderedTriangles, 0);

    m_triangles = std::move(orderedTriangles);

    spdlog::info("[BVH] Built {} nodes (leaves: {}, max depth: {}) for {} triangles. Bounds: [{:.2f}, {:.2f}, {:.2f}] to [{:.2f}, {:.2f}, {:.2f}]",
                 m_nodes.size(), GetLeafCount(), GetMaxDepth(), m_triangles.size(),
                 m_nodes[0].bounds.min.x, m_nodes[0].bounds.min.y, m_nodes[0].bounds.min.z,
                 m_nodes[0].bounds.max.x, m_nodes[0].bounds.max.y, m_nodes[0].bounds.max.z);
}

uint32_t BVH::RecursiveBuild(std::vector<BuildPrimitive>& prims,
                             uint32_t start,
                             uint32_t end,
                             const std::vector<AcousticTriangle>& sourceTriangles,
                             std::vector<AcousticTriangle>& orderedTriangles,
                             uint32_t currentDepth)
{
    const uint32_t count = end - start;
    const uint32_t nodeIndex = static_cast<uint32_t>(m_nodes.size());
    m_nodes.emplace_back();

    // Compute bounding boxes for geometry and centroids
    AABB nodeBounds = AABB::Empty();
    AABB centroidBounds = AABB::Empty();
    for (uint32_t i = start; i < end; ++i) {
        nodeBounds.Expand(prims[i].bounds);
        centroidBounds.Expand(prims[i].centroid);
    }
    m_nodes[nodeIndex].bounds = nodeBounds;

    // Helper to turn current node into a leaf
    auto makeLeaf = [&]() -> uint32_t {
        m_nodes[nodeIndex].primitivesOffset = static_cast<uint32_t>(orderedTriangles.size());
        m_nodes[nodeIndex].primitiveCount = static_cast<uint16_t>(count);
        m_nodes[nodeIndex].axis = 0;
        for (uint32_t i = start; i < end; ++i) {
            orderedTriangles.push_back(sourceTriangles[prims[i].triangleIndex]);
        }
        return nodeIndex;
    };

    if (count <= kMaxLeafPrimitives || currentDepth >= 32) {
        return makeLeaf();
    }

    const glm::vec3 centroidExtent = centroidBounds.max - centroidBounds.min;
    if (glm::length(centroidExtent) < 1e-6f) {
        return makeLeaf();
    }

    // Binned SAH across 3 axes
    const float nodeArea = ComputeAABBSurfaceArea(nodeBounds);
    const float leafCost = static_cast<float>(count) * kIntersectionCost;

    float bestCost = std::numeric_limits<float>::max();
    int bestAxis = -1;
    int bestSplitBin = -1;

    for (int axis = 0; axis < 3; ++axis) {
        const float extent = centroidExtent[axis];
        if (extent < 1e-6f) continue;

        Bin bins[kNumBins];
        const float scale = static_cast<float>(kNumBins) / extent;
        const float minVal = centroidBounds.min[axis];

        for (uint32_t i = start; i < end; ++i) {
            const float norm = (prims[i].centroid[axis] - minVal) * scale;
            const int b = std::clamp(static_cast<int>(norm), 0, static_cast<int>(kNumBins - 1));
            bins[b].count++;
            bins[b].bounds.Expand(prims[i].bounds);
        }

        // Forward and backward sweeps for candidate splits
        AABB leftBounds[kNumBins - 1];
        uint32_t leftCounts[kNumBins - 1];
        AABB curLeftBox = AABB::Empty();
        uint32_t curLeftCount = 0;

        for (uint32_t i = 0; i < kNumBins - 1; ++i) {
            curLeftBox.Expand(bins[i].bounds);
            curLeftCount += bins[i].count;
            leftBounds[i] = curLeftBox;
            leftCounts[i] = curLeftCount;
        }

        AABB rightBounds[kNumBins - 1];
        uint32_t rightCounts[kNumBins - 1];
        AABB curRightBox = AABB::Empty();
        uint32_t curRightCount = 0;

        for (int i = static_cast<int>(kNumBins) - 2; i >= 0; --i) {
            curRightBox.Expand(bins[i + 1].bounds);
            curRightCount += bins[i + 1].count;
            rightBounds[i] = curRightBox;
            rightCounts[i] = curRightCount;
        }

        const float invNodeArea = (nodeArea > 1e-7f) ? (1.0f / nodeArea) : 0.0f;
        for (uint32_t i = 0; i < kNumBins - 1; ++i) {
            if (leftCounts[i] == 0 || rightCounts[i] == 0) continue;

            const float leftArea = ComputeAABBSurfaceArea(leftBounds[i]);
            const float rightArea = ComputeAABBSurfaceArea(rightBounds[i]);

            const float cost = kTraversalCost + kIntersectionCost * (leftCounts[i] * leftArea + rightCounts[i] * rightArea) * invNodeArea;

            if (cost < bestCost) {
                bestCost = cost;
                bestAxis = axis;
                bestSplitBin = static_cast<int>(i);
            }
        }
    }

    // If SAH indicates that creating a leaf is cheaper than splitting
    if (bestAxis == -1 || (bestCost >= leafCost && count <= 8)) {
        return makeLeaf();
    }

    // Partition primitives according to best axis and bin
    const float extent = centroidExtent[bestAxis];
    const float scale = static_cast<float>(kNumBins) / extent;
    const float minVal = centroidBounds.min[bestAxis];

    auto midIter = std::partition(prims.begin() + start, prims.begin() + end, [&](const BuildPrimitive& p) {
        const float norm = (p.centroid[bestAxis] - minVal) * scale;
        const int b = std::clamp(static_cast<int>(norm), 0, static_cast<int>(kNumBins - 1));
        return b <= bestSplitBin;
    });

    uint32_t mid = static_cast<uint32_t>(midIter - prims.begin());
    if (mid == start || mid == end) {
        mid = start + (count / 2);
        std::nth_element(prims.begin() + start, prims.begin() + mid, prims.begin() + end,
                         [bestAxis](const BuildPrimitive& a, const BuildPrimitive& b) {
                             return a.centroid[bestAxis] < b.centroid[bestAxis];
                         });
    }

    m_nodes[nodeIndex].primitiveCount = 0; // Interior node
    m_nodes[nodeIndex].axis = static_cast<uint8_t>(bestAxis);

    // Build left child (depth-first: left child will be placed at nodeIndex + 1)
    RecursiveBuild(prims, start, mid, sourceTriangles, orderedTriangles, currentDepth + 1);

    // Build right child
    const uint32_t rightChildIndex = static_cast<uint32_t>(m_nodes.size());
    m_nodes[nodeIndex].secondChildOffset = rightChildIndex;
    RecursiveBuild(prims, mid, end, sourceTriangles, orderedTriangles, currentDepth + 1);

    return nodeIndex;
}

AcousticHit BVH::Intersect(const AcousticRay& ray, float tMin, float tMax) const {
    AcousticHit closestHit;
    if (m_nodes.empty()) return closestHit;

    closestHit.distance = tMax;

    uint32_t stack[64];
    int stackPtr = 0;
    stack[stackPtr++] = 0;

    while (stackPtr > 0) {
        const uint32_t nodeIdx = stack[--stackPtr];
        const BVHNode& node = m_nodes[nodeIdx];

        float tNear = 0.0f;
        if (!IntersectAABB(ray, node.bounds, tMin, closestHit.distance, tNear)) {
            continue;
        }

        if (node.IsLeaf()) {
            for (uint16_t i = 0; i < node.primitiveCount; ++i) {
                const uint32_t triIdx = node.primitivesOffset + i;
                const AcousticTriangle& tri = m_triangles[triIdx];

                float t, u, v;
                if (IntersectTriangle(ray, tri, tMin, closestHit.distance, t, u, v)) {
                    closestHit.distance = t;
                    closestHit.point = ray.PointAt(t);
                    closestHit.u = u;
                    closestHit.v = v;
                    closestHit.triangleIndex = triIdx;
                    closestHit.materialId = tri.materialId;
                    closestHit.normal = (glm::dot(ray.direction, tri.normal) < 0.0f) ? tri.normal : -tri.normal;
                }
            }
        } else {
            const uint32_t leftChild = nodeIdx + 1;
            const uint32_t rightChild = node.secondChildOffset;

            // Front-to-back traversal order: push further child first so nearer child is popped first
            if (stackPtr < 62) {
                if (ray.direction[node.axis] < 0.0f) {
                    stack[stackPtr++] = leftChild;
                    stack[stackPtr++] = rightChild;
                } else {
                    stack[stackPtr++] = rightChild;
                    stack[stackPtr++] = leftChild;
                }
            }
        }
    }

    return closestHit;
}

bool BVH::IntersectAny(const AcousticRay& ray, float tMin, float tMax) const {
    if (m_nodes.empty()) return false;

    uint32_t stack[64];
    int stackPtr = 0;
    stack[stackPtr++] = 0;

    while (stackPtr > 0) {
        const uint32_t nodeIdx = stack[--stackPtr];
        const BVHNode& node = m_nodes[nodeIdx];

        float tNear = 0.0f;
        if (!IntersectAABB(ray, node.bounds, tMin, tMax, tNear)) {
            continue;
        }

        if (node.IsLeaf()) {
            for (uint16_t i = 0; i < node.primitiveCount; ++i) {
                const AcousticTriangle& tri = m_triangles[node.primitivesOffset + i];
                float t, u, v;
                if (IntersectTriangle(ray, tri, tMin, tMax, t, u, v)) {
                    return true;
                }
            }
        } else {
            const uint32_t leftChild = nodeIdx + 1;
            const uint32_t rightChild = node.secondChildOffset;

            if (stackPtr < 62) {
                if (ray.direction[node.axis] < 0.0f) {
                    stack[stackPtr++] = leftChild;
                    stack[stackPtr++] = rightChild;
                } else {
                    stack[stackPtr++] = rightChild;
                    stack[stackPtr++] = leftChild;
                }
            }
        }
    }

    return false;
}

size_t BVH::GetLeafCount() const {
    size_t count = 0;
    for (const auto& node : m_nodes) {
        if (node.IsLeaf()) ++count;
    }
    return count;
}

uint32_t BVH::GetMaxDepth() const {
    if (m_nodes.empty()) return 0;

    struct DepthItem {
        uint32_t nodeIndex;
        uint32_t depth;
    };
    std::vector<DepthItem> stack;
    stack.reserve(64);
    stack.push_back({0, 1});

    uint32_t maxDepth = 1;
    while (!stack.empty()) {
        const DepthItem item = stack.back();
        stack.pop_back();

        maxDepth = std::max(maxDepth, item.depth);
        const auto& node = m_nodes[item.nodeIndex];
        if (!node.IsLeaf()) {
            stack.push_back({item.nodeIndex + 1, item.depth + 1});
            stack.push_back({node.secondChildOffset, item.depth + 1});
        }
    }
    return maxDepth;
}
