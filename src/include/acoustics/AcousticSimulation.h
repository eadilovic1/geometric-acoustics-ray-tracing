
#pragma once

#include "acoustics/BVH.h"
#include "acoustics/AcousticMaterial.h"
#include "scene/Light.h"

#include <glm/glm.hpp>
#include <array>
#include <cstdint>
#include <random>
#include <vector>

class Camera;
class LightEnvironment;

namespace acoustics {

// Energija po oktavnim opsezima.
// Ovo nisu uzorci zvucnog pritiska.
struct Source {
    enum class Pattern {
        Omni,
        Cone,
        Parallel
    };

    Pattern pattern = Pattern::Omni;

    glm::vec3 position{0.0f};

    glm::vec3 direction{0.0f, 0.0f, -1.0f};

    float coneHalfAngleDeg = 20.0f;

    OctaveBands power{1.0f};
};

// Pretvaranje svjetlosnih izvora u akusticke izvore.
// Ovo je demonstracijsko mapiranje,
// ne fizicka kalibracija svjetla u zvuk.
Source FromLight(const PointLight& light);

Source FromLight(const SpotLight& light);

Source FromLight(
    const DirectionalLight& light,
    const glm::vec3& emitterPosition
);

std::vector<Source> FromLights(
    const LightEnvironment& lights,
    const glm::vec3& directionalOrigin
);


// DIO 4: binauralni prijemnik
struct Receiver {
    glm::vec3 headCenter{0.0f};

    glm::vec3 right{1.0f, 0.0f, 0.0f};

    float earSpacing = 0.18f;

    float earRadius = 0.09f;

    float headRadius = 0.085f;

    // Pojednostavljena geometrijska sjena glave.
    bool headShadow = true;

    [[nodiscard]] glm::vec3 LeftEar() const;

    [[nodiscard]] glm::vec3 RightEar() const;

    void FollowCamera(const Camera& camera);
};


// DIO 4: ehogram
struct Echogram {
    float sampleRate = 1000.0f;

    std::vector<OctaveBands> left;

    std::vector<OctaveBands> right;

    Echogram(
        float durationSec = 2.0f,
        float binsPerSecond = 1000.0f
    );

    void Clear();

    void Deposit(
        bool isLeft,
        float arrivalTime,
        const OctaveBands& energy
    );
};


// DIO 3: postavke simulacije
struct Settings {
    uint32_t raysPerSource = 10000;

    uint32_t maxBounces = 12;

    float speedOfSound = 343.0f;

    float maxDuration = 2.0f;

    float maxSegmentDistance = 1000.0f;

    float minEnergy = 1e-8f;

    float rayOffset = 0.001f;

    uint32_t russianRouletteStart = 4;

    // Priblizni koeficijenti slabljenja po metru.
    OctaveBands airAbsorption{
        0.0001f,
        0.0002f,
        0.0004f,
        0.0008f,
        0.002f,
        0.006f,
        0.018f
    };

    uint32_t randomSeed = 42;
};


// Statistika izvrsavanja simulacije
struct TraceStats {
    uint64_t emitted = 0;

    uint64_t segments = 0;

    uint64_t surfaceHits = 0;

    uint64_t earHits = 0;

    uint64_t escaped = 0;
};


// Glavna klasa akusticke simulacije
class Simulator {
public:
    Simulator(
        const BVH& bvh,
        const AcousticMaterialLibrary& materials
    )
        : bvh_(bvh),
          materials_(materials) {}

    TraceStats Run(
        const std::vector<Source>& sources,
        const Receiver& receiver,
        Echogram& echogram,
        const Settings& settings = {}
    ) const;

private:
    const BVH& bvh_;

    const AcousticMaterialLibrary& materials_;
};

} // namespace acoustics
