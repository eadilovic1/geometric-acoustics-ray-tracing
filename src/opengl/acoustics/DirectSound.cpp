
#include "acoustics/DirectSound.h"

#include <algorithm>
#include <cmath>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

namespace acoustics {

namespace {

float DirectGeometricFactor(float distance) {
    const float r = std::max(distance, 0.01f);
    return 1.0f / (4.0f * glm::pi<float>() * r * r);
}

OctaveBands ApplyDirectAirAbsorption(
    const OctaveBands& energy,
    float distance,
    const OctaveBands& airAbsorption
) {
    OctaveBands result = energy;
    const float d = std::max(0.0f, distance);

    for (size_t b = 0; b < OctaveBands::Count; ++b) {
        result[b] *= std::exp(
            -std::max(0.0f, airAbsorption[b]) * d
        );
    }

    return result;
}

OctaveBands CalculateHeadShadow(
    const glm::vec3& sourcePosition,
    const glm::vec3& earPosition,
    const Receiver& receiver
) {
    OctaveBands factor = OctaveBands::One();

    if (!receiver.headShadow || receiver.headRadius <= 0.0f) {
        return factor;
    }

    const glm::vec3 delta = earPosition - sourcePosition;
    const float distance = glm::length(delta);

    if (distance <= 1e-6f) {
        return factor;
    }

    const glm::vec3 direction = delta / distance;
    const glm::vec3 oc = sourcePosition - receiver.headCenter;

    const float b = glm::dot(oc, direction);
    const float c = glm::dot(oc, oc)
                  - receiver.headRadius * receiver.headRadius;

    const float discriminant = b * b - c;

    if (discriminant < 0.0f) {
        return factor;
    }

    const float sqrtD = std::sqrt(discriminant);
    const float nearT = -b - sqrtD;
    const float farT = -b + sqrtD;

    // Presjek mora postojati izmedju izvora i uha.
    const bool blocked =
        farT > 0.0f &&
        nearT < distance - 1e-4f &&
        nearT < farT;

    if (blocked) {
        for (size_t b = 0; b < OctaveBands::Count; ++b) {
            factor[b] = std::clamp(
                1.0f - receiver.headMaterialAbsorption[b],
                0.0f,
                1.0f
            );
        }
    }

    return factor;
}

void DepositForEar(
    bool isLeft,
    const glm::vec3& earPosition,
    const Source& source,
    const Receiver& receiver,
    const BVH& bvh,
    const Settings& settings,
    Echogram& echogram
) {
    const glm::vec3 delta = earPosition - source.position;
    const float distance = glm::length(delta);

    if (distance <= settings.rayOffset ||
        settings.speedOfSound <= 0.0f ||
        receiver.earRadius <= 0.0f) {
        return;
    }

    const glm::vec3 direction = delta / distance;

    AcousticRay visibilityRay(source.position, direction);

    if (bvh.IntersectAny(
            visibilityRay,
            settings.rayOffset,
            distance - settings.rayOffset
        )) {
        return;
    }

    const float arrivalTime = distance / settings.speedOfSound;

    if (arrivalTime >= settings.maxDuration) {
        return;
    }

    
    // Procjena energijske gustine kompatibilna sa
    // Monte Carlo modelom sfernog prijemnika.
    const float earVolume =
        (4.0f / 3.0f) *
        glm::pi<float>() *
        receiver.earRadius *
        receiver.earRadius *
        receiver.earRadius;

    const float geometricFactor = DirectGeometricFactor(distance);

    // Ocekivana duzina presjeka zrake sa sferom je 4r/3.
    const float meanChord =
        (4.0f / 3.0f) * receiver.earRadius;

    const float depositScale =
        geometricFactor *
        glm::pi<float>() *
        receiver.earRadius *
        receiver.earRadius *
        meanChord /
        (settings.speedOfSound * earVolume);

    OctaveBands energy = source.power * depositScale;


    energy = ApplyDirectAirAbsorption(
        energy,
        distance,
        settings.airAbsorption
    );

    energy *= CalculateHeadShadow(
        source.position,
        earPosition,
        receiver
    );

    echogram.Deposit(isLeft, arrivalTime, energy);
}

} // anonymous namespace

void DepositDirectSound(
    const Source& source,
    const Receiver& receiver,
    const BVH& bvh,
    const Settings& settings,
    Echogram& echogram
) {
    // Deterministicki model trenutno podrzava Omni.
    if (source.pattern != Source::Pattern::Omni) {
        return;
    }

    DepositForEar(
        true,
        receiver.LeftEar(),
        source,
        receiver,
        bvh,
        settings,
        echogram
    );

    DepositForEar(
        false,
        receiver.RightEar(),
        source,
        receiver,
        bvh,
        settings,
        echogram
    );
}

} // namespace acoustics
