
#include "acoustics/AcousticSimulation.h"
#include "core/Camera.h"
#include "scene/LightEnvironment.h"

#include <glm/gtc/constants.hpp>
#include <glm/gtx/norm.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace acoustics {

namespace {

// ============================================
// POMOCNE MATEMATICKE FUNKCIJE
// ============================================

// Generisanje slucajnog broja iz intervala [0, 1).
float Rand(std::mt19937& rng) {
    return std::generate_canonical<float, 24>(rng);
}

// Normalizacija vektora uz rezervni smjer.
glm::vec3 SafeUnit(
    glm::vec3 v,
    glm::vec3 fallback = {0, 0, -1}
) {
    return glm::length2(v) > 1e-12f
        ? glm::normalize(v)
        : fallback;
}

// Konstrukcija dva okomita vektora.
// Koristi se pri uzorkovanju smjerova.
void Basis(
    glm::vec3 normal,
    glm::vec3& tangent,
    glm::vec3& bitangent
) {
    normal = SafeUnit(normal, {0, 1, 0});

    const glm::vec3 helper =
        std::abs(normal.z) < 0.9f
            ? glm::vec3(0, 0, 1)
            : glm::vec3(0, 1, 0);

    tangent = glm::normalize(
        glm::cross(helper, normal)
    );

    bitangent = glm::cross(normal, tangent);
}


// ============================================
// DIO 3.1 - EMISIJA AKUSTICKIH ZRAKA
// ============================================

// Generisanje nasumicnog smjera unutar konusa.
glm::vec3 ConeSample(
    glm::vec3 axis,
    float cosineMin,
    std::mt19937& rng
) {
    axis = SafeUnit(axis);

    glm::vec3 t, b;
    Basis(axis, t, b);

    const float z =
        cosineMin + (1.0f - cosineMin) * Rand(rng);

    const float phi =
        2.0f * glm::pi<float>() * Rand(rng);

    const float r =
        std::sqrt(std::max(0.0f, 1.0f - z*z));

    return SafeUnit(
        t * (r * std::cos(phi))
        + b * (r * std::sin(phi))
        + axis * z
    );
}

// Odabir smjera na osnovu tipa izvora.
glm::vec3 SampleDirection(
    const Source& s,
    std::mt19937& rng
) {
    if (s.pattern == Source::Pattern::Parallel) {
        return SafeUnit(s.direction);
    }

    if (s.pattern == Source::Pattern::Cone) {
        return ConeSample(
            s.direction,
            std::cos(
                glm::radians(
                    std::clamp(
                        s.coneHalfAngleDeg,
                        0.0f,
                        180.0f
                    )
                )
            ),
            rng
        );
    }

    return ConeSample({0, 1, 0}, -1.0f, rng);
}


// ============================================
// DIO 3.2 - SLABLJENJE ENERGIJE
// ============================================

// Apsorpcija energije tokom prolaska kroz zrak.
OctaveBands AtDistance(
    const OctaveBands& energy,
    float dist,
    const OctaveBands& air
) {
    OctaveBands result;

    for (size_t i = 0; i < OctaveBands::Count; ++i) {
        result[i] =
            energy[i] * std::exp(
                -std::max(0.0f, air[i])
                * std::max(0.0f, dist)
            );
    }

    return result;
}

// Najveca energija medju frekvencijskim opsezima.
float MaxEnergy(const OctaveBands& e) {
    return *std::max_element(
        e.values.begin(),
        e.values.end()
    );
}


// ============================================
// DIO 4.3 - DETEKCIJA ZRAKE NA PRIJEMNIKU
// ============================================

void CheckReceiver(
    const AcousticRay& ray,
    float segmentLength,
    const Receiver& receiver,
    const Settings& settings,
    Echogram& out,
    TraceStats& stats
) {
    for (int side = 0; side < 2; ++side) {

        const glm::vec3 ear =
            side == 0
                ? receiver.LeftEar()
                : receiver.RightEar();

        float nearT, farT;

        if (!IntersectSphere(
                ray,
                ear,
                receiver.earRadius,
                0.0f,
                segmentLength,
                nearT,
                farT
            )) {
            continue;
        }

        // Aproksimacija zaklanjanja uha glavom.
        if (receiver.headShadow) {
            float headNear, headFar;

            if (IntersectSphere(
                    ray,
                    receiver.headCenter,
                    receiver.headRadius,
                    0.0f,
                    segmentLength,
                    headNear,
                    headFar
                ) && headNear + 1e-4f < nearT) {
                continue;
            }
        }

        // Duzina presjeka zrake sa sferom uha.
        const float chord = farT - nearT;

        const float effectiveChord =
            std::max(0.0f, chord);

        if (effectiveChord <= 0.0f) {
            continue;
        }

        // Tezinski faktor na osnovu duzine tetive.
        const float weight = std::min(
            1.0f,
            effectiveChord / (2.0f * receiver.earRadius)
        );

        // Vrijeme dolaska zvuka.
        const float arrival =
            (ray.distanceTraveled + nearT)
            / settings.speedOfSound;

        // Upis energije u ehogram.
        out.Deposit(
            side == 0,
            arrival,
            AtDistance(
                ray.energy,
                nearT,
                settings.airAbsorption
            ) * weight
        );

        ++stats.earHits;
    }
}

} // anonymous namespace


// ============================================
// DIO 3.1 - PRETVARANJE SVJETALA U IZVORE
// ============================================

// Point light -> izvor u svim smjerovima.
Source FromLight(const PointLight& l) {
    Source s;

    s.pattern = Source::Pattern::Omni;
    s.position = l.position;

    s.power = OctaveBands(
        l.enabled
            ? std::max(0.0f, l.intensity)
            : 0.0f
    );

    return s;
}

// Spot light -> konusno usmjeren izvor.
Source FromLight(const SpotLight& l) {
    Source s;

    s.pattern = Source::Pattern::Cone;
    s.position = l.position;
    s.direction = SafeUnit(l.direction);

    s.coneHalfAngleDeg =
        std::max(0.0f, l.cone.outerDeg);

    s.power = OctaveBands(
        l.enabled
            ? std::max(0.0f, l.intensity)
            : 0.0f
    );

    return s;
}

// Directional light -> paralelne zrake.
Source FromLight(
    const DirectionalLight& l,
    const glm::vec3& emitterPosition
) {
    Source s;

    s.pattern = Source::Pattern::Parallel;
    s.position = emitterPosition;
    s.direction = SafeUnit(l.direction);

    s.power = OctaveBands(
        l.enabled
            ? std::max(0.0f, l.intensity)
            : 0.0f
    );

    return s;
}

// Pretvaranje svih aktivnih svjetala u izvore.
std::vector<Source> FromLights(
    const LightEnvironment& lights,
    const glm::vec3& directionalOrigin
) {
    std::vector<Source> result;

    const auto& directional =
        lights.GetDirectionalLight();

    if (directional.enabled) {
        result.push_back(
            FromLight(directional, directionalOrigin)
        );
    }

    for (const auto& light : lights.GetPointLights()) {
        if (light.enabled) {
            result.push_back(FromLight(light));
        }
    }

    for (const auto& light : lights.GetSpotLights()) {
        if (light.enabled) {
            result.push_back(FromLight(light));
        }
    }

    return result;
}


// ============================================
// DIO 4.1 I 4.2 - BINAURALNI PRIJEMNIK
// ============================================

// Pozicija lijevog uha.
glm::vec3 Receiver::LeftEar() const {
    return headCenter
        - SafeUnit(right, {1, 0, 0})
        * (earSpacing * 0.5f);
}

// Pozicija desnog uha.
glm::vec3 Receiver::RightEar() const {
    return headCenter
        + SafeUnit(right, {1, 0, 0})
        * (earSpacing * 0.5f);
}

// Prijemnik preuzima polozaj kamere.
void Receiver::FollowCamera(const Camera& camera) {
    headCenter = camera.GetPosition();

    right = SafeUnit(
        camera.GetRight(),
        {1, 0, 0}
    );
}


// ============================================
// DIO 4.4 - EHOGRAM
// ============================================

// Kreiranje vremenskih intervala.
Echogram::Echogram(
    float durationSec,
    float binsPerSecond
)
    : sampleRate(binsPerSecond) {

    if (!(durationSec > 0.0f && binsPerSecond > 0.0f)) {
        throw std::invalid_argument(
            "Positive echogram duration and bin rate required"
        );
    }

    const size_t count = static_cast<size_t>(
        std::ceil(durationSec * binsPerSecond)
    );

    left.resize(count);
    right.resize(count);
}

// Brisanje prethodnih rezultata.
void Echogram::Clear() {
    std::fill(
        left.begin(),
        left.end(),
        OctaveBands::Zero()
    );

    std::fill(
        right.begin(),
        right.end(),
        OctaveBands::Zero()
    );
}

// Dodavanje energije u vremenski interval.
void Echogram::Deposit(
    bool isLeft,
    float arrivalTime,
    const OctaveBands& energy
) {
    if (!std::isfinite(arrivalTime)
        || arrivalTime < 0.0f) {
        return;
    }

    const size_t index = static_cast<size_t>(
        arrivalTime * sampleRate
    );

    auto& bins = isLeft ? left : right;

    if (index < bins.size()) {
        bins[index] += energy;
    }
}


// ============================================
// DIO 3 - GLAVNI ALGORITAM SIMULACIJE
// ============================================

TraceStats Simulator::Run(
    const std::vector<Source>& sources,
    const Receiver& receiver,
    Echogram& echogram,
    const Settings& cfg
) const {

    // Provjera parametara.
    if (cfg.raysPerSource == 0
        || cfg.speedOfSound <= 0.0f
        || cfg.maxDuration <= 0.0f
        || cfg.maxSegmentDistance <= 0.0f
        || receiver.earRadius <= 0.0f) {

        throw std::invalid_argument(
            "Invalid acoustic tracing settings"
        );
    }

    echogram.Clear();

    TraceStats stats;

    std::mt19937 rng(cfg.randomSeed);

    // Prolazak kroz sve izvore zvuka.
    for (const auto& source : sources) {

        // Emitovanje zraka za svaki izvor.
        for (uint32_t i = 0;
             i < cfg.raysPerSource;
             ++i) {

            ++stats.emitted;

            AcousticRay ray(
                source.position,
                SampleDirection(source, rng),
                source.power
                    / static_cast<float>(cfg.raysPerSource)
            );

            // Pracenje zrake kroz refleksije.
            for (;;) {

                // Zaustavljanje zbog premale energije
                // ili prekoracenog broja refleksija.
                if (MaxEnergy(ray.energy) < cfg.minEnergy
                    || ray.bounceCount > cfg.maxBounces) {
                    break;
                }

                // Preostala dozvoljena putanja.
                const float remaining =
                    cfg.speedOfSound * cfg.maxDuration
                    - ray.distanceTraveled;

                if (remaining <= 0.0f) {
                    break;
                }

                const float maxT = std::min(
                    remaining,
                    cfg.maxSegmentDistance
                );

                // BVH presjek sa geometrijom.
                const AcousticHit hit = bvh_.Intersect(
                    ray,
                    cfg.rayOffset,
                    maxT
                );

                const float segment =
                    hit.HasHit() ? hit.distance : maxT;

                ++stats.segments;

                // Provjera dolaska do uha.
                CheckReceiver(
                    ray,
                    segment,
                    receiver,
                    cfg,
                    echogram,
                    stats
                );

                // Napredovanje zrake.
                ray.Advance(segment);

                // Slabljenje energije u zraku.
                ray.energy = AtDistance(
                    ray.energy,
                    segment,
                    cfg.airAbsorption
                );

                // Ako nema povrsine, zraka napusta scenu.
                if (!hit.HasHit()) {
                    ++stats.escaped;
                    break;
                }

                ++stats.surfaceHits;

                if (ray.bounceCount >= cfg.maxBounces) {
                    break;
                }

                // Materijal pogodjene povrsine.
                const AcousticMaterial& material =
                    materials_.GetMaterial(hit.materialId);

                // Dio energije koji se reflektuje.
                ray.energy *= material.ReflectedEnergy();

                // Normalni vektor povrsine.
                const glm::vec3 normal =
                    glm::dot(hit.normal, ray.direction) < 0.0f
                        ? hit.normal
                        : -hit.normal;

                // Spekularna refleksija.
                const glm::vec3 specular = glm::reflect(
                    ray.direction,
                    SafeUnit(normal, {0, 1, 0})
                );

                // Izbor izmedju difuzne
                // i spekularne refleksije.
                const bool diffuse =
                    Rand(rng) < std::clamp(
                        material.scattering,
                        0.0f,
                        1.0f
                    );

                const glm::vec3 next =
                    diffuse
                        ? ConeSample(normal, 0.0f, rng)
                        : specular;

                // Ruski rulet.
                if (ray.bounceCount
                    >= cfg.russianRouletteStart) {

                    const float survival = std::clamp(
                        MaxEnergy(ray.energy)
                        / std::max(
                            1e-6f,
                            MaxEnergy(
                                source.power
                                / static_cast<float>(
                                    cfg.raysPerSource
                                )
                            )
                        ),
                        0.1f,
                        0.95f
                    );

                    if (Rand(rng) > survival) {
                        break;
                    }

                    ray.energy *= 1.0f / survival;
                }

                // Mali pomak od povrsine.
                ray.origin += SafeUnit(
                    normal,
                    {0, 1, 0}
                ) * cfg.rayOffset;

                // Novi smjer zrake.
                ray.SetDirection(next);

                ++ray.bounceCount;
            }
        }
    }

    return stats;
}

} // namespace acoustics
