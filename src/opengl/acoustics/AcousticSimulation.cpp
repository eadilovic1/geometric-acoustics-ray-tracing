#include "acoustics/AcousticSimulation.h"
#include "core/Camera.h"
#include "scene/LightEnvironment.h"

#include <glm/gtc/constants.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace acoustics {

namespace {

// ===========================================================================
// POMOĆNE MATEMATIČKE FUNKCIJE I UZORKOVANJE
// ===========================================================================

/// Generisanje pseudo-slučajnog broja u intervalu [0, 1)
float Rand(std::mt19937& rng) {
    return std::generate_canonical<float, 24>(rng);
}

/// Normalizacija vektora uz siguran fallback u slučaju nula-vektora
glm::vec3 SafeUnit(glm::vec3 v, glm::vec3 fallback = {0, 0, -1}) {
    const float len2 = glm::dot(v, v);
    return len2 > 1e-12f ? v * (1.0f / std::sqrt(len2)) : fallback;
}

/// Konstrukcija desnog ortonormiranog koordinatnog sistema (t, b, n) oko zadatog jediničnog vektora n
void Basis(const glm::vec3& n, glm::vec3& t, glm::vec3& b) {
    const glm::vec3 helper = (std::abs(n.z) < 0.999f) ? glm::vec3(0, 0, 1) : glm::vec3(1, 0, 0);
    t = glm::normalize(glm::cross(n, helper));
    b = glm::cross(n, t);
}

// ===========================================================================
// DIO 3.1 - EMISIJA ZRAKA (PointLight, SpotLight, DirectionalLight)
// ===========================================================================

/// Fibonaccijeva sferna spirala za tačkasti svesmjerni izvor (PointLight).
/// Garantuje egzaktno uniformnu, determinističku i nisko-diskrepantnu raspodjelu smjerova po sferi
/// bez polarnog grupisanja (prema formuli iz plana 3.1 i odjeljka 8.3 u PRS_Projekat.pdf).
glm::vec3 SampleFibonacciSphere(uint32_t i, uint32_t N) {
    const float kGoldenRatio = 1.0f + std::sqrt(5.0f); // 1 + sqrt(5)
    const float z = 1.0f - (2.0f * (static_cast<float>(i) + 0.5f)) / static_cast<float>(N);
    const float sinTheta = std::sqrt(std::max(0.0f, 1.0f - z * z));
    const float phi = glm::pi<float>() * kGoldenRatio * static_cast<float>(i);
    return glm::vec3(sinTheta * std::cos(phi), sinTheta * std::sin(phi), z);
}

/// Usmjerena konusna emisija za SpotLight sa kardioidnim / glatkim opadanjem.
/// Zrake se emituju unutar konusa [0, outerAngle], a intenzitet glatko opada između innerAngle i outerAngle.
glm::vec3 SampleSpotCone(
    const Source& s,
    std::mt19937& rng,
    float& outFalloffFactor
) {
    const glm::vec3 axis = SafeUnit(s.direction, {0, -1, 0});
    glm::vec3 t, b;
    Basis(axis, t, b);

    const float outerRad = glm::radians(std::clamp(s.coneOuterAngleDeg, 0.1f, 180.0f));
    const float innerRad = glm::radians(std::clamp(s.coneInnerAngleDeg, 0.0f, s.coneOuterAngleDeg));

    const float cosOuter = std::cos(outerRad);
    const float xi1 = Rand(rng);
    const float xi2 = Rand(rng);

    // Uniformno uzorkovanje prostornog ugla unutar konusa
    const float z = cosOuter + (1.0f - cosOuter) * xi1;
    const float sinTheta = std::sqrt(std::max(0.0f, 1.0f - z * z));
    const float phi = 2.0f * glm::pi<float>() * xi2;

    const glm::vec3 dir = SafeUnit(t * (sinTheta * std::cos(phi)) + b * (sinTheta * std::sin(phi)) + axis * z);

    // Ugao od ose i kardioidni profil opadanja prema rubu konusa
    const float theta = std::acos(std::clamp(z, -1.0f, 1.0f));
    if (theta <= innerRad) {
        outFalloffFactor = 1.0f;
    } else if (theta < outerRad && outerRad > innerRad) {
        const float u = (theta - innerRad) / (outerRad - innerRad);
        // Glatko kosinusno/kardioidno opadanje
        outFalloffFactor = 0.5f * (1.0f + std::cos(glm::pi<float>() * u));
    } else {
        outFalloffFactor = 0.0f;
    }

    return dir;
}

/// Emisija paralelnih zraka ravnog talasa (DirectionalLight) sa površine diska aperture.
/// Sve zrake imaju identičan smjer, a ishodišta su ravnomjerno raspoređena po disku.
void SampleParallelRay(
    const Source& s,
    uint32_t i,
    uint32_t N,
    glm::vec3& outOrigin,
    glm::vec3& outDirection
) {
    outDirection = SafeUnit(s.direction, {0, -1, 0});
    glm::vec3 t, b;
    Basis(outDirection, t, b);

    // Vogel / Fibonacci raspodjela tačaka po disku aperture
    const float kGoldenRatio = 1.0f + std::sqrt(5.0f);
    const float r = s.emitterRadius * std::sqrt((static_cast<float>(i) + 0.5f) / static_cast<float>(N));
    const float phi = glm::pi<float>() * kGoldenRatio * static_cast<float>(i);

    outOrigin = s.position + t * (r * std::cos(phi)) + b * (r * std::sin(phi));
}

// ===========================================================================
// DIO 3.3 - DIFUZNO RASIPANJE (Lambertov kosinusni zakon)
// ===========================================================================

/// Kosinusno ponderisano uzorkovanje polusfere prema Lambertovom kosinusnom zakonu.
/// Vjerovatnoća smjera je proporcionalna sa cos(theta) (odjeljak 6.3 i stavka 3.3).
glm::vec3 SampleCosineHemisphere(const glm::vec3& normal, std::mt19937& rng) {
    glm::vec3 t, b;
    Basis(normal, t, b);

    const float xi1 = Rand(rng);
    const float xi2 = Rand(rng);

    // Lambertov kosinusni zakon: z = cos(theta) = sqrt(xi1)
    const float r = std::sqrt(xi1);
    const float phi = 2.0f * glm::pi<float>() * xi2;
    const float z = std::sqrt(std::max(0.0f, 1.0f - xi1));

    return SafeUnit(t * (r * std::cos(phi)) + b * (r * std::sin(phi)) + normal * z);
}

// ===========================================================================
// DIO 3.2 - SLABLJENJE ENERGIJE U ZRAKU (ISO 9613-1)
// ===========================================================================

/// Eksponencijalno slabljenje energije u zraku: E = E0 * exp(-m_b * dist)
OctaveBands AtDistance(
    const OctaveBands& energy,
    float dist,
    const OctaveBands& air
) {
    OctaveBands result;
    const float d = std::max(0.0f, dist);
    for (size_t i = 0; i < OctaveBands::Count; ++i) {
        result[i] = energy[i] * std::exp(-std::max(0.0f, air[i]) * d);
    }
    return result;
}

/// Najveća energija među frekvencijskim opsezima
float MaxEnergy(const OctaveBands& e) {
    return *std::max_element(e.values.begin(), e.values.end());
}

// ===========================================================================
// DIO 4.3 - BINAURALNA DETEKCIJA TETIVE I ENERGIJSKI UPIS
// ===========================================================================

void CheckReceiver(
    const AcousticRay& ray,
    float segmentLength,
    const Receiver& receiver,
    const Settings& settings,
    Echogram& out,
    TraceStats& stats
) {
    for (int side = 0; side < 2; ++side) {
        const bool isLeft = (side == 0);
        const glm::vec3 ear = isLeft ? receiver.LeftEar() : receiver.RightEar();

        float nearT = 0.0f;
        float farT  = 0.0f;

        // Analitički presjek segmenta zrake sa sferom uha
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

        // Dužina tetive kroz sferu uha: l = farT - nearT
        const float chord = farT - nearT;
        if (chord <= 1e-7f) {
            continue;
        }

        ++stats.earHits;

        // Sredina tetive duž segmenta
        const float midT = (nearT + farT) * 0.5f;

        // Provjera fizikalnog ILD zasjenjenja sferom glave (koža/kosa)
        OctaveBands shadowFactor = OctaveBands::One();
        if (receiver.headShadow) {
            float headNear = 0.0f;
            float headFar  = 0.0f;

            if (IntersectSphere(
                    ray,
                    receiver.headCenter,
                    receiver.headRadius,
                    0.0f,
                    segmentLength,
                    headNear,
                    headFar
                ) && headNear + 1e-4f < nearT) {
                
                ++stats.headOcclusions;
                // Frekvencijski zavisna apsorpcija tkiva glave (koža i kosa):
                // Niske frekvencije prolaze (difrakcija), visoke se guše
                for (size_t b = 0; b < OctaveBands::Count; ++b) {
                    const float trans = 1.0f - receiver.headMaterialAbsorption[b];
                    shadowFactor[b] = std::clamp(trans, 0.0f, 1.0f);
                }
            }
        }

        // Energija na sredini tetive nakon prolaska kroz zrak
        const OctaveBands arrivingEnergy = AtDistance(
            ray.energy,
            midT,
            settings.airAbsorption
        ) * shadowFactor;

        // Zapremina detektora sfere uha: Vd = (4/3) * pi * rd^3
        const float Vd = (4.0f / 3.0f) * glm::pi<float>() * std::pow(receiver.earRadius, 3.0f);

        // Fizikalna gustina energije prema teoretskoj osnovi:
        // Delta E_b = (E_b * l) / (c * Vd)
        const float depositScale = chord / (settings.speedOfSound * Vd);
        const OctaveBands depositedEnergy = arrivingEnergy * depositScale;

        // Vremenski interval boravka unutar sfere uha: [tStart, tEnd]
        const float tStart = (ray.distanceTraveled + nearT) / settings.speedOfSound;
        const float tEnd   = (ray.distanceTraveled + farT)  / settings.speedOfSound;

        // Kontinuirani raspodijeljeni upis tetive u vremenske binove ehograma (48 kHz)
        out.DepositChordSpan(isLeft, tStart, tEnd, depositedEnergy);
    }
}

} // anonymous namespace


// ===========================================================================
// DIO 3.1 - PRETVARANJE SVJETALA U AKUSTIČKE IZVORE
// ===========================================================================

Source FromLight(const PointLight& l) {
    Source s;
    s.pattern = Source::Pattern::Omni;
    s.position = l.position;
    s.power = OctaveBands(l.enabled ? std::max(0.0f, l.intensity) : 0.0f);
    return s;
}

Source FromLight(const SpotLight& l) {
    Source s;
    s.pattern = Source::Pattern::Cone;
    s.position = l.position;
    s.direction = SafeUnit(l.direction, {0, -1, 0});
    s.coneInnerAngleDeg = std::max(0.0f, l.cone.innerDeg);
    s.coneOuterAngleDeg = std::max(s.coneInnerAngleDeg, l.cone.outerDeg);
    s.power = OctaveBands(l.enabled ? std::max(0.0f, l.intensity) : 0.0f);
    return s;
}

Source FromLight(
    const DirectionalLight& l,
    const glm::vec3& emitterPosition,
    float beamRadius
) {
    Source s;
    s.pattern = Source::Pattern::Parallel;
    s.position = emitterPosition;
    s.direction = SafeUnit(l.direction, {0, -1, 0});
    s.emitterRadius = std::max(0.1f, beamRadius);
    s.power = OctaveBands(l.enabled ? std::max(0.0f, l.intensity) : 0.0f);
    return s;
}

std::vector<Source> FromLights(
    const LightEnvironment& lights,
    const glm::vec3& directionalOrigin,
    float directionalBeamRadius
) {
    std::vector<Source> result;

    if (lights.HasDirectionalLight()) {
        const auto& directional = lights.GetDirectionalLight();
        if (directional.enabled) {
            result.push_back(FromLight(directional, directionalOrigin, directionalBeamRadius));
        }
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


// ===========================================================================
// DIO 4.1 & 4.2 - BINAURALNI PRIJEMNIK
// ===========================================================================

glm::vec3 Receiver::LeftEar() const {
    return headCenter - SafeUnit(right, {1, 0, 0}) * (earSpacing * 0.5f);
}

glm::vec3 Receiver::RightEar() const {
    return headCenter + SafeUnit(right, {1, 0, 0}) * (earSpacing * 0.5f);
}

void Receiver::FollowCamera(const Camera& camera, float eyeHeightOffset) {
    if (camera.GetMode() == CameraMode::ThirdPerson) {
        // U ThirdPerson modu uši su vezane za glavu lika (centar orbite + visina očiju)
        headCenter = camera.GetOrbitTarget() + SafeUnit(camera.GetUp(), {0, 1, 0}) * eyeHeightOffset;
    } else {
        // U FreeFly i FirstPerson modu uši prate stvarnu poziciju kamere
        headCenter = camera.GetPosition();
    }

    right   = SafeUnit(camera.GetRight(),   {1, 0, 0});
    forward = SafeUnit(camera.GetForward(), {0, 0, -1});
    up      = SafeUnit(camera.GetUp(),      {0, 1, 0});
}


// ===========================================================================
// DIO 4.4 - DVOKANALNI EHOGRAM
// ===========================================================================

Echogram::Echogram(
    float durationSec,
    float binsPerSecond
)
    : sampleRate(binsPerSecond) {

    if (!(durationSec > 0.0f && binsPerSecond > 0.0f)) {
        throw std::invalid_argument("Echogram duration and sample rate must be strictly positive");
    }

    const size_t count = static_cast<size_t>(std::ceil(durationSec * binsPerSecond));
    left.resize(count, OctaveBands::Zero());
    right.resize(count, OctaveBands::Zero());
}

void Echogram::Clear() {
    std::fill(left.begin(), left.end(), OctaveBands::Zero());
    std::fill(right.begin(), right.end(), OctaveBands::Zero());
}

void Echogram::Deposit(
    bool isLeft,
    float arrivalTime,
    const OctaveBands& energy
) {
    if (!std::isfinite(arrivalTime) || arrivalTime < 0.0f) {
        return;
    }

    const size_t index = static_cast<size_t>(arrivalTime * sampleRate);
    auto& bins = isLeft ? left : right;

    if (index < bins.size()) {
        bins[index] += energy;
    }
}

void Echogram::DepositByIndex(
    bool isLeft,
    size_t binIndex,
    const OctaveBands& energy
) {
    auto& bins = isLeft ? left : right;
    if (binIndex < bins.size()) {
        bins[binIndex] += energy;
    }
}

void Echogram::DepositChordSpan(
    bool isLeft,
    float tStart,
    float tEnd,
    const OctaveBands& totalEnergy
) {
    if (!std::isfinite(tStart) || !std::isfinite(tEnd) || tEnd <= tStart || tStart < 0.0f) {
        return;
    }

    auto& bins = isLeft ? left : right;
    if (bins.empty()) return;

    const float duration = tEnd - tStart;
    if (duration <= 1e-9f) {
        Deposit(isLeft, (tStart + tEnd) * 0.5f, totalEnergy);
        return;
    }

    const int64_t kStart = static_cast<int64_t>(std::floor(tStart * sampleRate));
    const int64_t kEnd   = static_cast<int64_t>(std::floor(tEnd * sampleRate));

    // Ako čitava tetiva upada u jedan bin
    if (kStart == kEnd) {
        if (kStart >= 0 && static_cast<size_t>(kStart) < bins.size()) {
            bins[static_cast<size_t>(kStart)] += totalEnergy;
        }
        return;
    }

    // Ako tetiva presijeca granice više binova, energija se raspodjeljuje
    // proporcionalno vremenu provedenom u svakom binu (potpuno anti-aliasing)
    const float invDuration = 1.0f / duration;
    const float invSampleRate = 1.0f / sampleRate;

    const int64_t clampedStart = std::max<int64_t>(0, kStart);
    const int64_t clampedEnd = std::min<int64_t>(static_cast<int64_t>(bins.size() - 1), kEnd);

    for (int64_t k = clampedStart; k <= clampedEnd; ++k) {
        const float binT0 = static_cast<float>(k) * invSampleRate;
        const float binT1 = static_cast<float>(k + 1) * invSampleRate;

        const float overlap0 = std::max(tStart, binT0);
        const float overlap1 = std::min(tEnd, binT1);

        if (overlap1 > overlap0) {
            const float weight = (overlap1 - overlap0) * invDuration;
            bins[static_cast<size_t>(k)] += totalEnergy * weight;
        }
    }
}


// ===========================================================================
// DIO 3 - GLAVNI ALGORITAM SIMULACIJE (Simulator::Run)
// ===========================================================================

TraceStats Simulator::Run(
    const std::vector<Source>& sources,
    const Receiver& receiver,
    Echogram& echogram,
    const Settings& cfg
) const {
    if (cfg.raysPerSource == 0
        || cfg.speedOfSound <= 0.0f
        || cfg.maxDuration <= 0.0f
        || cfg.maxSegmentDistance <= 0.0f
        || receiver.earRadius <= 0.0f) {
        throw std::invalid_argument("Invalid acoustic simulation settings");
    }

    echogram.Clear();

    TraceStats stats;
    std::mt19937 rng(cfg.randomSeed);

    for (const auto& source : sources) {
        // Početna energija po zraci za ovaj izvor: E0 = power / N
        const OctaveBands rayInitialEnergy = source.power / static_cast<float>(cfg.raysPerSource);
        const float initialMaxEnergy = MaxEnergy(rayInitialEnergy);

        if (initialMaxEnergy <= 0.0f) {
            continue; // Ugašen ili izvor nulte snage
        }

        for (uint32_t i = 0; i < cfg.raysPerSource; ++i) {
            ++stats.emitted;

            glm::vec3 rayOrigin = source.position;
            glm::vec3 rayDirection{0.0f, 0.0f, -1.0f};
            OctaveBands rayEnergy = rayInitialEnergy;

            // 1. Emisija u zavisnosti od tipa izvora
            if (source.pattern == Source::Pattern::Omni) {
                // Tačkasti izvor: Fibonaccijeva sferna spirala (minimalna varijansa)
                rayDirection = SampleFibonacciSphere(i, cfg.raysPerSource);
            } else if (source.pattern == Source::Pattern::Cone) {
                // Spot konus sa kardioidnim opadanjem ka rubu
                float falloff = 1.0f;
                rayDirection = SampleSpotCone(source, rng, falloff);
                rayEnergy = rayEnergy * falloff;
            } else if (source.pattern == Source::Pattern::Parallel) {
                // Ravni talas: paralelne zrake sa diska aperture
                SampleParallelRay(source, i, cfg.raysPerSource, rayOrigin, rayDirection);
            }

            AcousticRay ray(rayOrigin, rayDirection, rayEnergy);

            // 2. Praćenje zrake kroz višestruke refleksije
            for (;;) {
                // Zaustavljanje zbog prevelikog broja odskoka
                if (ray.bounceCount > cfg.maxBounces) {
                    break;
                }

                // Preostala dozvoljena putanja do maxDuration
                const float remainingDistance =
                    cfg.speedOfSound * cfg.maxDuration - ray.distanceTraveled;

                if (remainingDistance <= 0.0f) {
                    break;
                }

                const float maxT = std::min(remainingDistance, cfg.maxSegmentDistance);

                // BVH presjek sa geometrijom scene
                const AcousticHit hit = bvh_.Intersect(ray, cfg.rayOffset, maxT);
                const float segment = hit.HasHit() ? hit.distance : maxT;

                ++stats.segments;

                // Provjera presjeka sa binauralnim prijemnikom (uši i glava)
                CheckReceiver(ray, segment, receiver, cfg, echogram, stats);

                // Napredovanje zrake duž segmenta
                ray.Advance(segment);

                // Fizičko slabljenje u zraku duž segmenta (ISO 9613-1)
                ray.energy = AtDistance(ray.energy, segment, cfg.airAbsorption);

                // Ako nema presjeka sa geometrijom, zraka napušta scenu
                if (!hit.HasHit()) {
                    ++stats.escaped;
                    break;
                }

                ++stats.surfaceHits;

                // Ako smo dostigli maksimalan broj odskoka, završavamo
                if (ray.bounceCount >= cfg.maxBounces) {
                    break;
                }

                // Interakcija sa materijalom površine
                const AcousticMaterial& material = materials_.GetMaterial(hit.materialId);

                // 1. Apsorpcija: energija se množi faktorom refleksije (1 - alpha_b)
                ray.energy *= material.ReflectedEnergy();

                // Orijentacija normale prema nailazećoj zraci
                const glm::vec3 normal = SafeUnit(
                    glm::dot(hit.normal, ray.direction) < 0.0f ? hit.normal : -hit.normal,
                    {0, 1, 0}
                );

                // 2. Spekularno odbijanje: r = d - 2(d . n)n
                const glm::vec3 specularDir = glm::reflect(ray.direction, normal);

                // 3. Difuzno rasipanje (Lambertov kosinusni zakon)
                const bool isDiffuse = Rand(rng) < std::clamp(material.scattering, 0.0f, 1.0f);
                const glm::vec3 nextDirection = isDiffuse
                    ? SampleCosineHemisphere(normal, rng)
                    : specularDir;

                // 4. Ruski rulet: aktivira se tek kad energija padne ispod -60 dB (10^-6 E0)
                // Zraka se s vjerovatnoćom p = 0.1 gasi, a preživjela skalira sa 1/(1-p)
                // (nepristrasna ocjena bez unošenja varijanse u rane refleksije!)
                const float currentMax = MaxEnergy(ray.energy);
                if (currentMax < initialMaxEnergy * cfg.minEnergyFraction) {
                    if (Rand(rng) < cfg.russianRouletteProb) {
                        ++stats.russianRouletteKills;
                        break;
                    }
                    ray.energy *= (1.0f / (1.0f - cfg.russianRouletteProb));
                }

                // Pomak ishodišta duž normale radi sprječavanja samo-presjeka
                ray.origin += normal * cfg.rayOffset;
                ray.SetDirection(nextDirection);

                ++ray.bounceCount;
            }
        }
    }

    return stats;
}

} // namespace acoustics
