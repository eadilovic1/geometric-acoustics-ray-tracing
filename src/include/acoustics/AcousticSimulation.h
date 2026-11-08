#pragma once

#include "acoustics/BVH.h"
#include "acoustics/AcousticMaterial.h"
#include "acoustics/AcousticRay.h"
#include "scene/Light.h"

#include <glm/glm.hpp>
#include <array>
#include <cstdint>
#include <random>
#include <vector>

class Camera;
class LightEnvironment;

namespace acoustics {

// ===========================================================================
// DIO 3.1: Akustički izvori zvuka (preslikavanje svjetala u akustičke izvore)
// ===========================================================================

struct Source {
    enum class Pattern {
        Omni,      ///< PointLight -> Svesmjerni emiter (Fibonaccijeva sferna spirala)
        Cone,      ///< SpotLight -> Usmjereni konusni emiter sa kardioidnim opadanjem
        Parallel   ///< DirectionalLight -> Ravni talas (paralelne zrake preko diska aperture)
    };

    Pattern pattern = Pattern::Omni;

    glm::vec3 position{0.0f};

    glm::vec3 direction{0.0f, 0.0f, -1.0f};

    /// Parametri konusa za SpotLight (u stepenima)
    float coneInnerAngleDeg = 12.5f; ///< Puni intenzitet unutar ovog ugla
    float coneOuterAngleDeg = 20.0f; ///< Opadanje do nule prema rubu konusa

    /// Radijus diska aperture za paralelne zrake ravnog talasa (u metrima)
    float emitterRadius = 5.0f;

    /// Akustička snaga po oktavnim opsezima (W ili relativna energija)
    OctaveBands power{1.0f};
};

/// Pretvaranje PointLight u svesmjerni tačkasti akustički izvor
Source FromLight(const PointLight& light);

/// Pretvaranje SpotLight u usmjereni konusni akustički izvor
Source FromLight(const SpotLight& light);

/// Pretvaranje DirectionalLight u ravni talas (paralelne zrake) sa aperturom
Source FromLight(
    const DirectionalLight& light,
    const glm::vec3& emitterPosition,
    float beamRadius = 5.0f
);

/// Pretvaranje svih aktivnih svjetala iz okruženja u akustičke izvore
std::vector<Source> FromLights(
    const LightEnvironment& lights,
    const glm::vec3& directionalOrigin,
    float directionalBeamRadius = 5.0f
);


// ===========================================================================
// DIO 4.1 & 4.2: Binauralni prijemnik (Dvije sfere ušiju + centralna sfera glave)
// ===========================================================================

struct Receiver {
    glm::vec3 headCenter{0.0f};

    glm::vec3 right{1.0f, 0.0f, 0.0f};

    glm::vec3 forward{0.0f, 0.0f, -1.0f};

    glm::vec3 up{0.0f, 1.0f, 0.0f};

    /// Antropometrijski polurazmak glave a = 8.75 cm -> razmak ušiju 2a = 17.5 cm (0.175 m).
    float earSpacing = 0.175f;

    /// Radijus sfere uha rd = 2.5 cm (0.025 m).
    /// Zadovoljava rd < a i rd <= a / 2: sfere ušiju se NIKADA ne preklapaju (razmak je 12.5 cm).
    /// Vremenski razmaz unutar sfere 2*rd/c ≈ 0.14 ms << ITD_max (0.51 ms).
    float earRadius = 0.025f;

    /// Sfera glave u centru za ILD: Rh = a - rd = 6.25 cm (0.0625 m).
    float headRadius = 0.0625f;

    /// Prekidač za fizikalno ILD zasjenjenje glavom
    bool headShadow = true;

    /// Koeficijenti apsorpcije materijala glave (koža i kosa) po ISO oktavnim opsezima
    OctaveBands headMaterialAbsorption{0.10f, 0.15f, 0.25f, 0.40f, 0.65f, 0.80f, 0.85f};

    [[nodiscard]] glm::vec3 LeftEar() const;

    [[nodiscard]] glm::vec3 RightEar() const;

    /// Postavlja centar i orijentaciju slušaoca prema kameri:
    /// - FreeFly i FirstPerson: headCenter = camera.GetPosition()
    /// - ThirdPerson: headCenter = camera.GetOrbitTarget() + up * eyeHeightOffset
    void FollowCamera(const Camera& camera, float eyeHeightOffset = 1.65f);
};


// ===========================================================================
// DIO 4.4: Dvokanalni Ehogram (L & R energetski histogram odziva)
// ===========================================================================

struct Echogram {
    /// Frekvencija odabiranja ehograma.
    /// Po planu i ISO standardu fs = 48 kHz -> Δt = 20.83 μs
    /// Neophodna rezolucija za očuvanje sub-milisekundnog ITD-a (24-25 uzoraka na 90°).
    float sampleRate = 48000.0f;

    std::vector<OctaveBands> left;

    std::vector<OctaveBands> right;

    explicit Echogram(
        float durationSec = 2.0f,
        float binsPerSecond = 48000.0f
    );

    void Clear();

    [[nodiscard]] size_t BinCount() const { return left.size(); }
    [[nodiscard]] float DurationSec() const { return sampleRate > 0.0f ? static_cast<float>(left.size()) / sampleRate : 0.0f; }
    [[nodiscard]] float BinDurationSec() const { return sampleRate > 0.0f ? 1.0f / sampleRate : 0.0f; }

    /// Direktni upis energije u pojedinačni vremenski trenutak
    void Deposit(
        bool isLeft,
        float arrivalTime,
        const OctaveBands& energy
    );

    /// Upis energije po indeksu vremenskog bina
    void DepositByIndex(
        bool isLeft,
        size_t binIndex,
        const OctaveBands& energy
    );

    /// Kontinuirani raspodijeljeni upis energije tetive kroz vremenske binove
    void DepositChordSpan(
        bool isLeft,
        float tStart,
        float tEnd,
        const OctaveBands& totalEnergy
    );
};


// ===========================================================================
// DIO 3: Postavke simulacije
// ===========================================================================

struct AirConditions {
    static constexpr float MinTemperatureC = -20.0f;
    static constexpr float MaxTemperatureC = 50.0f;
    static constexpr float MinRelativeHumidityPercent = 10.0f;
    static constexpr float MaxRelativeHumidityPercent = 100.0f;

    float temperatureC = 20.0f;
    float relativeHumidityPercent = 50.0f;
};

struct Settings {
    uint32_t raysPerSource = 50000;

    uint32_t maxBounces = 50;

    float speedOfSound = 343.0f;

    float maxDuration = 2.0f;

    float maxSegmentDistance = 1000.0f;

    /// Prag za Ruski rulet: -60 dB (10^-6 E0)
    float minEnergyFraction = 1e-6f;

    /// Vjerovatnoća gašenja ispod -60 dB (p = 0.1)
    float russianRouletteProb = 0.1f;

    float rayOffset = 0.001f;

    /// Default energy attenuation coefficients [1/m] at the octave center frequencies.
    OctaveBands airAbsorption{
        0.00003f,
        0.00009f,
        0.00023f,
        0.00044f,
        0.00085f,
        0.00223f,
        0.00755f
    };

    uint32_t randomSeed = 42;

    /// Set sound speed and octave-band energy attenuation at 101.325 kPa.
    void SetAirConditions(const AirConditions& air);
};


// Statistika izvrsavanja simulacije
struct TraceStats {
    uint64_t emitted = 0;

    uint64_t segments = 0;

    uint64_t surfaceHits = 0;

    uint64_t earHits = 0;

    uint64_t headOcclusions = 0;

    uint64_t escaped = 0;

    uint64_t russianRouletteKills = 0;
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
