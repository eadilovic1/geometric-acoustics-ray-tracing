#pragma once

#include "acoustics/AcousticSimulation.h"

#include <array>
#include <string>
#include <vector>

namespace acoustics {

// ---------------------------------------------------------------------------
// Korak 5: Schroederova povratna integracija i ISO 3382-1 metrike.
//
// Ulaz je postojeci dvokanalni ehogram (korak 4). Simulator se ne mijenja:
//
//   Echogram echogram(settings.maxDuration, 48000.0f);
//   simulator.Run(sources, receiver, echogram, settings);
//   const RoomMetricsResult metrics = RoomMetrics::Compute(echogram);
//
// Nulta vremenska os je prvi znacajan dolazak u tom kanalu i opsegu
// (direktni zvuk). Vodeća tišina ne ulazi u EDT.
// ---------------------------------------------------------------------------

enum class MetricStatus {
    Ok = 0,
    EmptyInput,
    InvalidSampleRate,
    NonFiniteInput,
    NoEnergy,
    InsufficientDynamicRange,
    PoorFit,
    UndefinedRatio
};

[[nodiscard]] inline const char* MetricStatusName(MetricStatus status) noexcept {
    switch (status) {
    case MetricStatus::Ok: return "Ok";
    case MetricStatus::EmptyInput: return "EmptyInput";
    case MetricStatus::InvalidSampleRate: return "InvalidSampleRate";
    case MetricStatus::NonFiniteInput: return "NonFiniteInput";
    case MetricStatus::NoEnergy: return "NoEnergy";
    case MetricStatus::InsufficientDynamicRange: return "InsufficientDynamicRange";
    case MetricStatus::PoorFit: return "PoorFit";
    case MetricStatus::UndefinedRatio: return "UndefinedRatio";
    }
    return "Unknown";
}

/// Vrijednost ima smisla samo kada je status Ok. U suprotnom value je 0.
struct ScalarMetric {
    MetricStatus status = MetricStatus::EmptyInput;
    double value = 0.0;

    [[nodiscard]] bool IsValid() const noexcept { return status == MetricStatus::Ok; }
};

/// Normalizovana Schroederova kriva jednog opsega.
/// levelDb[k] = 10 log10(E[k] / E[0]), E[k] = suma energije od k do kraja.
struct BandDecay {
    MetricStatus status = MetricStatus::EmptyInput;
    std::vector<double> backwardEnergy;
    std::vector<double> levelDb;

    [[nodiscard]] bool IsValid() const noexcept { return status == MetricStatus::Ok; }
};

struct BandResult {
    BandDecay decay;
    ScalarMetric rt60FromT30; ///< T30 regresija (-5 dB .. -35 dB), ekstrapolacija na 60 dB
    ScalarMetric edt;         ///< EDT regresija (0 dB .. -10 dB), ekstrapolacija na 60 dB
    ScalarMetric c50Db;
    ScalarMetric c80Db;
    ScalarMetric d50Percent;  ///< 0 .. 100
};

struct ChannelMetrics {
    std::array<BandResult, OctaveBands::Count> bands{};
};

struct RoomMetricsResult {
    float sampleRate = 0.0f;
    ChannelMetrics left;
    ChannelMetrics right;
};

class RoomMetrics {
public:
    struct Options {
        /// Minimalni koeficijent determinacije linearne regresije.
        /// Ispod praga kriva nije dovoljno blizu pravoj i rezultat je PoorFit.
        double minimumDetermination = 0.90;

        std::size_t minimumRegressionSamples = 8;

        /// Uzorci tisji od peak * onsetFraction tretiraju se kao tišina prije direktnog zvuka.
        double onsetFraction = 1e-6;

        /// Numerička prašina ispod peak * dustFraction ne produžava dinamički raspon.
        double dustFraction = 1e-10;
    };

    [[nodiscard]] static RoomMetricsResult Compute(
        const Echogram& echogram,
        const Options& options = {}
    );
};

} // namespace acoustics
