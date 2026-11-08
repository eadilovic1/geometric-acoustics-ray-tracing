#include "acoustics/RoomMetrics.h"

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>

namespace {

struct ReverberationSample {
    double mean;
    double standardDeviation;
};

std::optional<ReverberationSample> MeasureReverberation(
    const acoustics::Simulator& simulator,
    const std::vector<acoustics::Source>& sources,
    const acoustics::Receiver& listener,
    const acoustics::Settings& baseSettings,
    uint32_t rayCount,
    std::size_t band
) {
    auto settings = baseSettings;
    settings.raysPerSource = rayCount;
    acoustics::Echogram echogram(settings.maxDuration);
    std::array<double, 8> estimates{};
    for (std::size_t i = 0; i < estimates.size(); ++i) {
        settings.randomSeed = baseSettings.randomSeed + static_cast<uint32_t>(i);
        simulator.Run(sources, listener, echogram, settings);
        const auto metrics = acoustics::RoomMetrics::Compute(echogram);
        const auto& left = metrics.left.bands[band].rt60FromT30;
        const auto& right = metrics.right.bands[band].rt60FromT30;
        if (!left.IsValid() || !right.IsValid()) {
            std::cerr << "Invalid T30 at " << rayCount << " rays, seed " << settings.randomSeed
                      << " (L: " << acoustics::MetricStatusName(left.status)
                      << ", R: " << acoustics::MetricStatusName(right.status) << ")\n";
            return std::nullopt;
        }
        estimates[i] = (left.value + right.value) * 0.5;
    }

    const double mean = std::accumulate(estimates.begin(), estimates.end(), 0.0) / estimates.size();
    double squaredDeviation = 0.0;
    for (double estimate : estimates)
        squaredDeviation += (estimate - mean) * (estimate - mean);
    const double standardDeviation = std::sqrt(squaredDeviation / (estimates.size() - 1));
    std::cout << rayCount << " rays, " << estimates.size() << " runs: mean RT60 = "
              << mean << " s, standard deviation = " << standardDeviation << " s" << std::endl;
    return ReverberationSample{mean, standardDeviation};
}

bool TestShoeboxReverberation() {
    constexpr std::size_t band = 3; // 1 kHz
    constexpr uint32_t lowRayCount = 32'000;
    constexpr uint32_t highRayCount = 128'000;

    AcousticMaterialLibrary materials;
    // Use diffuse walls to exercise the diffuse-field estimates in a controlled case.
    const auto wall = materials.RegisterMaterial({"Diffuse validation", OctaveBands(0.2f), 1.0f});
    AcousticSceneGeometry room;
    room.AddShoeboxRoom({10.0f, 3.0f, 8.0f}, wall, wall, wall, {0.0f, 1.5f, 0.0f});
    const double sabine = room.CalculateSabineRT60(band, materials);
    const double eyring = room.CalculateEyringRT60(band, materials);
    std::cout << std::fixed << std::setprecision(4)
              << "Shoebox RT60 at 1 kHz (T30, 10 x 8 x 3 m, alpha = 0.2, diffuse walls)\n"
              << "Analytical references: Eyring = " << eyring << " s, Sabine = " << sabine << " s\n";
    if (std::abs(eyring - 0.6461) > 0.001 || std::abs(sabine - 0.7209) > 0.001) {
        std::cerr << "FAIL analytical references for the Shoebox room\n";
        return false;
    }

    BVH bvh;
    bvh.Build(room);
    const acoustics::Simulator simulator(bvh, materials);
    acoustics::Receiver listener;
    listener.headCenter = {0.0f, 1.5f, 0.5f};
    listener.headShadow = false;
    acoustics::Source source;
    source.position = {0.0f, 1.5f, -1.5f};
    const std::vector sources{source};
    acoustics::Settings settings;
    settings.airAbsorption = OctaveBands::Zero(); // The references model surface losses only.

    const auto low = MeasureReverberation(simulator, sources, listener, settings, lowRayCount, band);
    const auto high = MeasureReverberation(simulator, sources, listener, settings, highRayCount, band);
    if (!low || !high)
        return false;

    const bool inRange = high->mean >= eyring && high->mean <= sabine;
    const double expectedRatio = std::sqrt(static_cast<double>(lowRayCount) / highRayCount);
    // The eight-seed estimate of spread is noisy. Check a reduction consistent with 1/sqrt(N).
    const double maxRatio = 1.6 * expectedRatio;
    const bool converged = low->standardDeviation > 0.0 &&
                           high->standardDeviation <= maxRatio * low->standardDeviation;
    std::cout << (inRange ? "PASS" : "FAIL") << " mean RT60 between Eyring and Sabine\n"
              << (converged ? "PASS" : "FAIL") << " RT60 spread falls with ray count"
              << " (expected spread ratio " << expectedRatio << ", limit " << maxRatio << ")\n";
    return inRange && converged;
}

} // namespace

int main() {
    return TestShoeboxReverberation() ? 0 : 1;
}
