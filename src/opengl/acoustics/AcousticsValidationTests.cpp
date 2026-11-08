#include "acoustics/AudioEngine.h"
#include "acoustics/RoomMetrics.h"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

namespace {

constexpr glm::vec3 kShoeboxDimensions{10.0f, 3.0f, 8.0f};
constexpr glm::vec3 kShoeboxCenter{0.0f, 1.5f, 0.0f};
constexpr glm::vec3 kListenerPosition{0.0f, 1.5f, 0.5f};
constexpr std::size_t kBand1kHz = 3;

bool Report(bool passed, const char* name) {
    std::cout << (passed ? "PASS " : "FAIL ") << name << '\n';
    return passed;
}

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
    constexpr uint32_t lowRayCount = 32'000;
    constexpr uint32_t highRayCount = 128'000;

    AcousticMaterialLibrary materials;
    // Use diffuse walls to exercise the diffuse-field estimates in a controlled case.
    const auto wall = materials.RegisterMaterial({"Diffuse validation", OctaveBands(0.2f), 1.0f});
    AcousticSceneGeometry room;
    room.AddShoeboxRoom(kShoeboxDimensions, wall, wall, wall, kShoeboxCenter);
    const double sabine = room.CalculateSabineRT60(kBand1kHz, materials);
    const double eyring = room.CalculateEyringRT60(kBand1kHz, materials);
    std::cout << "Shoebox RT60 at 1 kHz (T30, 10 x 8 x 3 m, alpha = 0.2, diffuse walls)\n"
              << "Analytical references: Eyring = " << eyring << " s, Sabine = " << sabine << " s\n";
    if (std::abs(eyring - 0.6461) > 0.001 || std::abs(sabine - 0.7209) > 0.001) {
        std::cerr << "FAIL analytical references for the Shoebox room\n";
        return false;
    }

    BVH bvh;
    bvh.Build(room);
    const acoustics::Simulator simulator(bvh, materials);
    acoustics::Receiver listener;
    listener.headCenter = kListenerPosition;
    listener.headShadow = false;
    acoustics::Source source;
    source.position = {0.0f, 1.5f, -1.5f};
    const std::vector sources{source};
    acoustics::Settings settings;
    settings.airAbsorption = OctaveBands::Zero(); // The references model surface losses only.

    const auto low = MeasureReverberation(simulator, sources, listener, settings, lowRayCount, kBand1kHz);
    const auto high = MeasureReverberation(simulator, sources, listener, settings, highRayCount, kBand1kHz);
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

struct DirectArrival {
    double energy = 0.0;
    double centerSample = 0.0;
    float peakEnergy = 0.0f;
};

DirectArrival MeasureArrival(const std::vector<OctaveBands>& bins) {
    DirectArrival arrival;
    for (std::size_t bin = 0; bin < bins.size(); ++bin) {
        const float energy = bins[bin][kBand1kHz];
        arrival.energy += energy;
        arrival.centerSample += (static_cast<double>(bin) + 0.5) * energy;
        arrival.peakEnergy = std::max(arrival.peakEnergy, energy);
    }
    if (arrival.energy > 0.0)
        arrival.centerSample /= arrival.energy;
    return arrival;
}

bool MatchesDirectArrival(const std::vector<OctaveBands>& bins, const DirectArrival& arrival,
                          double expectedSample) {
    // A chord spreads energy over several bins. The geometric arrival lies on the peak,
    // and the energy centroid gives a stable fractional-sample estimate across the plateau.
    const auto expectedBin = static_cast<std::size_t>(expectedSample);
    return expectedBin < bins.size() && arrival.energy > 0.0 &&
           std::abs(arrival.centerSample - expectedSample) <= 1.0 &&
           bins[expectedBin][kBand1kHz] >= 0.98f * arrival.peakEnergy;
}

bool TestFrontalSource(const acoustics::Simulator& simulator, const acoustics::Receiver& listener,
                       acoustics::Settings settings) {
    settings.raysPerSource = 4'000'000;
    acoustics::Echogram near(settings.maxDuration), far(settings.maxDuration);
    acoustics::Source source;
    source.position = listener.headCenter + 2.0f * listener.forward;
    simulator.Run({source}, listener, near, settings);
    source.position = listener.headCenter + 4.0f * listener.forward;
    simulator.Run({source}, listener, far, settings);

    const auto nearLeft = MeasureArrival(near.left);
    const auto nearRight = MeasureArrival(near.right);
    const auto farLeft = MeasureArrival(far.left);
    const auto farRight = MeasureArrival(far.right);
    const auto expectedSample = [&](double distance) {
        return std::hypot(distance, listener.earSpacing * 0.5) * near.sampleRate / settings.speedOfSound;
    };
    const bool directTime = Report(
        MatchesDirectArrival(near.left, nearLeft, expectedSample(2.0)) &&
        MatchesDirectArrival(near.right, nearRight, expectedSample(2.0)) &&
        MatchesDirectArrival(far.left, farLeft, expectedSample(4.0)) &&
        MatchesDirectArrival(far.right, farRight, expectedSample(4.0)),
        "9.2 Direct peak at ear distance / sound speed");

    const double levelChange = 10.0 * std::log10(
        (farLeft.energy + farRight.energy) / (nearLeft.energy + nearRight.energy));
    std::cout << "Direct level change from 2 to 4 m: " << levelChange << " dB\n";
    const bool inverseSquare = Report(std::abs(levelChange + 6.0206) <= 0.5,
                                      "9.2 Doubling distance lowers energy level by about 6 dB");

    const double itd = nearLeft.centerSample - nearRight.centerSample;
    std::cout << "Frontal ITD: " << itd << " samples\n";
    const bool frontalItd = Report(
        nearLeft.energy > 0.0 && nearRight.energy > 0.0 &&
        std::round(nearLeft.centerSample) == std::round(nearRight.centerSample) && std::abs(itd) < 1.0,
        "9.3 Frontal source arrives in the same sample at both ears");
    return directTime && inverseSquare && frontalItd;
}

double BandEnergy(const std::vector<OctaveBands>& bins, std::size_t band) {
    return std::accumulate(bins.begin(), bins.end(), 0.0,
        [band](double sum, const OctaveBands& bin) { return sum + bin[band]; });
}

bool TestLateralSources(const acoustics::Simulator& simulator, acoustics::Receiver listener,
                        acoustics::Settings settings) {
    // An axial pencil ray crosses the centers of both ears, isolating the binaural receiver.
    settings.raysPerSource = 1;
    acoustics::Source rightSource;
    rightSource.pattern = acoustics::Source::Pattern::Parallel;
    rightSource.emitterRadius = 0.0f;
    rightSource.position = listener.headCenter + 2.0f * listener.right;
    rightSource.direction = -listener.right;
    auto leftSource = rightSource;
    leftSource.position = listener.headCenter - 2.0f * listener.right;
    leftSource.direction = listener.right;

    acoustics::Echogram clear(settings.maxDuration), fromRight(settings.maxDuration), fromLeft(settings.maxDuration);
    simulator.Run({rightSource}, listener, clear, settings);
    listener.headShadow = true;
    const auto rightStats = simulator.Run({rightSource}, listener, fromRight, settings);
    const auto leftStats = simulator.Run({leftSource}, listener, fromLeft, settings);

    const auto left = MeasureArrival(fromRight.left);
    const auto right = MeasureArrival(fromRight.right);
    const double delay = left.centerSample - right.centerSample;
    const auto sampleDelay = std::lround(delay);
    const double expectedDelay = listener.earSpacing * fromRight.sampleRate / settings.speedOfSound;
    std::cout << "Right-side ITD: " << delay << " samples (expected " << expectedDelay << ")\n";
    const bool sideItd = Report(left.energy > 0.0 && right.energy > 0.0 &&
                                (sampleDelay == 24 || sampleDelay == 25) &&
                                std::abs(delay - expectedDelay) <= 0.25,
                                "9.4 Right ear leads at 90 degrees");

    const auto equalBin = [](const OctaveBands& a, const OctaveBands& b) { return a.values == b.values; };
    const bool mirrored = Report(rightStats.earHits == 2 && leftStats.earHits == 2 &&
        std::equal(fromRight.left.begin(), fromRight.left.end(), fromLeft.right.begin(), equalBin) &&
        std::equal(fromRight.right.begin(), fromRight.right.end(), fromLeft.left.begin(), equalBin),
        "9.5 Mirroring the source exactly swaps both direct echograms");
    acoustics::AudioEngine::SynthOptions synthOptions;
    const auto rightResponse = acoustics::AudioEngine::SynthesizeImpulseResponse(fromRight, synthOptions);
    // Move each channel's noise stream with its corresponding echogram.
    std::swap(synthOptions.leftSeed, synthOptions.rightSeed);
    const auto leftResponse = acoustics::AudioEngine::SynthesizeImpulseResponse(fromLeft, synthOptions);
    const bool mirroredAudio = Report(rightResponse.ok && leftResponse.ok &&
        rightResponse.audio.left == leftResponse.audio.right &&
        rightResponse.audio.right == leftResponse.audio.left,
        "9.5 Mirroring the echograms and seeds exactly swaps the synthesized response");

    bool highBandsAttenuated = rightStats.headOcclusions == 1 && leftStats.headOcclusions == 1;
    for (std::size_t band : {std::size_t{5}, std::size_t{6}}) {
        const double exposed = BandEnergy(fromRight.right, band);
        const double shadowed = BandEnergy(fromRight.left, band);
        const double unshadowed = BandEnergy(clear.left, band);
        const double levelDifference = 10.0 * std::log10(shadowed / exposed);
        const double expected = 10.0 * std::log10(1.0 - listener.headMaterialAbsorption[band]);
        std::cout << static_cast<int>(OctaveBands::Frequencies[band] / 1000.0f)
                  << " kHz far-ear level relative to near ear: "
                  << levelDifference << " dB\n";
        highBandsAttenuated &= levelDifference < -3.0 && std::abs(levelDifference - expected) <= 0.05 &&
                               std::abs(10.0 * std::log10(unshadowed / exposed)) <= 0.05;
    }
    const bool ild = Report(highBandsAttenuated, "9.5 Head shadow attenuates the far ear at 4 and 8 kHz");
    return sideItd && mirrored && mirroredAudio && ild;
}

bool TestDirectSound() {
    const auto& materials = AcousticMaterialLibrary::Instance();
    const auto wall = materials.GetMaterialId("Uniform02");
    AcousticSceneGeometry room;
    room.AddShoeboxRoom(kShoeboxDimensions, wall, wall, wall, kShoeboxCenter);
    BVH bvh;
    bvh.Build(room);
    const acoustics::Simulator simulator(bvh, materials);
    acoustics::Receiver listener;
    listener.headCenter = kListenerPosition;
    listener.headShadow = false;
    acoustics::Settings settings;
    settings.maxDuration = 0.05f;
    settings.maxBounces = 0;
    settings.airAbsorption = OctaveBands::Zero();

    const bool frontal = TestFrontalSource(simulator, listener, settings);
    const bool lateral = TestLateralSources(simulator, listener, settings);
    return frontal && lateral;
}

bool CheckComplexIntersections(const BVH& bvh, const AcousticSceneGeometry& geometry) {
    const auto& bounds = geometry.GetBounds();
    const glm::vec3 size = bounds.max - bounds.min;
    const float diagonal = glm::length(size);
    const float maxDistance = 2.0f * diagonal;
    constexpr float minDistance = 0.001f;
    unsigned hitCount = 0;
    unsigned missCount = 0;
    for (int row = 0; row < 10; ++row) {
        for (int column = 0; column < 10; ++column) {
            // Include probes just outside the bounding box to verify the miss path.
            const glm::vec3 origin{bounds.min.x + (0.14f * column - 0.12f) * size.x,
                                   bounds.max.y + diagonal,
                                   bounds.min.z + (0.14f * row - 0.12f) * size.z};
            const AcousticRay ray(origin, {0.0f, -1.0f, 0.0f});
            bool referenceHit = false;
            float closest = maxDistance;
            for (const auto& triangle : geometry.GetTriangles()) {
                float distance, u, v;
                if (IntersectTriangle(ray, triangle, minDistance, closest, distance, u, v)) {
                    closest = distance;
                    referenceHit = true;
                }
            }

            const auto hit = bvh.Intersect(ray, minDistance, maxDistance);
            const bool anyHit = bvh.IntersectAny(ray, minDistance, maxDistance);
            if (referenceHit != hit.HasHit() || referenceHit != anyHit ||
                (referenceHit && std::abs(closest - hit.distance) > 0.0001f)) {
                std::cerr << "BVH differs from the triangle scan at probe " << row << ", " << column << '\n';
                return false;
            }
            if (referenceHit)
                ++hitCount;
            else
                ++missCount;
        }
    }
    std::cout << "BVH vs. exhaustive scan: " << hitCount << " hits, " << missCount << " misses\n";
    return hitCount > 0 && missCount > 0;
}

bool TestComplexGeometry() {
    constexpr char modelPath[] = PRS_ASSET_DIR "/models/gltf/dragon_gate_inn_with_neon/scene.gltf";
    Assimp::Importer importer;
    const aiScene* model = importer.ReadFile(modelPath, aiProcess_Triangulate | aiProcess_PreTransformVertices);
    if (!model || !model->HasMeshes()) {
        std::cerr << "Could not read Neon City geometry: " << importer.GetErrorString() << '\n';
        return false;
    }

    AcousticSceneGeometry geometry;
    for (unsigned i = 0; i < model->mNumMeshes; ++i) {
        const aiMesh& mesh = *model->mMeshes[i];
        for (unsigned j = 0; j < mesh.mNumFaces; ++j) {
            const aiFace& face = mesh.mFaces[j];
            if (face.mNumIndices != 3)
                continue;
            const auto& a = mesh.mVertices[face.mIndices[0]];
            const auto& b = mesh.mVertices[face.mIndices[1]];
            const auto& c = mesh.mVertices[face.mIndices[2]];
            geometry.AddTriangle({a.x, a.y, a.z}, {b.x, b.y, b.z}, {c.x, c.y, c.z}, 0);
        }
    }
    geometry.Finalize();
    BVH bvh;
    bvh.Build(geometry);
    std::cout << "Neon City model: " << geometry.GetTriangleCount() << " triangles, "
              << bvh.GetNodeCount() << " BVH nodes\n";
    if (!Report(bvh.IsValid() && geometry.GetTriangleCount() >= 30'000 &&
                bvh.GetTriangleCount() == geometry.GetTriangleCount(), "9.6 BVH built from the complex model"))
        return false;
    const bool intersections = Report(CheckComplexIntersections(bvh, geometry),
                                      "9.6 BVH matches an exhaustive triangle scan");

    const acoustics::Simulator simulator(bvh, AcousticMaterialLibrary::Instance());
    acoustics::Receiver listener;
    listener.headCenter = {0.0f, 1.0f, 2.0f}; // Neon City camera in model coordinates.
    acoustics::Source source;
    source.position = listener.headCenter + glm::vec3(0.0f, 0.0f, 0.5f);
    acoustics::Settings settings;
    settings.maxDuration = 0.3f;
    acoustics::Echogram echogram(settings.maxDuration);
    const auto stats = simulator.Run({source}, listener, echogram, settings);
    const auto hasValidEnergy = [](const std::vector<OctaveBands>& bins) {
        return BandEnergy(bins, kBand1kHz) > 0.0 && std::all_of(bins.begin(), bins.end(), [](const OctaveBands& bin) {
            return std::all_of(bin.values.begin(), bin.values.end(), [](float energy) {
                return std::isfinite(energy) && energy >= 0.0f;
            });
        });
    };
    std::cout << "Complex trace: " << stats.emitted << " rays, " << stats.surfaceHits
              << " surface hits, " << stats.earHits << " ear hits\n";
    const bool simulation = Report(stats.emitted == settings.raysPerSource &&
        stats.surfaceHits > 0 && stats.segments > stats.emitted &&
        hasValidEnergy(echogram.left) && hasValidEnergy(echogram.right),
        "9.6 Complex trace produces finite energy in both ears");
    return intersections && simulation;
}

} // namespace

int main() {
    std::cout << std::fixed << std::setprecision(4);
    const bool directSound = TestDirectSound();
    const bool complexGeometry = TestComplexGeometry();
    const bool reverberation = TestShoeboxReverberation();
    return directSound && complexGeometry && reverberation ? 0 : 1;
}
