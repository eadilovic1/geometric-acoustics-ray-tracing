#include "acoustics/AudioEngine.h"
#include "acoustics/RoomMetrics.h"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
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

// ---------------------------------------------------------------------------
// Distance, absorption and playback. The tracer is unchanged; these checks
// measure the echogram in a time window and the audio path around it.
// ---------------------------------------------------------------------------

constexpr float kSafePeak = 0.99f;

double SumEnergy(const std::vector<OctaveBands>& bins, std::size_t begin, std::size_t end) {
    double sum = 0.0;
    end = std::min(end, bins.size());
    for (std::size_t i = begin; i < end; ++i) {
        for (float value : bins[i].values) {
            if (std::isfinite(value) && value > 0.0f) sum += value;
        }
    }
    return sum;
}

double SumBand(const std::vector<OctaveBands>& bins, std::size_t band, std::size_t begin, std::size_t end) {
    double sum = 0.0;
    end = std::min(end, bins.size());
    for (std::size_t i = begin; i < end; ++i) {
        const float value = bins[i][band];
        if (std::isfinite(value) && value > 0.0f) sum += value;
    }
    return sum;
}

struct EnergyParts {
    double direct = 0.0;
    double total = 0.0;
    double directBand = 0.0;
    double totalBand = 0.0;
};

EnergyParts SplitEnergy(const acoustics::Echogram& echo, float t0, float t1) {
    const std::size_t last = echo.BinCount();
    const auto index = [&](float time) {
        if (!(time > 0.0f)) return std::size_t{0};
        const auto bin = static_cast<std::size_t>(std::floor(time * echo.sampleRate));
        return std::min(bin, last);
    };
    const std::size_t begin = index(t0);
    const std::size_t end = std::max(begin, index(t1));
    EnergyParts parts;
    parts.direct = SumEnergy(echo.left, begin, end) + SumEnergy(echo.right, begin, end);
    parts.total = SumEnergy(echo.left, 0, last) + SumEnergy(echo.right, 0, last);
    parts.directBand = SumBand(echo.left, kBand1kHz, begin, end) + SumBand(echo.right, kBand1kHz, begin, end);
    parts.totalBand = SumBand(echo.left, kBand1kHz, 0, last) + SumBand(echo.right, kBand1kHz, 0, last);
    return parts;
}

float DirectArrivalTime(float distance, float earSpacing, float speed) {
    return std::hypot(distance, earSpacing * 0.5f) / speed;
}

double Decibels(double value, double reference) {
    return 10.0 * std::log10(value / reference);
}

double AmplitudeDecibels(double value, double reference) {
    return 20.0 * std::log10(value / reference);
}

float PeakAbs(const acoustics::StereoAudio& audio) {
    float peak = 0.0f;
    for (float sample : audio.left) peak = std::max(peak, std::abs(sample));
    for (float sample : audio.right) peak = std::max(peak, std::abs(sample));
    return peak;
}

double Rms(const acoustics::StereoAudio& audio) {
    double sum = 0.0;
    std::size_t count = 0;
    for (float sample : audio.left) {
        sum += static_cast<double>(sample) * sample;
        ++count;
    }
    for (float sample : audio.right) {
        sum += static_cast<double>(sample) * sample;
        ++count;
    }
    return count > 0 ? std::sqrt(sum / static_cast<double>(count)) : 0.0;
}

float LimiterGain(float peak) {
    return peak > kSafePeak ? kSafePeak / peak : 1.0f;
}

acoustics::MonoAudio MakeDrySignal() {
    acoustics::MonoAudio dry;
    dry.sampleRate = 48000.0f;
    dry.samples.resize(4800);
    for (std::size_t n = 0; n < dry.samples.size(); ++n) {
        const float phase = 2.0f * 3.14159265f * 1000.0f * static_cast<float>(n) / dry.sampleRate;
        dry.samples[n] = 0.5f * std::sin(phase);
    }
    return dry;
}

struct AudioLevels {
    float peak = 0.0f;
    double rms = 0.0f;
    float limiterGain = 1.0f;
};

AudioLevels MeasureAudio(const acoustics::StereoAudio& audio) {
    AudioLevels levels;
    levels.peak = PeakAbs(audio);
    levels.rms = Rms(audio);
    levels.limiterGain = LimiterGain(levels.peak);
    return levels;
}

void PrintPair(const char* label, double nearValue, double farValue, bool amplitude) {
    const double db = (nearValue > 0.0 && farValue > 0.0)
        ? (amplitude ? AmplitudeDecibels(farValue, nearValue) : Decibels(farValue, nearValue))
        : std::numeric_limits<double>::quiet_NaN();
    std::cout << label << "  2m=" << nearValue << "  4m=" << farValue << "  4m-2m=" << db << " dB\n";
}

acoustics::Source SourceAt(const acoustics::Receiver& listener, float distance, float power) {
    acoustics::Source source;
    source.position = listener.headCenter + distance * listener.forward;
    source.power = OctaveBands(power);
    return source;
}

acoustics::Echogram Trace(
    const acoustics::Simulator& simulator,
    const acoustics::Receiver& listener,
    float distance,
    acoustics::Settings settings,
    float power,
    acoustics::TraceStats* stats = nullptr
) {
    acoustics::Echogram echogram(settings.maxDuration);
    const auto result = simulator.Run({SourceAt(listener, distance, power)}, listener, echogram, settings);
    if (stats) *stats = result;
    return echogram;
}

void PrintAudioPolicy(const char* name, const acoustics::Echogram& nearEcho, const acoustics::Echogram& farEcho,
                      const acoustics::MonoAudio& dry) {
    acoustics::AudioEngine::SynthOptions raw;
    raw.peakNormalize = false;
    const acoustics::AudioEngine::SynthOptions normalized{};
    const auto nearRaw = acoustics::AudioEngine::SynthesizeImpulseResponse(nearEcho, raw);
    const auto farRaw = acoustics::AudioEngine::SynthesizeImpulseResponse(farEcho, raw);
    const auto nearNorm = acoustics::AudioEngine::SynthesizeImpulseResponse(nearEcho, normalized);
    const auto farNorm = acoustics::AudioEngine::SynthesizeImpulseResponse(farEcho, normalized);
    const auto nearWet = acoustics::AudioEngine::Convolve(dry, nearRaw.audio);
    const auto farWet = acoustics::AudioEngine::Convolve(dry, farRaw.audio);
    const auto nearWetNorm = acoustics::AudioEngine::Convolve(dry, nearNorm.audio);
    const auto farWetNorm = acoustics::AudioEngine::Convolve(dry, farNorm.audio);

    const auto rawIrNear = MeasureAudio(nearRaw.audio);
    const auto rawIrFar = MeasureAudio(farRaw.audio);
    const auto normIrNear = MeasureAudio(nearNorm.audio);
    const auto normIrFar = MeasureAudio(farNorm.audio);
    const auto rawWetNear = MeasureAudio(nearWet.audio);
    const auto rawWetFar = MeasureAudio(farWet.audio);
    const auto normWetNear = MeasureAudio(nearWetNorm.audio);
    const auto normWetFar = MeasureAudio(farWetNorm.audio);

    std::cout << name << "\n";
    PrintPair("  RIR peak, peak-normalize on ", normIrNear.peak, normIrFar.peak, true);
    PrintPair("  RIR RMS,  peak-normalize on ", normIrNear.rms, normIrFar.rms, true);
    PrintPair("  RIR peak, peak-normalize off", rawIrNear.peak, rawIrFar.peak, true);
    PrintPair("  RIR RMS,  peak-normalize off", rawIrNear.rms, rawIrFar.rms, true);
    PrintPair("  Wet peak, peak-normalize on ", normWetNear.peak, normWetFar.peak, true);
    PrintPair("  Wet RMS,  peak-normalize on ", normWetNear.rms, normWetFar.rms, true);
    PrintPair("  Wet peak, peak-normalize off", rawWetNear.peak, rawWetFar.peak, true);
    PrintPair("  Wet RMS,  peak-normalize off", rawWetNear.rms, rawWetFar.rms, true);
    std::cout << "  WavIo limiter gain, normalize on:  2m=" << normWetNear.limiterGain
              << "  4m=" << normWetFar.limiterGain << "\n";
    std::cout << "  WavIo limiter gain, normalize off: 2m=" << rawWetNear.limiterGain
              << "  4m=" << rawWetFar.limiterGain << "\n";
}

bool TestDirectDistance(const acoustics::Receiver& listener, const acoustics::Settings& settings,
                        const acoustics::Echogram& nearEcho, const acoustics::Echogram& farEcho,
                        const acoustics::TraceStats& nearStats, const acoustics::TraceStats& farStats) {
    const float nearT = DirectArrivalTime(2.0f, listener.earSpacing, settings.speedOfSound);
    const float farT = DirectArrivalTime(4.0f, listener.earSpacing, settings.speedOfSound);
    const auto nearEnergy = SplitEnergy(nearEcho, nearT - 0.0004f, nearT + 0.0012f);
    const auto farEnergy = SplitEnergy(farEcho, farT - 0.0004f, farT + 0.0012f);
    std::cout << "\nTest A — direct sound, maxBounces=0, " << settings.raysPerSource
              << " rays, seed " << settings.randomSeed << "\n";
    std::cout << std::scientific << std::setprecision(6);
    PrintPair("  direct-window energy, all bands", nearEnergy.direct, farEnergy.direct, false);
    PrintPair("  direct-window energy, 1 kHz   ", nearEnergy.directBand, farEnergy.directBand, false);
    PrintPair("  echogram energy, all bands    ", nearEnergy.total, farEnergy.total, false);
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  ear hits 2m=" << nearStats.earHits << "  4m=" << farStats.earHits << "\n";

    const double level = Decibels(farEnergy.directBand, nearEnergy.directBand);
    const bool windowHoldsDirect =
        nearEnergy.direct > 0.98 * nearEnergy.total && farEnergy.direct > 0.98 * farEnergy.total;
    const bool inverseSquare = std::abs(level + 6.0206) <= 0.5;
    return Report(windowHoldsDirect, "A direct-window contains the direct arrival") &&
           Report(inverseSquare, "A doubling distance lowers direct-window energy by about 6 dB");
}

bool TestReverberantDistance(const acoustics::Receiver& listener,
                             const acoustics::Echogram& nearEcho, const acoustics::Echogram& farEcho,
                             const acoustics::Settings& settings) {
    const float nearT = DirectArrivalTime(2.0f, listener.earSpacing, settings.speedOfSound);
    const float farT = DirectArrivalTime(4.0f, listener.earSpacing, settings.speedOfSound);
    const auto nearEnergy = SplitEnergy(nearEcho, nearT - 0.0004f, nearT + 0.0012f);
    const auto farEnergy = SplitEnergy(farEcho, farT - 0.0004f, farT + 0.0012f);
    const double nearReflected = nearEnergy.total - nearEnergy.direct;
    const double farReflected = farEnergy.total - farEnergy.direct;
    std::cout << "\nTest B — Uniform02, maxBounces=50, " << settings.raysPerSource << " rays, seed "
              << settings.randomSeed << "\n";
    std::cout << std::scientific << std::setprecision(6);
    PrintPair("  total energy                  ", nearEnergy.total, farEnergy.total, false);
    PrintPair("  direct-window energy          ", nearEnergy.direct, farEnergy.direct, false);
    PrintPair("  reflected energy              ", nearReflected, farReflected, false);
    std::cout << std::fixed << std::setprecision(4);
    if (nearEnergy.direct > 0.0) {
        std::cout << "  reflected/direct 2m=" << (nearReflected / nearEnergy.direct)
                  << "  4m=" << (farEnergy.direct > 0.0 ? farReflected / farEnergy.direct : 0.0) << "\n";
    }

    const double totalDb = Decibels(farEnergy.total, nearEnergy.total);
    const bool energyPresent = nearEnergy.total > 0.0 && farEnergy.total > 0.0 &&
                               nearReflected > 0.0 && farReflected > 0.0;
    const bool reverbDominates = nearReflected > nearEnergy.direct && farReflected > farEnergy.direct;
    // Free-field decay would be about -6 dB. The room is past its critical distance.
    const bool notFreeField = totalDb > -4.0;
    return Report(energyPresent, "B direct and reflected energy are present at 2 m and 4 m") &&
           Report(reverbDominates, "B reflected energy exceeds the direct window") &&
           Report(notFreeField, "B total energy does not fall by a free-field 6 dB");
}

bool TestSingleBounceAbsorption() {
    AcousticMaterialLibrary materials;
    const auto hard = materials.RegisterMaterial(
        AcousticMaterial("HardMirror", OctaveBands(0.0f), 0.0f));
    const auto half = materials.RegisterMaterial(
        AcousticMaterial("HalfAbsorb", OctaveBands(0.5f), 0.0f));

    AcousticSceneGeometry hardRoom;
    hardRoom.AddShoeboxRoom(kShoeboxDimensions, hard, hard, hard, kShoeboxCenter);
    AcousticSceneGeometry halfRoom;
    halfRoom.AddShoeboxRoom(kShoeboxDimensions, half, half, half, kShoeboxCenter);
    BVH hardBvh;
    hardBvh.Build(hardRoom);
    BVH halfBvh;
    halfBvh.Build(halfRoom);
    const acoustics::Simulator hardSim(hardBvh, materials);
    const acoustics::Simulator halfSim(halfBvh, materials);

    acoustics::Receiver listener;
    listener.headCenter = kListenerPosition;
    listener.headShadow = false;
    acoustics::Settings settings;
    settings.maxDuration = 0.08f;
    settings.maxBounces = 1;
    settings.raysPerSource = 200'000;
    settings.airAbsorption = OctaveBands::Zero();
    settings.randomSeed = 42;

    const auto hardEcho = Trace(hardSim, listener, 2.0f, settings, 1.0f);
    const auto halfEcho = Trace(halfSim, listener, 2.0f, settings, 1.0f);
    const float arrival = DirectArrivalTime(2.0f, listener.earSpacing, settings.speedOfSound);
    const auto hardEnergy = SplitEnergy(hardEcho, arrival - 0.0004f, arrival + 0.0012f);
    const auto halfEnergy = SplitEnergy(halfEcho, arrival - 0.0004f, arrival + 0.0012f);
    const double hardReflected = hardEnergy.total - hardEnergy.direct;
    const double halfReflected = halfEnergy.total - halfEnergy.direct;
    const double ratio = halfReflected / hardReflected;

    std::cout << "\nTest C — one bounce, alpha 0 vs alpha 0.5, scattering 0\n";
    std::cout << std::scientific << std::setprecision(6);
    std::cout << "  direct energy  alpha0=" << hardEnergy.direct << "  alpha0.5=" << halfEnergy.direct << "\n";
    std::cout << "  reflected energy alpha0=" << hardReflected << "  alpha0.5=" << halfReflected
              << "  ratio=" << ratio << "\n";
    std::cout << std::fixed << std::setprecision(4);

    const bool directUnchanged = hardEnergy.direct > 0.0 &&
                                 std::abs(halfEnergy.direct - hardEnergy.direct) <= 1e-4 * hardEnergy.direct;
    const bool reflectedScaled = hardReflected > 0.0 && std::abs(ratio - 0.5) <= 0.01;
    return Report(directUnchanged, "C direct sound ignores the wall absorption") &&
           Report(reflectedScaled, "C one bounce scales reflected energy by (1 - alpha)");
}

bool WavPreservesRatio(const acoustics::StereoAudio& nearAudio, const acoustics::StereoAudio& farAudio,
                       const std::filesystem::path& directory) {
    const auto nearPath = directory / "loudness-near.wav";
    const auto farPath = directory / "loudness-far.wav";
    const auto nearWrite = acoustics::AudioEngine::WriteStereoWav(nearPath, nearAudio);
    const auto farWrite = acoustics::AudioEngine::WriteStereoWav(farPath, farAudio);
    const auto nearRead = acoustics::AudioEngine::ReadWav(nearPath);
    const auto farRead = acoustics::AudioEngine::ReadWav(farPath);
    if (!nearWrite.ok || !farWrite.ok || !nearRead.ok || !farRead.ok) return false;

    const auto peakOf = [](const acoustics::WavData& wav) {
        float peak = 0.0f;
        for (float sample : wav.interleaved) peak = std::max(peak, std::abs(sample));
        return peak;
    };
    const float nearPeak = PeakAbs(nearAudio);
    const float farPeak = PeakAbs(farAudio);
    const float nearReadPeak = peakOf(nearRead);
    const float farReadPeak = peakOf(farRead);
    const double before = AmplitudeDecibels(farPeak, nearPeak);
    const double after = AmplitudeDecibels(farReadPeak, nearReadPeak);
    std::cout << "  WAV peak dB before write=" << before << "  after PCM16 read=" << after << "\n";
    std::cout << "  write messages: near=\"" << nearWrite.message << "\" far=\"" << farWrite.message << "\"\n";
    const bool limiterIdle = nearPeak <= kSafePeak && farPeak <= kSafePeak &&
                             std::abs(nearReadPeak - nearPeak) < 0.002f &&
                             std::abs(farReadPeak - farPeak) < 0.002f;
    return limiterIdle && std::abs(after - before) <= 0.25;
}

bool TestPlayback(const acoustics::Echogram& nearDirect, const acoustics::Echogram& farDirect,
                  const acoustics::Echogram& nearRoom, const acoustics::Echogram& farRoom) {
    const auto dry = MakeDrySignal();
    std::cout << "\nMeasured audio (same dry 1 kHz burst, same seeds)\n";
    PrintAudioPolicy("Direct, 4e6 rays:", nearDirect, farDirect, dry);
    PrintAudioPolicy("Uniform02 room:", nearRoom, farRoom, dry);

    // Same direct echogram with one quarter of the energy. This is the 2 m -> 4 m
    // energy ratio on an identical arrival, so the noise and filters do not move.
    auto quarter = nearDirect;
    for (OctaveBands& bin : quarter.left) bin *= 0.25f;
    for (OctaveBands& bin : quarter.right) bin *= 0.25f;

    acoustics::AudioEngine::SynthOptions playback;
    playback.peakNormalize = false;
    const auto fullIr = acoustics::AudioEngine::SynthesizeImpulseResponse(nearDirect, playback);
    const auto quarterIr = acoustics::AudioEngine::SynthesizeImpulseResponse(quarter, playback);
    const auto fullWet = acoustics::AudioEngine::Convolve(dry, fullIr.audio);
    const auto quarterWet = acoustics::AudioEngine::Convolve(dry, quarterIr.audio);
    const auto fullLevels = MeasureAudio(fullWet.audio);
    const auto quarterLevels = MeasureAudio(quarterWet.audio);
    const double wetDb = AmplitudeDecibels(quarterLevels.rms, fullLevels.rms);
    const bool scaled = fullIr.ok && quarterIr.ok && fullWet.ok && quarterWet.ok &&
                        fullLevels.limiterGain == 1.0f && quarterLevels.limiterGain == 1.0f &&
                        std::abs(wetDb + 6.0206) <= 0.05;

    acoustics::AudioEngine::SynthOptions normalized;
    const auto fullNorm = acoustics::AudioEngine::SynthesizeImpulseResponse(nearDirect, normalized);
    const auto quarterNorm = acoustics::AudioEngine::SynthesizeImpulseResponse(quarter, normalized);
    const double normPeakDb = AmplitudeDecibels(PeakAbs(quarterNorm.audio), PeakAbs(fullNorm.audio));
    const bool oldPathFlattens = fullNorm.ok && quarterNorm.ok && std::abs(normPeakDb) <= 0.05;

    const auto roomNear = acoustics::AudioEngine::SynthesizeImpulseResponse(nearRoom, playback);
    const auto roomFar = acoustics::AudioEngine::SynthesizeImpulseResponse(farRoom, playback);
    const auto roomNearWet = acoustics::AudioEngine::Convolve(dry, roomNear.audio);
    const auto roomFarWet = acoustics::AudioEngine::Convolve(dry, roomFar.audio);
    const auto roomNearLevels = MeasureAudio(roomNearWet.audio);
    const auto roomFarLevels = MeasureAudio(roomFarWet.audio);
    const bool roomLimiterIdle = roomNearWet.ok && roomFarWet.ok &&
                                 roomNearLevels.limiterGain == 1.0f && roomFarLevels.limiterGain == 1.0f;

    const auto realNear = acoustics::AudioEngine::SynthesizeImpulseResponse(nearDirect, normalized);
    const auto realFar = acoustics::AudioEngine::SynthesizeImpulseResponse(farDirect, normalized);
    const double realNormPeakDb = AmplitudeDecibels(PeakAbs(realFar.audio), PeakAbs(realNear.audio));
    const bool realPeaksEquated = realNear.ok && realFar.ok && std::abs(realNormPeakDb) <= 0.05;

    std::filesystem::path directory;
    try {
        directory = std::filesystem::temp_directory_path() / "geometric-acoustics-loudness";
        std::filesystem::create_directories(directory);
    } catch (const std::exception&) {
        return Report(false, "D could not create a temporary WAV directory");
    }
    const bool wav = WavPreservesRatio(fullWet.audio, quarterWet.audio, directory);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  quarter-energy wet RMS dB=" << wetDb
              << "  normalized quarter-energy peak dB=" << normPeakDb
              << "  normalized real direct peak dB=" << realNormPeakDb << "\n";
    return Report(scaled, "D quarter energy keeps a 6 dB convolution drop and the limiter stays idle") &&
           Report(oldPathFlattens, "D default peak normalization equates a 6 dB energy difference") &&
           Report(realPeaksEquated, "D default peak normalization equates the real 2 m and 4 m direct peaks") &&
           Report(roomLimiterIdle, "D reverberant 2 m and 4 m convolution stays under the limiter") &&
           Report(wav, "D WAV round-trip keeps the quarter-energy peak ratio");
}

bool TestLoudness() {
    const auto& materials = AcousticMaterialLibrary::Instance();
    const auto wall = materials.GetMaterialId("Uniform02");
    AcousticSceneGeometry room;
    room.AddShoeboxRoom(kShoeboxDimensions, wall, wall, wall, kShoeboxCenter);
    BVH bvh;
    bvh.Build(room);
    const acoustics::Simulator simulator(bvh, materials);
    const auto dry = MakeDrySignal();

    acoustics::Receiver sceneListener;
    sceneListener.headCenter = kListenerPosition;
    sceneListener.headShadow = true;
    acoustics::Settings sceneSettings;
    sceneSettings.raysPerSource = 50'000;
    sceneSettings.maxBounces = 50;
    sceneSettings.maxDuration = 2.0f;
    sceneSettings.randomSeed = 42;
    sceneSettings.SetAirConditions({});
    std::cout << "\nBaseline before the playback change, shoebox Uniform02, power 3, "
              << "50000 rays, seed 42, head shadow, air 20 C / 50 %\n";
    std::cout << "  speed of sound " << sceneSettings.speedOfSound << " m/s\n";
    const auto sceneNear = Trace(simulator, sceneListener, 2.0f, sceneSettings, 3.0f);
    const auto sceneFar = Trace(simulator, sceneListener, 4.0f, sceneSettings, 3.0f);
    const float sceneNearT = DirectArrivalTime(2.0f, sceneListener.earSpacing, sceneSettings.speedOfSound);
    const float sceneFarT = DirectArrivalTime(4.0f, sceneListener.earSpacing, sceneSettings.speedOfSound);
    const auto sceneNearEnergy = SplitEnergy(sceneNear, sceneNearT - 0.0004f, sceneNearT + 0.0012f);
    const auto sceneFarEnergy = SplitEnergy(sceneFar, sceneFarT - 0.0004f, sceneFarT + 0.0012f);
    std::cout << std::scientific << std::setprecision(6);
    PrintPair("  total energy        ", sceneNearEnergy.total, sceneFarEnergy.total, false);
    PrintPair("  direct-window energy", sceneNearEnergy.direct, sceneFarEnergy.direct, false);
    PrintPair("  reflected energy    ", sceneNearEnergy.total - sceneNearEnergy.direct,
              sceneFarEnergy.total - sceneFarEnergy.direct, false);
    std::cout << std::fixed << std::setprecision(4);
    PrintAudioPolicy("Baseline playback policies:", sceneNear, sceneFar, dry);

    acoustics::Receiver listener;
    listener.headCenter = kListenerPosition;
    listener.headShadow = false;
    acoustics::Settings directSettings;
    directSettings.maxDuration = 0.05f;
    directSettings.maxBounces = 0;
    directSettings.raysPerSource = 4'000'000;
    directSettings.airAbsorption = OctaveBands::Zero();
    directSettings.randomSeed = 42;
    acoustics::TraceStats nearDirectStats;
    acoustics::TraceStats farDirectStats;
    const auto nearDirect = Trace(simulator, listener, 2.0f, directSettings, 1.0f, &nearDirectStats);
    const auto farDirect = Trace(simulator, listener, 4.0f, directSettings, 1.0f, &farDirectStats);
    const bool direct = TestDirectDistance(listener, directSettings, nearDirect, farDirect,
                                           nearDirectStats, farDirectStats);

    acoustics::Settings roomSettings;
    roomSettings.raysPerSource = 100'000;
    roomSettings.maxBounces = 50;
    roomSettings.maxDuration = 2.0f;
    roomSettings.airAbsorption = OctaveBands::Zero();
    roomSettings.randomSeed = 42;
    const auto nearRoom = Trace(simulator, listener, 2.0f, roomSettings, 1.0f);
    const auto farRoom = Trace(simulator, listener, 4.0f, roomSettings, 1.0f);
    const bool reverberant = TestReverberantDistance(listener, nearRoom, farRoom, roomSettings);
    const bool playback = TestPlayback(nearDirect, farDirect, nearRoom, farRoom);
    return direct && reverberant && playback;
}

} // namespace

int main() {
    std::cout << std::fixed << std::setprecision(4);
    const bool directSound = TestDirectSound();
    const bool complexGeometry = TestComplexGeometry();
    const bool reverberation = TestShoeboxReverberation();
    const bool loudness = TestLoudness();
    const bool absorption = TestSingleBounceAbsorption();
    return directSound && complexGeometry && reverberation && loudness && absorption ? 0 : 1;
}
