#include "acoustics/AcousticGeometry.h"
#include "acoustics/AcousticSimulation.h"
#include "acoustics/AudioEngine.h"
#include "acoustics/BVH.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kSeed = 42;
constexpr uint32_t kMaxBounces = 50;
constexpr float kDurationSec = 2.0f;
constexpr double kProfileTargetSec = 3.0;
constexpr int kProfileRepeatCap = 60;
constexpr std::array<uint32_t, 5> kSuiteRays{50'000u, 100'000u, 250'000u, 500'000u, 1'000'000u};

const glm::vec3 kRoomDimensions{10.0f, 3.0f, 8.0f};
const glm::vec3 kRoomCenter{0.0f, 1.5f, 0.0f};
const glm::vec3 kListenerPosition{0.0f, 1.5f, 0.5f};
const glm::vec3 kSourcePosition{0.0f, 1.5f, -1.5f};

volatile double g_consume = 0.0;

void Consume(double value) {
    g_consume = value;
}

struct Options {
    uint32_t rays = 100'000;
    int repeats = 5;
    int warmup = 1;
    bool suite = false;
    bool profile = false;
    bool audio = true;
    bool repeatsExplicit = false;
    std::optional<std::filesystem::path> csv;
};

struct Moments {
    double mean = 0.0;
    double median = 0.0;
    double minimum = 0.0;
    double stddev = 0.0;
};

struct Fingerprint {
    acoustics::TraceStats stats{};
    double energy = 0.0;
};

bool SameFingerprint(const Fingerprint& a, const Fingerprint& b) {
    return a.stats.emitted == b.stats.emitted
        && a.stats.segments == b.stats.segments
        && a.stats.surfaceHits == b.stats.surfaceHits
        && a.stats.earHits == b.stats.earHits
        && a.stats.headOcclusions == b.stats.headOcclusions
        && a.stats.escaped == b.stats.escaped
        && a.stats.russianRouletteKills == b.stats.russianRouletteKills
        && a.energy == b.energy;
}

void PrintUsage() {
    std::cout
        << "AcousticsSerialBench — serial acoustic timing harness\n"
        << "  --rays N       rays per source (default 100000)\n"
        << "  --repeats N    timed repetitions of each phase (default 5)\n"
        << "  --warmup N     untimed repetitions before each phase (default 1)\n"
        << "  --suite        measure 50000, 100000, 250000, 500000 and 1000000 rays\n"
        << "  --csv PATH     append rows; the header is written when the file is empty\n"
        << "  --profile      no audio; trace repeats stay under --repeats.\n"
        << "                 When --repeats is omitted and one pass is shorter than\n"
        << "                 " << kProfileTargetSec << " s, trace repeats increase up to\n"
        << "                 " << kProfileRepeatCap << " so CPU sampling has a longer window.\n"
        << "  --no-audio     skip RIR synthesis and convolution\n";
}

bool ParseUint(const std::string& text, uint32_t& out) {
    if (text.empty() || !std::all_of(text.begin(), text.end(), [](unsigned char c) { return std::isdigit(c); }))
        return false;
    try {
        const unsigned long value = std::stoul(text);
        if (value == 0 || value > std::numeric_limits<uint32_t>::max())
            return false;
        out = static_cast<uint32_t>(value);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

bool ParseInt(const std::string& text, int& out) {
    uint32_t value = 0;
    if (!ParseUint(text, value) || value > static_cast<uint32_t>(std::numeric_limits<int>::max()))
        return false;
    out = static_cast<int>(value);
    return true;
}

bool ParseArgs(int argc, char** argv, Options& options, std::string& error) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto requireValue = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                error = std::string(name) + " needs a value";
                return nullptr;
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") {
            PrintUsage();
            return false;
        }
        if (arg == "--suite") {
            options.suite = true;
            continue;
        }
        if (arg == "--profile") {
            options.profile = true;
            options.audio = false;
            continue;
        }
        if (arg == "--no-audio") {
            options.audio = false;
            continue;
        }
        if (arg == "--rays") {
            const char* value = requireValue("--rays");
            if (!value)
                return false;
            if (!ParseUint(value, options.rays)) {
                error = "--rays expects a positive integer";
                return false;
            }
            continue;
        }
        if (arg == "--repeats") {
            const char* value = requireValue("--repeats");
            if (!value)
                return false;
            if (!ParseInt(value, options.repeats)) {
                error = "--repeats expects a positive integer";
                return false;
            }
            options.repeatsExplicit = true;
            continue;
        }
        if (arg == "--warmup") {
            const char* value = requireValue("--warmup");
            if (!value)
                return false;
            uint32_t warmup = 0;
            if (!ParseUint(value, warmup) && value != std::string("0")) {
                error = "--warmup expects a non-negative integer";
                return false;
            }
            if (value == std::string("0"))
                options.warmup = 0;
            else
                options.warmup = static_cast<int>(warmup);
            continue;
        }
        if (arg == "--csv") {
            const char* value = requireValue("--csv");
            if (!value)
                return false;
            options.csv = std::filesystem::path(value);
            continue;
        }
        error = "Unknown argument: " + arg;
        return false;
    }

    if (options.profile)
        options.audio = false;
    return true;
}

double ElapsedMs(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

Moments Summarize(std::vector<double> samples) {
    Moments moments;
    if (samples.empty())
        return moments;
    moments.minimum = *std::min_element(samples.begin(), samples.end());
    const double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    moments.mean = sum / static_cast<double>(samples.size());
    std::sort(samples.begin(), samples.end());
    const std::size_t count = samples.size();
    if (count % 2 == 1)
        moments.median = samples[count / 2];
    else
        moments.median = 0.5 * (samples[count / 2 - 1] + samples[count / 2]);
    if (count > 1) {
        double squared = 0.0;
        for (double sample : samples) {
            const double delta = sample - moments.mean;
            squared += delta * delta;
        }
        moments.stddev = std::sqrt(squared / static_cast<double>(count - 1));
    }
    return moments;
}

int TraceRepeats(const Options& options, double warmupMs) {
    if (!options.profile || options.repeatsExplicit || !(warmupMs > 0.0))
        return options.repeats;
    const double warmupSec = warmupMs / 1000.0;
    if (warmupSec * static_cast<double>(options.repeats) >= kProfileTargetSec)
        return options.repeats;
    const int needed = static_cast<int>(std::ceil(kProfileTargetSec / warmupSec));
    return std::clamp(std::max(needed, options.repeats), 1, kProfileRepeatCap);
}

acoustics::Settings MakeSettings(uint32_t rays) {
    acoustics::Settings settings;
    settings.raysPerSource = rays;
    settings.maxBounces = kMaxBounces;
    settings.maxDuration = kDurationSec;
    settings.randomSeed = kSeed;
    settings.SetAirConditions({});
    return settings;
}

acoustics::Source MakeSource() {
    acoustics::Source source;
    source.pattern = acoustics::Source::Pattern::Omni;
    source.position = kSourcePosition;
    source.power = OctaveBands(3.0f);
    return source;
}

acoustics::Receiver MakeListener() {
    acoustics::Receiver listener;
    listener.headCenter = kListenerPosition;
    listener.headShadow = true;
    return listener;
}

uint32_t UniformMaterial() {
    const auto& materials = AcousticMaterialLibrary::Instance();
    if (!materials.FindByName("Uniform02"))
        throw std::runtime_error("Acoustic material Uniform02 is not registered");
    return materials.GetMaterialId("Uniform02");
}

void BuildRoom(AcousticSceneGeometry& room, uint32_t materialId) {
    room.Clear();
    room.AddShoeboxRoom(kRoomDimensions, materialId, materialId, materialId, kRoomCenter);
}

double GeometrySignature(const AcousticSceneGeometry& room) {
    double signature = static_cast<double>(room.GetTriangleCount()) + static_cast<double>(room.GetEstimatedVolume());
    for (const AcousticTriangle& triangle : room.GetTriangles()) {
        signature += static_cast<double>(triangle.v0.x + triangle.v1.y + triangle.v2.z + triangle.area);
    }
    Consume(signature);
    return signature;
}

double BvhSignature(const BVH& bvh) {
    const AABB bounds = bvh.GetBounds();
    const double signature = static_cast<double>(bvh.GetNodeCount() + bvh.GetTriangleCount())
        + static_cast<double>(bounds.min.x + bounds.min.y + bounds.min.z
            + bounds.max.x + bounds.max.y + bounds.max.z);
    Consume(signature);
    return signature;
}

double EchogramEnergy(const acoustics::Echogram& echogram) {
    double energy = 0.0;
    const auto accumulate = [&](const std::vector<OctaveBands>& channel) {
        for (const OctaveBands& bin : channel) {
            for (float value : bin.values)
                energy += static_cast<double>(value);
        }
    };
    accumulate(echogram.left);
    accumulate(echogram.right);
    Consume(energy);
    return energy;
}

double AudioEnergy(const acoustics::StereoAudio& audio) {
    double energy = 0.0;
    for (float sample : audio.left)
        energy += static_cast<double>(sample);
    for (float sample : audio.right)
        energy += static_cast<double>(sample);
    Consume(energy);
    return energy;
}

acoustics::MonoAudio MakeDrySignal() {
    acoustics::MonoAudio dry;
    dry.sampleRate = 48000.0f;
    dry.samples.resize(static_cast<std::size_t>(dry.sampleRate));
    constexpr double kPi = 3.14159265358979323846;
    for (std::size_t i = 0; i < dry.samples.size(); ++i) {
        const double time = static_cast<double>(i) / static_cast<double>(dry.sampleRate);
        dry.samples[i] = static_cast<float>(std::sin(2.0 * kPi * 440.0 * time));
    }
    return dry;
}

struct Row {
    uint32_t rays = 0;
    const char* phase = "";
    int repeats = 0;
    Moments moments;
    double raysPerSec = 0.0;
    std::uint64_t emitted = 0;
    std::uint64_t segments = 0;
    std::uint64_t surfaceHits = 0;
    std::uint64_t earHits = 0;
    double checksum = 0.0;
};

void PrintRow(const Row& row) {
    std::cout << std::fixed << std::setprecision(6)
              << "  " << row.phase
              << "  mean " << row.moments.mean << " ms"
              << "  median " << row.moments.median << " ms"
              << "  min " << row.moments.minimum << " ms"
              << "  stddev " << row.moments.stddev << " ms";
    if (row.raysPerSec > 0.0)
        std::cout << std::setprecision(1) << "  " << row.raysPerSec << " rays/s";
    std::cout << "\n";
}

class CsvWriter {
public:
    explicit CsvWriter(const std::optional<std::filesystem::path>& path) {
        if (!path)
            return;
        const bool exists = std::filesystem::exists(*path);
        const bool empty = !exists || std::filesystem::file_size(*path) == 0;
        stream_.open(*path, std::ios::app);
        if (!stream_)
            throw std::runtime_error("Could not open CSV: " + path->string());
        if (empty) {
            stream_ << "variant,threads,rays,seed,phase,repeats,mean_ms,median_ms,min_ms,stddev_ms,"
                       "rays_per_sec,emitted,segments,surface_hits,ear_hits,checksum\n";
        }
    }

    void Write(const Row& row) {
        if (!stream_)
            return;
        stream_ << std::fixed
                << "serial,1," << row.rays << ',' << kSeed << ',' << row.phase << ',' << row.repeats << ','
                << std::setprecision(6) << row.moments.mean << ','
                << row.moments.median << ','
                << row.moments.minimum << ','
                << row.moments.stddev << ','
                << std::setprecision(3) << row.raysPerSec << ','
                << row.emitted << ',' << row.segments << ',' << row.surfaceHits << ',' << row.earHits << ','
                << std::setprecision(17) << row.checksum << '\n';
    }

private:
    std::ofstream stream_;
};

std::vector<double> TimePhase(int warmup, int repeats, const auto& call) {
    for (int i = 0; i < warmup; ++i)
        call();
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeats));
    for (int i = 0; i < repeats; ++i) {
        const auto start = std::chrono::steady_clock::now();
        call();
        samples.push_back(ElapsedMs(start));
    }
    return samples;
}

bool RunRayCount(const Options& options, uint32_t rays, uint32_t materialId, CsvWriter& csv) {
    const acoustics::Settings settings = MakeSettings(rays);
    const acoustics::Source source = MakeSource();
    const acoustics::Receiver listener = MakeListener();
    const std::vector<acoustics::Source> sources{source};

    std::cout << "\nRays " << rays << "  seed " << kSeed
              << "  speed " << settings.speedOfSound << " m/s\n";

    AcousticSceneGeometry scratch;
    const std::vector<double> geometryMs = TimePhase(options.warmup, options.repeats, [&] {
        BuildRoom(scratch, materialId);
    });
    const double geometryChecksum = GeometrySignature(scratch);
    Row geometry;
    geometry.rays = rays;
    geometry.phase = "geometry";
    geometry.repeats = options.repeats;
    geometry.moments = Summarize(geometryMs);
    geometry.checksum = geometryChecksum;
    PrintRow(geometry);
    csv.Write(geometry);

    AcousticSceneGeometry room;
    BuildRoom(room, materialId);
    BVH bvh;
    const std::vector<double> bvhMs = TimePhase(options.warmup, options.repeats, [&] {
        bvh.Build(room);
    });
    const double bvhChecksum = BvhSignature(bvh);
    Row build;
    build.rays = rays;
    build.phase = "bvh_build";
    build.repeats = options.repeats;
    build.moments = Summarize(bvhMs);
    build.checksum = bvhChecksum;
    PrintRow(build);
    csv.Write(build);
    if (!bvh.IsValid()) {
        std::cerr << "BVH build produced an empty tree\n";
        return false;
    }

    const acoustics::Simulator simulator(bvh, AcousticMaterialLibrary::Instance());
    acoustics::Echogram echogram(settings.maxDuration);
    std::optional<Fingerprint> expected;
    auto accept = [&](const Fingerprint& fingerprint) {
        Consume(fingerprint.energy + static_cast<double>(fingerprint.stats.earHits));
        if (!expected) {
            expected = fingerprint;
            return true;
        }
        if (SameFingerprint(*expected, fingerprint))
            return true;
        std::cerr << "Result changed between passes with seed " << kSeed << " and " << rays << " rays\n"
                  << "  emitted " << expected->stats.emitted << " vs " << fingerprint.stats.emitted
                  << "  earHits " << expected->stats.earHits << " vs " << fingerprint.stats.earHits
                  << "  energy " << std::setprecision(17) << expected->energy << " vs " << fingerprint.energy << "\n";
        return false;
    };

    double lastWarmupMs = 0.0;
    for (int i = 0; i < options.warmup; ++i) {
        const auto start = std::chrono::steady_clock::now();
        const acoustics::TraceStats stats = simulator.Run(sources, listener, echogram, settings, nullptr);
        lastWarmupMs = ElapsedMs(start);
        Fingerprint fingerprint;
        fingerprint.stats = stats;
        fingerprint.energy = EchogramEnergy(echogram);
        if (!accept(fingerprint))
            return false;
    }

    const int traceRepeats = TraceRepeats(options, lastWarmupMs);
    if (options.profile && traceRepeats != options.repeats) {
        std::cout << "  profile window: warmup pass " << lastWarmupMs
                  << " ms, trace repeats " << options.repeats << " -> " << traceRepeats
                  << " (target " << kProfileTargetSec << " s, cap " << kProfileRepeatCap << ")\n";
    }

    std::vector<double> traceMs;
    traceMs.reserve(static_cast<std::size_t>(traceRepeats));
    Fingerprint last{};
    std::cout << "TRACE_BEGIN rays=" << rays << " repeats=" << traceRepeats << "\n" << std::flush;
    for (int i = 0; i < traceRepeats; ++i) {
        const auto start = std::chrono::steady_clock::now();
        const acoustics::TraceStats stats = simulator.Run(sources, listener, echogram, settings, nullptr);
        traceMs.push_back(ElapsedMs(start));
        last.stats = stats;
        last.energy = EchogramEnergy(echogram);
        if (!accept(last))
            return false;
    }
    std::cout << "TRACE_END\n";

    Row trace;
    trace.rays = rays;
    trace.phase = "trace";
    trace.repeats = traceRepeats;
    trace.moments = Summarize(traceMs);
    trace.emitted = last.stats.emitted;
    trace.segments = last.stats.segments;
    trace.surfaceHits = last.stats.surfaceHits;
    trace.earHits = last.stats.earHits;
    trace.checksum = last.energy;
    if (trace.moments.mean > 0.0)
        trace.raysPerSec = static_cast<double>(last.stats.emitted) / (trace.moments.mean / 1000.0);
    PrintRow(trace);
    std::cout << "  emitted " << last.stats.emitted
              << "  segments " << last.stats.segments
              << "  surfaceHits " << last.stats.surfaceHits
              << "  earHits " << last.stats.earHits
              << "  headOcclusions " << last.stats.headOcclusions
              << "  escaped " << last.stats.escaped
              << "  russianRoulette " << last.stats.russianRouletteKills
              << "  checksum " << std::setprecision(17) << last.energy << "\n";
    csv.Write(trace);

    if (!options.audio)
        return true;

    acoustics::AudioEngine::SynthOptions synthOptions;
    synthOptions.peakNormalize = false;
    const acoustics::MonoAudio dry = MakeDrySignal();
    acoustics::StereoAudio impulse;
    const std::vector<double> rirMs = TimePhase(options.warmup, options.repeats, [&] {
        const acoustics::SynthResult synthesized =
            acoustics::AudioEngine::SynthesizeImpulseResponse(echogram, synthOptions);
        if (!synthesized.ok)
            throw std::runtime_error(synthesized.message);
        impulse = synthesized.audio;
    });
    const double impulseChecksum = AudioEnergy(impulse);
    Row rir;
    rir.rays = rays;
    rir.phase = "rir";
    rir.repeats = options.repeats;
    rir.moments = Summarize(rirMs);
    rir.checksum = impulseChecksum;
    PrintRow(rir);
    csv.Write(rir);

    acoustics::StereoAudio wetAudio;
    const std::vector<double> convolveMs = TimePhase(options.warmup, options.repeats, [&] {
        const acoustics::ConvolveResult wet = acoustics::AudioEngine::Convolve(dry, impulse);
        if (!wet.ok)
            throw std::runtime_error(wet.message);
        wetAudio = wet.audio;
    });
    const double wetChecksum = AudioEnergy(wetAudio);
    Row convolve;
    convolve.rays = rays;
    convolve.phase = "convolve";
    convolve.repeats = options.repeats;
    convolve.moments = Summarize(convolveMs);
    convolve.checksum = wetChecksum;
    PrintRow(convolve);
    csv.Write(convolve);
    return true;
}

} // namespace

int main(int argc, char** argv) {
    spdlog::set_level(spdlog::level::warn);

    Options options;
    std::string error;
    if (!ParseArgs(argc, argv, options, error)) {
        if (!error.empty()) {
            std::cerr << error << "\n";
            PrintUsage();
            return 2;
        }
        return 0;
    }

    try {
        const uint32_t materialId = UniformMaterial();
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "AcousticsSerialBench serial\n"
                  << "Shoebox " << kRoomDimensions.x << " x " << kRoomDimensions.y << " x " << kRoomDimensions.z
                  << " m, center (" << kRoomCenter.x << ", " << kRoomCenter.y << ", " << kRoomCenter.z
                  << "), Uniform02\n"
                  << "Source omni (" << kSourcePosition.x << ", " << kSourcePosition.y << ", " << kSourcePosition.z
                  << "), power 3\n"
                  << "Listener (" << kListenerPosition.x << ", " << kListenerPosition.y << ", " << kListenerPosition.z
                  << "), head shadow on\n"
                  << "Seed " << kSeed << ", max bounces " << kMaxBounces << ", duration " << kDurationSec
                  << " s, air 20 C / 50 %\n"
                  << "Repeats " << options.repeats << ", warmup " << options.warmup
                  << (options.profile ? ", profile" : "")
                  << (options.audio ? ", audio" : ", no audio") << "\n";

        CsvWriter csv(options.csv);
        const std::vector<uint32_t> rayCounts = options.suite
            ? std::vector<uint32_t>(kSuiteRays.begin(), kSuiteRays.end())
            : std::vector<uint32_t>{options.rays};

        for (uint32_t rays : rayCounts) {
            if (!RunRayCount(options, rays, materialId, csv))
                return 1;
        }
        std::cout << "\nConsume sink " << std::setprecision(17) << g_consume << "\n";
        return 0;
    } catch (const std::exception& failure) {
        std::cerr << failure.what() << "\n";
        return 1;
    }
}
