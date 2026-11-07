#include "acoustics/AudioEngine.h"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>

namespace {

void PrintUsage() {
    std::cout
        << "Upotreba:\n"
        << "  AuralizationDemo.exe <suhi-mono.wav> [--play]\n\n"
        << "Ucitava mono PCM16 ili float32 WAV, gradi sinteticki ehogram\n"
        << "prostorije, pravi stereo impulsni odziv i snima output-reverb.wav.\n"
        << "Za poredjenje snima i output-dry.wav (isti signal, bez odjeka).\n"
        << "--play pusti prvo suhi, pa obradjeni snimak.\n";
}

acoustics::Echogram MakeNoticeableRoom(float sampleRate) {
    constexpr float kDurationSec = 1.8f;
    constexpr double kRt60Sec = 1.50;
    acoustics::Echogram echogram(kDurationSec, sampleRate);

    const double fs = static_cast<double>(sampleRate);
    const double tau = kRt60Sec / std::log(1.0e6);

    const auto deposit = [&](bool left, double timeSec, float gain) {
        if (!(timeSec >= 0.0) || !(gain > 0.0f)) return;
        const std::size_t index = static_cast<std::size_t>(timeSec * fs);
        if (index < echogram.BinCount()) {
            echogram.DepositByIndex(left, index, OctaveBands(gain));
        }
    };

    deposit(true, 0.000, 1.00f);
    deposit(false, 0.00045, 0.90f);

    const double leftReflections[][2] = {
        {0.018, 0.46}, {0.031, 0.30}, {0.049, 0.20}, {0.067, 0.13}, {0.084, 0.09}
    };
    const double rightReflections[][2] = {
        {0.016, 0.34}, {0.027, 0.38}, {0.043, 0.18}, {0.062, 0.15}, {0.091, 0.08}
    };
    for (const auto& reflection : leftReflections) deposit(true, reflection[0], static_cast<float>(reflection[1]));
    for (const auto& reflection : rightReflections) deposit(false, reflection[0], static_cast<float>(reflection[1]));

    constexpr double kLateStartSec = 0.050;
    for (std::size_t n = 0; n < echogram.BinCount(); ++n) {
        const double time = static_cast<double>(n) / fs;
        if (time < kLateStartSec) continue;
        const float late = static_cast<float>(0.22 * std::exp(-(time - kLateStartSec) / tau));
        echogram.DepositByIndex(true, n, OctaveBands(late));
        echogram.DepositByIndex(false, n, OctaveBands(late * 0.82f));
    }
    return echogram;
}

float ChooseImpulseRate(float dryRate) {
    if (dryRate >= 32000.0f && std::isfinite(dryRate)) return dryRate;
    return 48000.0f;
}

void PrintSeconds(const char* label, std::size_t frames, float sampleRate) {
    const double seconds = sampleRate > 0.0f
        ? static_cast<double>(frames) / static_cast<double>(sampleRate)
        : 0.0;
    std::cout << label << seconds << " s (" << frames << " uzoraka, " << sampleRate << " Hz)\n";
}

int Fail(const std::string& message) {
    std::cerr << message << "\n";
    return 1;
}

} // namespace

int main(int argc, char** argv) {
    std::filesystem::path inputPath;
    bool play = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--play") {
            play = true;
        } else if (arg == "--help" || arg == "-h" || arg == "/?") {
            PrintUsage();
            return 0;
        } else if (inputPath.empty()) {
            inputPath = arg;
        } else {
            PrintUsage();
            return Fail("Nepoznat argument: " + arg);
        }
    }

    if (inputPath.empty()) {
        PrintUsage();
        return 1;
    }

    acoustics::MonoAudio dry;
    const acoustics::WavIoStatus loaded = acoustics::AudioEngine::ReadMonoWav(inputPath, dry);
    if (!loaded.ok) return Fail(loaded.message);

    std::cout << "Ulaz: " << inputPath.string() << "\n";
    PrintSeconds("  suhi signal: ", dry.samples.size(), dry.sampleRate);

    const float impulseRate = ChooseImpulseRate(dry.sampleRate);
    acoustics::Echogram echogram = MakeNoticeableRoom(impulseRate);
    std::cout << "Sinteticki ehogram: direktni zvuk, rane refleksije i rep od 1.5 s\n";

    std::cout << "Sinteza stereo impulsnog odziva...\n";
    const acoustics::SynthResult impulse = acoustics::AudioEngine::SynthesizeImpulseResponse(echogram);
    if (!impulse.ok) return Fail(impulse.message);
    PrintSeconds("  impulsni odziv: ", impulse.audio.left.size(), impulse.audio.sampleRate);

    std::cout << "FFT konvolucija...\n";
    const acoustics::ConvolveResult wet = acoustics::AudioEngine::Convolve(dry, impulse.audio);
    if (!wet.ok) return Fail(wet.message);

    const std::filesystem::path reverbPath = std::filesystem::current_path() / "output-reverb.wav";
    const std::filesystem::path dryPath = std::filesystem::current_path() / "output-dry.wav";

    const acoustics::WavIoStatus reverbWritten = acoustics::AudioEngine::WriteStereoWav(reverbPath, wet.audio);
    if (!reverbWritten.ok) return Fail(reverbWritten.message);

    acoustics::MonoAudio aligned = dry;
    if (std::abs(dry.sampleRate - wet.audio.sampleRate) > 0.01f) {
        const acoustics::WavIoStatus resampled = acoustics::AudioEngine::ResampleLinear(dry, wet.audio.sampleRate, aligned);
        if (!resampled.ok) return Fail(resampled.message);
    }

    acoustics::StereoAudio dryStereo;
    dryStereo.sampleRate = aligned.sampleRate;
    dryStereo.left = aligned.samples;
    dryStereo.right = aligned.samples;
    const acoustics::WavIoStatus dryWritten = acoustics::AudioEngine::WriteStereoWav(dryPath, dryStereo);
    if (!dryWritten.ok) return Fail(dryWritten.message);

    PrintSeconds("  obradjeni signal: ", wet.audio.left.size(), wet.audio.sampleRate);
    std::cout << "Snimljeno:\n";
    std::cout << "  suhi:      " << dryPath.string() << "\n";
    std::cout << "  odjek:     " << reverbPath.string() << "\n";
    std::cout << "Uporedite output-dry.wav i output-reverb.wav u istom plejeru.\n";
    std::cout << reverbWritten.message << "\n";

    if (!play) return 0;

    std::cout << "Reprodukcija suhog signala...\n";
    const acoustics::PlaybackResult dryPlayback = acoustics::AudioEngine::PlayWavFile(dryPath);
    std::cout << dryPlayback.message << "\n";
    if (dryPlayback.status != acoustics::PlaybackStatus::Played) return 1;

    const double drySeconds = static_cast<double>(aligned.samples.size()) / static_cast<double>(aligned.sampleRate);
    std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(std::ceil((drySeconds + 0.4) * 1000.0))));

    std::cout << "Reprodukcija signala sa odjekom...\n";
    const acoustics::PlaybackResult wetPlayback = acoustics::AudioEngine::PlayWavFile(reverbPath);
    std::cout << wetPlayback.message << "\n";
    if (wetPlayback.status != acoustics::PlaybackStatus::Played) return 1;

    const double wetSeconds = static_cast<double>(wet.audio.left.size()) / static_cast<double>(wet.audio.sampleRate);
    std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(std::ceil((wetSeconds + 0.4) * 1000.0))));
    return 0;
}
