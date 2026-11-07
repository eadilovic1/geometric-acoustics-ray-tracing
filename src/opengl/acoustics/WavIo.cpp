#include "acoustics/AudioEngine.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <fstream>
#include <limits>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <mmsystem.h>
#pragma comment(lib, "winmm.lib")
#endif

namespace acoustics {

namespace {

constexpr float kSafePeak = 0.99f;
constexpr std::uint32_t kMaxWavBytes = 64u * 1024u * 1024u;

void WriteU16(std::ostream& out, std::uint16_t value) {
    const unsigned char bytes[2] = {
        static_cast<unsigned char>(value & 0xffu),
        static_cast<unsigned char>((value >> 8) & 0xffu)
    };
    out.write(reinterpret_cast<const char*>(bytes), 2);
}

void WriteU32(std::ostream& out, std::uint32_t value) {
    const unsigned char bytes[4] = {
        static_cast<unsigned char>(value & 0xffu),
        static_cast<unsigned char>((value >> 8) & 0xffu),
        static_cast<unsigned char>((value >> 16) & 0xffu),
        static_cast<unsigned char>((value >> 24) & 0xffu)
    };
    out.write(reinterpret_cast<const char*>(bytes), 4);
}

bool ReadExact(std::istream& in, void* destination, std::size_t count) {
    in.read(reinterpret_cast<char*>(destination), static_cast<std::streamsize>(count));
    return static_cast<std::size_t>(in.gcount()) == count;
}

std::uint16_t ReadU16(const unsigned char* bytes) {
    return static_cast<std::uint16_t>(bytes[0] | (static_cast<std::uint16_t>(bytes[1]) << 8));
}

std::uint32_t ReadU32(const unsigned char* bytes) {
    return static_cast<std::uint32_t>(bytes[0])
        | (static_cast<std::uint32_t>(bytes[1]) << 8)
        | (static_cast<std::uint32_t>(bytes[2]) << 16)
        | (static_cast<std::uint32_t>(bytes[3]) << 24);
}

float PeakAbs(const std::vector<float>& samples) {
    float peak = 0.0f;
    for (float sample : samples) peak = std::max(peak, std::abs(sample));
    return peak;
}

std::int16_t QuantizePcm16(float sample) {
    const float clamped = std::clamp(sample, -1.0f, 1.0f);
    const long rounded = std::lround(static_cast<double>(clamped) * 32767.0);
    return static_cast<std::int16_t>(std::clamp(rounded, -32768L, 32767L));
}

} // namespace

WavIoStatus AudioEngine::ResampleLinear(
    const MonoAudio& input,
    float targetSampleRate,
    MonoAudio& output
) {
    WavIoStatus status;
    output = {};

    if (!(input.sampleRate > 0.0f) || !(targetSampleRate > 0.0f)
        || !std::isfinite(input.sampleRate) || !std::isfinite(targetSampleRate)) {
        status.message = "Frekvencija uzorkovanja za prevzorcavanje nije ispravna.";
        return status;
    }
    if (!input.IsFinite()) {
        status.message = "Ulaz za prevzorcavanje nije konacan.";
        return status;
    }

    output.sampleRate = targetSampleRate;
    if (input.samples.empty()) {
        status.ok = true;
        status.message = "Prazan signal ostaje prazan.";
        return status;
    }
    if (std::abs(input.sampleRate - targetSampleRate) <= 0.01f) {
        output.samples = input.samples;
        status.ok = true;
        status.message = "Frekvencija je vec uskladena.";
        return status;
    }

    const double step = static_cast<double>(input.sampleRate) / static_cast<double>(targetSampleRate);
    const double scaled = static_cast<double>(input.samples.size()) * static_cast<double>(targetSampleRate)
        / static_cast<double>(input.sampleRate);
    std::size_t count = static_cast<std::size_t>(std::llround(scaled));
    if (count == 0) count = 1;

    output.samples.resize(count);
    const std::size_t last = input.samples.size() - 1;
    for (std::size_t i = 0; i < count; ++i) {
        const double position = std::min(static_cast<double>(i) * step, static_cast<double>(last));
        const std::size_t index = static_cast<std::size_t>(position);
        const std::size_t next = std::min(index + 1, last);
        const double fraction = position - static_cast<double>(index);
        const double sample = static_cast<double>(input.samples[index]) * (1.0 - fraction)
            + static_cast<double>(input.samples[next]) * fraction;
        output.samples[i] = static_cast<float>(sample);
    }

    status.ok = true;
    status.message = "Linearno prevzorcavanje je zavrseno.";
    return status;
}

WavData AudioEngine::ReadWav(const std::filesystem::path& path) {
    WavData data;
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        data.message = "WAV datoteka se ne moze otvoriti.";
        return data;
    }

    unsigned char riff[12] = {};
    if (!ReadExact(input, riff, 12)
        || riff[0] != 'R' || riff[1] != 'I' || riff[2] != 'F' || riff[3] != 'F'
        || riff[8] != 'W' || riff[9] != 'A' || riff[10] != 'V' || riff[11] != 'E') {
        data.message = "Datoteka nije RIFF/WAVE.";
        return data;
    }

    bool haveFormat = false;
    std::uint16_t audioFormat = 0;
    std::uint16_t channels = 0;
    std::uint32_t sampleRate = 0;
    std::uint16_t bitsPerSample = 0;
    std::vector<unsigned char> payload;

    while (input && !input.eof()) {
        unsigned char header[8] = {};
        if (!ReadExact(input, header, 8)) break;
        const std::uint32_t chunkSize = ReadU32(header + 4);
        if (chunkSize > kMaxWavBytes) {
            data.message = "WAV chunk je prevelik.";
            return data;
        }

        std::vector<unsigned char> chunk(chunkSize);
        if (chunkSize > 0 && !ReadExact(input, chunk.data(), chunkSize)) {
            data.message = "WAV chunk je nepotpun.";
            return data;
        }
        if ((chunkSize & 1u) != 0u) input.get();

        const bool isFormat = header[0] == 'f' && header[1] == 'm' && header[2] == 't' && header[3] == ' ';
        const bool isData = header[0] == 'd' && header[1] == 'a' && header[2] == 't' && header[3] == 'a';
        if (isFormat) {
            if (chunkSize < 16) {
                data.message = "fmt chunk je prekratak.";
                return data;
            }
            audioFormat = ReadU16(chunk.data());
            channels = ReadU16(chunk.data() + 2);
            sampleRate = ReadU32(chunk.data() + 4);
            bitsPerSample = ReadU16(chunk.data() + 14);
            haveFormat = true;
        } else if (isData) {
            payload = std::move(chunk);
        }
    }

    if (!haveFormat || payload.empty()) {
        data.message = "WAV nema fmt ili data chunk.";
        return data;
    }
    if (channels == 0 || sampleRate == 0) {
        data.message = "WAV format nije ispravan.";
        return data;
    }

    const bool pcm16 = audioFormat == 1 && bitsPerSample == 16;
    const bool float32 = audioFormat == 3 && bitsPerSample == 32;
    if (!pcm16 && !float32) {
        data.message = "Podrzani su samo PCM16 i IEEE float32.";
        return data;
    }

    const std::size_t bytesPerFrame = static_cast<std::size_t>(channels) * (bitsPerSample / 8u);
    if (bytesPerFrame == 0 || payload.size() < bytesPerFrame || (payload.size() % bytesPerFrame) != 0) {
        data.message = "Duzina WAV podataka nije cijeli broj okvira.";
        return data;
    }

    const std::size_t frames = payload.size() / bytesPerFrame;
    data.interleaved.resize(frames * channels);
    for (std::size_t frame = 0; frame < frames; ++frame) {
        for (std::uint16_t channel = 0; channel < channels; ++channel) {
            const std::size_t offset = (frame * channels + channel) * (bitsPerSample / 8u);
            float sample = 0.0f;
            if (pcm16) {
                const std::int16_t raw = static_cast<std::int16_t>(ReadU16(payload.data() + offset));
                sample = static_cast<float>(raw) / 32768.0f;
            } else {
                static_assert(sizeof(float) == 4, "IEEE float32 WAV trazi 4-bajtni float");
                float raw = 0.0f;
                std::copy_n(payload.data() + offset, 4, reinterpret_cast<unsigned char*>(&raw));
                sample = raw;
            }
            if (!std::isfinite(sample)) {
                data.interleaved.clear();
                data.message = "WAV sadrzi nekonacnu vrijednost.";
                return data;
            }
            data.interleaved[frame * channels + channel] = sample;
        }
    }

    data.ok = true;
    data.sampleRate = static_cast<float>(sampleRate);
    data.channels = static_cast<int>(channels);
    data.bitsPerSample = static_cast<int>(bitsPerSample);
    data.message = "WAV je ucitan.";
    return data;
}

WavIoStatus AudioEngine::ReadMonoWav(const std::filesystem::path& path, MonoAudio& out) {
    WavIoStatus status;
    out = {};
    const WavData data = ReadWav(path);
    if (!data.ok) {
        status.message = data.message;
        return status;
    }
    if (data.channels != 1) {
        status.message = "Samo mono WAV je prihvacen kao suhi signal.";
        return status;
    }

    out.sampleRate = data.sampleRate;
    out.samples = data.interleaved;
    status.ok = true;
    status.message = "Mono WAV je ucitan.";
    return status;
}

WavIoStatus AudioEngine::WriteStereoWav(const std::filesystem::path& path, const StereoAudio& audio) {
    WavIoStatus status;
    if (!(audio.sampleRate > 0.0f) || !std::isfinite(audio.sampleRate)) {
        status.message = "Frekvencija stereo signala nije ispravna.";
        return status;
    }
    if (!audio.IsFinite()) {
        status.message = "Stereo signal nije konacan i nije snimljen.";
        return status;
    }

    const std::size_t frames = std::max(audio.left.size(), audio.right.size());
    if (frames == 0) {
        status.message = "Prazan stereo signal nije snimljen.";
        return status;
    }
    if (frames > (kMaxWavBytes / 4u)) {
        status.message = "Stereo signal je prevelik za WAV zapis.";
        return status;
    }

    // Oba kanala dijele isto slabljenje. Tih signal se ne pojacava.
    const float peak = std::max(PeakAbs(audio.left), PeakAbs(audio.right));
    const float gain = (peak > kSafePeak) ? (kSafePeak / peak) : 1.0f;
    const auto roundedRate = std::lround(static_cast<double>(audio.sampleRate));
    if (roundedRate <= 0 || roundedRate > 384000) {
        status.message = "Frekvencija se ne moze zapisati u WAV zaglavlje.";
        return status;
    }

    if (!path.parent_path().empty()) {
        std::error_code error;
        std::filesystem::create_directories(path.parent_path(), error);
        if (error) {
            status.message = "Izlazni direktorij se ne moze kreirati.";
            return status;
        }
    }

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        status.message = "WAV datoteka se ne moze kreirati.";
        return status;
    }

    const std::uint32_t dataBytes = static_cast<std::uint32_t>(frames * 4u);
    const std::uint32_t sampleRate = static_cast<std::uint32_t>(roundedRate);
    output.write("RIFF", 4);
    WriteU32(output, 36u + dataBytes);
    output.write("WAVE", 4);
    output.write("fmt ", 4);
    WriteU32(output, 16u);
    WriteU16(output, 1u);
    WriteU16(output, 2u);
    WriteU32(output, sampleRate);
    WriteU32(output, sampleRate * 4u);
    WriteU16(output, 4u);
    WriteU16(output, 16u);
    output.write("data", 4);
    WriteU32(output, dataBytes);

    for (std::size_t frame = 0; frame < frames; ++frame) {
        const float left = frame < audio.left.size() ? audio.left[frame] * gain : 0.0f;
        const float right = frame < audio.right.size() ? audio.right[frame] * gain : 0.0f;
        const std::int16_t leftPcm = QuantizePcm16(left);
        const std::int16_t rightPcm = QuantizePcm16(right);
        WriteU16(output, static_cast<std::uint16_t>(leftPcm));
        WriteU16(output, static_cast<std::uint16_t>(rightPcm));
    }

    if (!output.good()) {
        status.message = "Pisanje WAV datoteke nije uspjelo.";
        return status;
    }

    status.ok = true;
    status.message = peak > kSafePeak
        ? "Stereo WAV je snimljen uz zajednicko slabljenje kanala."
        : "Stereo WAV je snimljen.";
    return status;
}

PlaybackResult AudioEngine::PlayWavFile(const std::filesystem::path& path) {
    PlaybackResult result;
#if !defined(_WIN32)
    result.status = PlaybackStatus::Unavailable;
    result.message = "WinMM PlaySound postoji samo na Windows platformi. miniaudio nije dodan.";
    return result;
#else
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error)) {
        result.status = PlaybackStatus::Failed;
        result.message = "Datoteka za reprodukciju ne postoji.";
        return result;
    }

    const BOOL started = PlaySoundW(
        path.c_str(),
        nullptr,
        SND_FILENAME | SND_ASYNC | SND_NODEFAULT
    );
    if (!started) {
        result.status = PlaybackStatus::Failed;
        result.message = "WinMM PlaySound nije uspio pokrenuti datoteku.";
        return result;
    }

    result.status = PlaybackStatus::Played;
    result.message = "Reprodukcija je pokrenuta asinhrono preko WinMM.";
    return result;
#endif
}

DemoOutput AudioEngine::RunSyntheticDemo(const std::filesystem::path& wavPath, bool playAudio) {
    DemoOutput demo;
    constexpr float kSampleRate = 48000.0f;
    constexpr float kDurationSec = 0.20f;
    constexpr double kRt60Sec = 0.40;

    try {
        Echogram echogram(kDurationSec, kSampleRate);
        const double tau = kRt60Sec / std::log(1.0e6);
        const std::size_t delay = static_cast<std::size_t>(0.001f * kSampleRate);
        for (std::size_t n = 0; n < echogram.BinCount(); ++n) {
            const float energy = static_cast<float>(
                std::exp(-static_cast<double>(n) / (static_cast<double>(kSampleRate) * tau))
            );
            const OctaveBands bands(energy);
            echogram.DepositByIndex(true, n, bands);
            if (n + delay < echogram.BinCount()) {
                echogram.DepositByIndex(false, n + delay, bands * 0.6f);
            }
        }

        demo.metrics = RoomMetrics::Compute(echogram);
        const SynthResult impulse = SynthesizeImpulseResponse(echogram);
        if (!impulse.ok) {
            demo.message = impulse.message;
            return demo;
        }

        MonoAudio dry;
        dry.sampleRate = kSampleRate;
        constexpr int kToneSamples = 480;
        dry.samples.resize(kToneSamples);
        constexpr float kPi = 3.14159265f;
        for (int n = 0; n < kToneSamples; ++n) {
            const float envelope = 1.0f - static_cast<float>(n) / static_cast<float>(kToneSamples);
            dry.samples[static_cast<std::size_t>(n)] =
                0.2f * envelope * std::sin(2.0f * kPi * 440.0f * static_cast<float>(n) / kSampleRate);
        }

        const ConvolveResult wet = Convolve(dry, impulse.audio);
        if (!wet.ok) {
            demo.message = wet.message;
            return demo;
        }

        const WavIoStatus written = WriteStereoWav(wavPath, wet.audio);
        if (!written.ok) {
            demo.message = written.message;
            return demo;
        }

        demo.ok = true;
        demo.wavPath = wavPath;
        demo.sampleRate = wet.audio.sampleRate;
        demo.frameCount = wet.audio.left.size();
        demo.message = "Sinteticki ehogram je pretvoren u stereo WAV.";
        if (playAudio) {
            demo.playback = PlayWavFile(wavPath);
        } else {
            demo.playback.status = PlaybackStatus::Unavailable;
            demo.playback.message = "Reprodukcija nije trazena.";
        }
        return demo;
    } catch (const std::exception& exception) {
        demo.ok = false;
        demo.message = exception.what();
        return demo;
    }
}

} // namespace acoustics
