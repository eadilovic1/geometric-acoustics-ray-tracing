#pragma once

#include "acoustics/RoomMetrics.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace acoustics {

// ---------------------------------------------------------------------------
// Korak 6: sinteza stereo impulsnog odziva i auralizacija.
//
// miniaudio nije dio repozitorija i nije dodavan. Citanje i pisanje WAV-a
// te FFT overlap-add su vlastita implementacija. Reprodukcija koristi
// sistemski WinMM PlaySound i ne uvodi novu biblioteku. Ako PlaySound nije
// dostupan, sinteza, konvolucija i WAV i dalje rade.
//
// Povezivanje sa korakom 4, bez izmjene Simulator-a:
//
//   const StereoAudio ir = AudioEngine::SynthesizeImpulseResponse(echogram).audio;
//   const auto wet = AudioEngine::Convolve(dryMono, ir);
//   AudioEngine::WriteStereoWav(path, wet.audio);
// ---------------------------------------------------------------------------

struct MonoAudio {
    float sampleRate = 48000.0f;
    std::vector<float> samples;

    [[nodiscard]] bool IsFinite() const;
};

struct StereoAudio {
    float sampleRate = 48000.0f;
    std::vector<float> left;
    std::vector<float> right;

    [[nodiscard]] bool IsFinite() const;
};

struct WavData {
    bool ok = false;
    std::string message;
    float sampleRate = 0.0f;
    int channels = 0;
    int bitsPerSample = 0;
    std::vector<float> interleaved;
};

struct WavIoStatus {
    bool ok = false;
    std::string message;
};

enum class PlaybackStatus {
    Played,
    Unavailable,
    Failed
};

struct PlaybackResult {
    PlaybackStatus status = PlaybackStatus::Failed;
    std::string message;
};

struct SynthResult {
    StereoAudio audio;
    bool ok = false;
    std::string message;
};

struct ConvolveResult {
    StereoAudio audio;
    bool ok = false;
    std::string message;
};

struct DemoOutput {
    bool ok = false;
    std::string message;
    std::filesystem::path wavPath;
    float sampleRate = 0.0f;
    std::size_t frameCount = 0;
    RoomMetricsResult metrics;
    PlaybackResult playback;
};

class AudioEngine {
public:
    struct SynthOptions {
        uint32_t leftSeed = 0xA341316Cu;
        uint32_t rightSeed = 0x00C0FFEEu;
        bool peakNormalize = true;
        float peakTarget = 0.99f;
    };

    /// Rekonstrukcija h_L / h_R iz ehograma.
    ///
    /// Za svaki od 7 oktavnih opsega (125 .. 8000 Hz):
    ///   A_b[n] = sqrt(max(E_b[n], 0))
    ///   w_b    = Butterworth4(bijeli sum; razlicit seed za L i R)
    ///   w_b se normalizuje na jedinicni RMS, zatim h += A_b * w_b
    ///
    /// Ogranicenje stohasticke rekonstrukcije: faza nosioca je slucajna.
    /// Direktni zvuk i rane refleksije zato nisu deterministicki impulsi,
    /// nego kratki sum u vremenskom binu ehograma. Obvojnica se primjenjuje
    /// nakon filtriranja, pa vremenski raspored energije ostaje na uzorcima
    /// ehograma. Ostar prekid obvojnice blago prosiri spektar oko tog bina.
    [[nodiscard]] static SynthResult SynthesizeImpulseResponse(
        const Echogram& echogram,
        const SynthOptions& options = {}
    );

    /// y_L = x * h_L, y_R = x * h_R, FFT overlap-add.
    /// Ako se frekvencije razlikuju, suhi signal se linearno prevzorkuje na
    /// frekvenciju impulsnog odziva. Izlaz NIJE normiran; nivo se ogranicava
    /// tek pri pisanju WAV-a, da testovi konvolucije vide linearni rezultat.
    [[nodiscard]] static ConvolveResult Convolve(
        const MonoAudio& dry,
        const StereoAudio& impulseResponse,
        std::size_t blockSize = 2048
    );

    [[nodiscard]] static WavData ReadWav(const std::filesystem::path& path);

    /// Prihvata samo mono PCM16 ili mono IEEE float32.
    [[nodiscard]] static WavIoStatus ReadMonoWav(
        const std::filesystem::path& path,
        MonoAudio& out
    );

    /// Stereo PCM16. Ako je vrhunac iznad 0.99, oba kanala dijele isto slabljenje.
    [[nodiscard]] static WavIoStatus WriteStereoWav(
        const std::filesystem::path& path,
        const StereoAudio& audio
    );

    [[nodiscard]] static WavIoStatus ResampleLinear(
        const MonoAudio& input,
        float targetSampleRate,
        MonoAudio& output
    );

    [[nodiscard]] static PlaybackResult PlayWavFile(const std::filesystem::path& path);

    /// Sinteticki ehogram + kratki sinus, bez vezivanja za korake 1-4.
    /// playAudio pokrece WinMM samo kada je eksplicitno trazeno.
    [[nodiscard]] static DemoOutput RunSyntheticDemo(
        const std::filesystem::path& wavPath,
        bool playAudio = false
    );
};

} // namespace acoustics
