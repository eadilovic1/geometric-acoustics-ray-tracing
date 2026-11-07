#include "acoustics/AudioEngine.h"
#include "acoustics/BandpassFilter.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace acoustics {

namespace {

class DeterministicNoise {
public:
    explicit DeterministicNoise(uint32_t seed) : state_(seed == 0u ? 0xA341316Cu : seed) {}

    float Next() noexcept {
        uint32_t x = state_;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        state_ = x;
        constexpr float kScale = 1.0f / 2147483648.0f;
        return static_cast<float>(static_cast<int32_t>(x)) * kScale;
    }

private:
    uint32_t state_;
};

bool ChannelFinite(const std::vector<float>& samples) {
    for (float sample : samples) {
        if (!std::isfinite(sample)) return false;
    }
    return true;
}

void SynthesizeChannel(
    const std::vector<OctaveBands>& bins,
    float sampleRate,
    uint32_t seed,
    std::vector<float>& destination,
    bool& finite
) {
    const std::size_t count = bins.size();
    destination.assign(count, 0.0f);
    if (count == 0) return;

    std::vector<double> accum(count, 0.0);
    DeterministicNoise noise(seed);

    for (std::size_t band = 0; band < OctaveBands::Count; ++band) {
        const float center = OctaveBands::Frequencies[band];
        ButterworthBandpass filter(sampleRate, center);
        if (!filter.IsValid()) continue;

        std::vector<float> carrier(count, 0.0f);
        double sumSquares = 0.0;
        for (std::size_t n = 0; n < count; ++n) {
            const float sample = filter.Process(noise.Next());
            carrier[n] = sample;
            sumSquares += static_cast<double>(sample) * static_cast<double>(sample);
        }

        const double rms = std::sqrt(sumSquares / static_cast<double>(count));
        if (!(rms > 1e-12) || !std::isfinite(rms)) continue;
        const double scale = 1.0 / rms;

        for (std::size_t n = 0; n < count; ++n) {
            const float energy = bins[n][band];
            if (!std::isfinite(energy)) {
                finite = false;
                continue;
            }
            const double amplitude = std::sqrt(static_cast<double>(std::max(0.0f, energy)));
            accum[n] += static_cast<double>(carrier[n]) * scale * amplitude;
        }
    }

    for (std::size_t n = 0; n < count; ++n) {
        if (!std::isfinite(accum[n])) {
            finite = false;
            destination[n] = 0.0f;
        } else {
            destination[n] = static_cast<float>(accum[n]);
        }
    }
}

void PeakNormalize(StereoAudio& audio, float peakTarget) {
    if (!(peakTarget > 0.0f) || !std::isfinite(peakTarget)) return;
    float peak = 0.0f;
    for (float sample : audio.left) peak = std::max(peak, std::abs(sample));
    for (float sample : audio.right) peak = std::max(peak, std::abs(sample));
    if (!(peak > 0.0f) || !std::isfinite(peak)) return;

    const float gain = peakTarget / peak;
    for (float& sample : audio.left) sample *= gain;
    for (float& sample : audio.right) sample *= gain;
}

} // namespace

bool StereoAudio::IsFinite() const {
    return ChannelFinite(left) && ChannelFinite(right);
}

SynthResult AudioEngine::SynthesizeImpulseResponse(const Echogram& echogram, const SynthOptions& options) {
    SynthResult result;
    result.audio.sampleRate = echogram.sampleRate;

    if (!(echogram.sampleRate > 0.0f) || !std::isfinite(echogram.sampleRate)) {
        result.message = "Frekvencija uzorkovanja ehograma nije ispravna.";
        return result;
    }
    if (echogram.left.empty() && echogram.right.empty()) {
        result.message = "Ehogram je prazan.";
        return result;
    }

    bool finite = true;
    SynthesizeChannel(echogram.left, echogram.sampleRate, options.leftSeed, result.audio.left, finite);
    SynthesizeChannel(echogram.right, echogram.sampleRate, options.rightSeed, result.audio.right, finite);

    if (!finite || !result.audio.IsFinite()) {
        result.audio.left.clear();
        result.audio.right.clear();
        result.message = "Sinteza impulsnog odziva nije ostala konacna.";
        return result;
    }

    if (options.peakNormalize) {
        PeakNormalize(result.audio, options.peakTarget);
    }

    result.ok = true;
    result.message = "Impulsni odziv je rekonstruisan iz ehograma.";
    return result;
}

} // namespace acoustics
