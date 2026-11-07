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

    float NextGaussian() noexcept {
        if (hasCached_) {
            hasCached_ = false;
            return cached_;
        }
        const float u1 = NextUniformPositive();
        const float u2 = NextUniformPositive();
        const float radius = std::sqrt(-2.0f * std::log(u1));
        constexpr float kTwoPi = 6.28318530717958647692f;
        const float theta = kTwoPi * u2;
        cached_ = radius * std::sin(theta);
        hasCached_ = true;
        return radius * std::cos(theta);
    }

private:
    uint32_t state_;
    float cached_ = 0.0f;
    bool hasCached_ = false;

    float NextUniformPositive() noexcept {
        uint32_t x = state_;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        state_ = x;
        // Strogo pozitivan broj u (0, 1] kako bi std::log(u1) bio uvijek definisan
        return (static_cast<float>(x & 0x00FFFFFFu) + 1.0f) / 16777218.0f;
    }
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

        for (std::size_t n = 0; n < count; ++n) {
            const float energy = bins[n][band];
            if (!std::isfinite(energy)) {
                finite = false;
                continue;
            }
            // 1. Obvojnica pritiska: A_b[n] = sqrt(max(E_b[n], 0))
            const double amplitude = std::sqrt(static_cast<double>(std::max(0.0f, energy)));
            // 2. Modulacija Gaussovim bijelim sumom: s_b[n] = A_b[n] * w_b[n], w_b ~ N(0, 1)
            const float w = noise.NextGaussian();
            const float modulated = static_cast<float>(amplitude * static_cast<double>(w));
            // 3. Filtriranje modulisanog signala kroz Butterworth propusnik 4. reda
            const float filtered = filter.Process(modulated);
            // 4. Sabiranje po opsezima
            accum[n] += static_cast<double>(filtered);
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

SynthResult AudioEngine::SynthesizeImpulseResponse(const Echogram& echogram) {
    return SynthesizeImpulseResponse(echogram, SynthOptions{});
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
