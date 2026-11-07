#include "acoustics/BandpassFilter.h"

#include <cmath>
#include <complex>

namespace acoustics {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kSqrt2 = 1.4142135623730951;
constexpr double kButterworthQ = 0.7071067811865476;

struct RawSection {
    double b0 = 1.0;
    double b1 = 0.0;
    double b2 = 0.0;
    double a1 = 0.0;
    double a2 = 0.0;
};

RawSection DesignBiquad(double sampleRate, double frequency, double q, bool highpass) {
    RawSection section;
    const double w0 = 2.0 * kPi * frequency / sampleRate;
    const double cosine = std::cos(w0);
    const double sine = std::sin(w0);
    const double alpha = sine / (2.0 * q);
    const double a0 = 1.0 + alpha;

    if (highpass) {
        section.b0 = (1.0 + cosine) * 0.5;
        section.b1 = -(1.0 + cosine);
        section.b2 = (1.0 + cosine) * 0.5;
    } else {
        section.b0 = (1.0 - cosine) * 0.5;
        section.b1 = 1.0 - cosine;
        section.b2 = (1.0 - cosine) * 0.5;
    }
    section.a1 = -2.0 * cosine;
    section.a2 = 1.0 - alpha;

    const double invA0 = 1.0 / a0;
    section.b0 *= invA0;
    section.b1 *= invA0;
    section.b2 *= invA0;
    section.a1 *= invA0;
    section.a2 *= invA0;
    return section;
}

std::complex<double> Response(const RawSection& section, double omega) {
    const std::complex<double> z1 = std::polar(1.0, -omega);
    const std::complex<double> z2 = z1 * z1;
    const std::complex<double> numerator = section.b0 + section.b1 * z1 + section.b2 * z2;
    const std::complex<double> denominator = 1.0 + section.a1 * z1 + section.a2 * z2;
    return numerator / denominator;
}

} // namespace

double ButterworthBandpass::Section::Process(double x) noexcept {
    const double y = b0 * x + z1;
    z1 = b1 * x - a1 * y + z2;
    z2 = b2 * x - a2 * y;
    return y;
}

ButterworthBandpass::ButterworthBandpass(float sampleRate, float centerFrequencyHz) {
    if (!(sampleRate > 0.0f) || !(centerFrequencyHz > 0.0f) || !std::isfinite(sampleRate) || !std::isfinite(centerFrequencyHz)) {
        return;
    }

    const double fs = static_cast<double>(sampleRate);
    const double center = static_cast<double>(centerFrequencyHz);
    if (center >= fs * 0.5) return;

    double low = center / kSqrt2;
    double high = center * kSqrt2;
    const double maxCutoff = fs * 0.45;
    if (high > maxCutoff) high = maxCutoff;
    if (low < 1.0) low = 1.0;
    if (!(low + 1.0 < high) || high >= fs * 0.5) return;

    const RawSection hp = DesignBiquad(fs, low, kButterworthQ, true);
    const RawSection lp = DesignBiquad(fs, high, kButterworthQ, false);
    highpass_.b0 = hp.b0;
    highpass_.b1 = hp.b1;
    highpass_.b2 = hp.b2;
    highpass_.a1 = hp.a1;
    highpass_.a2 = hp.a2;
    lowpass_.b0 = lp.b0;
    lowpass_.b1 = lp.b1;
    lowpass_.b2 = lp.b2;
    lowpass_.a1 = lp.a1;
    lowpass_.a2 = lp.a2;

    const double omega = 2.0 * kPi * center / fs;
    const double magnitude = std::abs(Response(hp, omega) * Response(lp, omega));
    if (!(magnitude > 1e-8) || !std::isfinite(magnitude)) return;

    outputGain_ = 1.0 / magnitude;
    lowHz_ = static_cast<float>(low);
    highHz_ = static_cast<float>(high);
    valid_ = std::isfinite(outputGain_);
}

void ButterworthBandpass::Reset() noexcept {
    highpass_.z1 = highpass_.z2 = 0.0;
    lowpass_.z1 = lowpass_.z2 = 0.0;
}

float ButterworthBandpass::Process(float sample) noexcept {
    if (!valid_ || !std::isfinite(sample)) return 0.0f;
    const double highpassed = highpass_.Process(static_cast<double>(sample));
    const double bandpassed = lowpass_.Process(highpassed);
    const double scaled = bandpassed * outputGain_;
    if (!std::isfinite(scaled)) return 0.0f;
    return static_cast<float>(scaled);
}

} // namespace acoustics
