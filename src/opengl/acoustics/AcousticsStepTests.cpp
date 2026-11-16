#include "acoustics/AcousticGeometry.h"
#include "acoustics/AcousticSimulation.h"
#include "acoustics/AudioEngine.h"
#include "acoustics/BVH.h"
#include "acoustics/BandpassFilter.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

int g_failed = 0;

void Expect(bool condition, const std::string& name) {
    if (condition) {
        std::cout << "PASS " << name << "\n";
        return;
    }
    std::cout << "FAIL " << name << "\n";
    ++g_failed;
}

bool Near(double actual, double expected, double tolerance) {
    return std::isfinite(actual) && std::isfinite(expected) && std::abs(actual - expected) <= tolerance;
}

acoustics::Echogram MakeEchogram(float durationSec, float sampleRate) {
    return acoustics::Echogram(durationSec, sampleRate);
}

void FillBand(std::vector<OctaveBands>& bins, std::size_t band, const std::vector<double>& energy) {
    if (bins.size() < energy.size()) bins.resize(energy.size());
    for (std::size_t i = 0; i < energy.size(); ++i) {
        bins[i][band] = static_cast<float>(energy[i]);
    }
}

std::vector<double> ExponentialEnergy(std::size_t count, double sampleRate, double rt60Sec) {
    const double tau = rt60Sec / std::log(1.0e6);
    std::vector<double> energy(count, 0.0);
    for (std::size_t n = 0; n < count; ++n) {
        energy[n] = std::exp(-static_cast<double>(n) / (sampleRate * tau));
    }
    return energy;
}

bool MonotoneNonIncreasing(const std::vector<double>& values) {
    for (std::size_t i = 1; i < values.size(); ++i) {
        if (!std::isfinite(values[i]) || values[i] > values[i - 1] + 1e-9) return false;
    }
    return !values.empty();
}

void TestBackwardIntegration() {
    auto echogram = MakeEchogram(0.003f, 1000.0f);
    FillBand(echogram.left, 0, {1.0, 2.0, 3.0});
    const auto metrics = acoustics::RoomMetrics::Compute(echogram);
    const auto& decay = metrics.left.bands[0].decay;

    Expect(decay.IsValid(), "backward integration status");
    Expect(decay.backwardEnergy.size() == 3, "backward integration length");
    Expect(Near(decay.backwardEnergy[0], 6.0, 1e-9), "E[0] = 6");
    Expect(Near(decay.backwardEnergy[1], 5.0, 1e-9), "E[1] = 5");
    Expect(Near(decay.backwardEnergy[2], 3.0, 1e-9), "E[2] = 3");
    Expect(Near(decay.levelDb[0], 0.0, 1e-9), "L[0] = 0 dB");
    Expect(Near(decay.levelDb[1], 10.0 * std::log10(5.0 / 6.0), 1e-9), "L[1]");
    Expect(Near(decay.levelDb[2], 10.0 * std::log10(0.5), 1e-9), "L[2]");
    Expect(MonotoneNonIncreasing(decay.levelDb), "short Schroeder curve is monotone");
}

void TestLeadingSilenceDoesNotChangeIntegral() {
    auto echogram = MakeEchogram(0.008f, 1000.0f);
    FillBand(echogram.left, 0, {0.0, 0.0, 0.0, 1.0, 2.0, 3.0});
    const auto metrics = acoustics::RoomMetrics::Compute(echogram);
    const auto& energy = metrics.left.bands[0].decay.backwardEnergy;
    Expect(energy.size() == 3, "leading silence is removed before integration");
    Expect(Near(energy[0], 6.0, 1e-6) && Near(energy[1], 5.0, 1e-6) && Near(energy[2], 3.0, 1e-6),
           "integral after onset matches the active bins");
}

void TestSchroederMonotoneOnExponential() {
    constexpr double kRate = 4000.0;
    const auto energy = ExponentialEnergy(2000, kRate, 0.50);
    auto echogram = MakeEchogram(static_cast<float>(energy.size() / kRate), static_cast<float>(kRate));
    FillBand(echogram.left, 2, energy);
    const auto metrics = acoustics::RoomMetrics::Compute(echogram);
    const auto& decay = metrics.left.bands[2].decay;
    Expect(decay.IsValid(), "exponential Schroeder curve is valid");
    Expect(MonotoneNonIncreasing(decay.backwardEnergy), "backward energy does not grow");
    Expect(MonotoneNonIncreasing(decay.levelDb), "Schroeder level does not grow");
}

void TestRt60AndEdt() {
    constexpr double kRate = 8000.0;
    constexpr double kRt60 = 0.50;
    const auto energy = ExponentialEnergy(static_cast<std::size_t>(kRate * 1.2), kRate, kRt60);
    auto echogram = MakeEchogram(1.2f, static_cast<float>(kRate));
    FillBand(echogram.left, 4, energy);
    const auto metrics = acoustics::RoomMetrics::Compute(echogram);
    const auto& band = metrics.left.bands[4];

    std::cout << "  RT60=" << band.rt60FromT30.value
              << " status=" << acoustics::MetricStatusName(band.rt60FromT30.status)
              << " EDT=" << band.edt.value
              << " status=" << acoustics::MetricStatusName(band.edt.status) << "\n";

    Expect(band.rt60FromT30.IsValid(), "RT60 regression is valid");
    Expect(Near(band.rt60FromT30.value, kRt60, 0.025), "RT60 is within 25 ms of 0.50 s");
    Expect(band.edt.IsValid(), "EDT regression is valid");
    Expect(band.rt60FromT20.IsValid(), "T20 regression is valid");
    Expect(Near(band.rt60FromT20.value, kRt60, 0.025), "T20 matches the same exponential");
    Expect(Near(band.edt.value, kRt60, 0.025), "EDT matches the same exponential");
    Expect(metrics.left.bands[0].rt60FromT30.status == acoustics::MetricStatus::NoEnergy,
           "unused octave band stays invalid");
}

void TestClarityAndDefinition() {
    auto echogram = MakeEchogram(0.15f, 1000.0f);
    std::vector<double> energy(150, 1.0);
    FillBand(echogram.left, 1, energy);
    const auto metrics = acoustics::RoomMetrics::Compute(echogram);
    const auto& band = metrics.left.bands[1];

    Expect(band.c50Db.IsValid() && Near(band.c50Db.value, 10.0 * std::log10(50.0 / 100.0), 1e-4),
           "C50 = 10 log10(50/100)");
    Expect(band.c80Db.IsValid() && Near(band.c80Db.value, 10.0 * std::log10(80.0 / 70.0), 1e-4),
           "C80 = 10 log10(80/70)");
    Expect(band.d50Percent.IsValid() && Near(band.d50Percent.value, 100.0 * 50.0 / 150.0, 1e-4),
           "D50 = 33.333 percent");
}

void TestFractionalClarityWindow() {
    auto echogram = MakeEchogram(0.08f, 250.0f);
    FillBand(echogram.left, 0, std::vector<double>(20, 1.0));
    const auto metrics = acoustics::RoomMetrics::Compute(echogram);
    const auto& band = metrics.left.bands[0];
    Expect(band.c50Db.IsValid() && Near(band.c50Db.value, 10.0 * std::log10(12.5 / 7.5), 1e-4),
           "C50 splits the bin that contains 50 ms");
    Expect(band.d50Percent.IsValid() && Near(band.d50Percent.value, 62.5, 1e-4),
           "D50 uses the same fractional split");
}

void TestDirectSoundOnly() {
    auto echogram = MakeEchogram(0.02f, 1000.0f);
    FillBand(echogram.left, 0, {4.0});
    const auto metrics = acoustics::RoomMetrics::Compute(echogram);
    const auto& band = metrics.left.bands[0];
    Expect(band.d50Percent.IsValid() && Near(band.d50Percent.value, 100.0, 1e-6),
           "D50 is 100 percent when all energy is early");
    Expect(band.c50Db.status == acoustics::MetricStatus::UndefinedRatio,
           "C50 is invalid when late energy is zero");
    Expect(!band.rt60FromT30.IsValid() && band.rt60FromT30.value == 0.0,
           "a single impulse does not invent an RT60");
}

void TestEmptyAndInvalidInput() {
    auto empty = MakeEchogram(0.05f, 1000.0f);
    empty.left.clear();
    empty.right.clear();
    const auto emptyMetrics = acoustics::RoomMetrics::Compute(empty);
    Expect(emptyMetrics.left.bands[0].decay.status == acoustics::MetricStatus::EmptyInput,
           "empty left channel");
    Expect(emptyMetrics.right.bands[6].rt60FromT30.status == acoustics::MetricStatus::EmptyInput,
           "empty right channel");

    auto invalidRate = MakeEchogram(0.01f, 1000.0f);
    invalidRate.left[0][0] = 1.0f;
    invalidRate.sampleRate = 0.0f;
    const auto rateMetrics = acoustics::RoomMetrics::Compute(invalidRate);
    Expect(rateMetrics.left.bands[0].decay.status == acoustics::MetricStatus::InvalidSampleRate,
           "zero sample rate is rejected");

    auto nonFinite = MakeEchogram(0.01f, 1000.0f);
    nonFinite.left[0][3] = std::numeric_limits<float>::quiet_NaN();
    const auto finiteMetrics = acoustics::RoomMetrics::Compute(nonFinite);
    Expect(finiteMetrics.left.bands[3].decay.status == acoustics::MetricStatus::NonFiniteInput,
           "NaN energy is rejected");
    Expect(finiteMetrics.left.bands[0].decay.status == acoustics::MetricStatus::NoEnergy,
           "a finite empty band stays NoEnergy beside NaN");

    auto negative = MakeEchogram(0.01f, 1000.0f);
    negative.left[0][2] = -5.0f;
    const auto negativeMetrics = acoustics::RoomMetrics::Compute(negative);
    Expect(negativeMetrics.left.bands[2].decay.status == acoustics::MetricStatus::NoEnergy,
           "negative energy is treated as no energy");
}

void TestInsufficientDynamicRange() {
    auto echogram = MakeEchogram(0.10f, 4000.0f);
    const double tau = 0.10 / 2.763102111592854;
    std::vector<double> energy(400);
    for (std::size_t n = 0; n < energy.size(); ++n) {
        energy[n] = std::exp(-static_cast<double>(n) / (4000.0 * tau));
    }
    FillBand(echogram.right, 5, energy);
    const auto metrics = acoustics::RoomMetrics::Compute(echogram);
    const auto& band = metrics.right.bands[5];
    Expect(band.rt60FromT30.status == acoustics::MetricStatus::InsufficientDynamicRange,
           "12 dB decay cannot support a T30 estimate");
    Expect(!band.rt60FromT30.IsValid() && band.rt60FromT30.value == 0.0,
           "insufficient range does not return a fake RT60");
    Expect(metrics.left.bands[5].decay.status == acoustics::MetricStatus::NoEnergy,
           "the untouched ear stays empty");
}

void TestNoiseFloor() {
    constexpr double kRate = 4000.0;
    constexpr double kRt60 = 0.40;
    auto energy = ExponentialEnergy(static_cast<std::size_t>(kRate), kRate, kRt60);
    for (double& sample : energy) sample += 1.0e-6;

    auto recoverable = MakeEchogram(1.0f, static_cast<float>(kRate));
    FillBand(recoverable.left, 0, energy);
    const auto recovered = acoustics::RoomMetrics::Compute(recoverable);
    const auto& recoveredBand = recovered.left.bands[0];
    std::cout << "  noise-floor RT60=" << recoveredBand.rt60FromT30.value
              << " status=" << acoustics::MetricStatusName(recoveredBand.rt60FromT30.status) << "\n";
    Expect(recoveredBand.rt60FromT30.IsValid() && Near(recoveredBand.rt60FromT30.value, kRt60, 0.04),
           "flat floor 60 dB down is removed before T30");

    auto limitedEnergy = ExponentialEnergy(static_cast<std::size_t>(kRate), kRate, kRt60);
    for (double& sample : limitedEnergy) sample += 1.0e-2;
    auto limited = MakeEchogram(1.0f, static_cast<float>(kRate));
    FillBand(limited.right, 0, limitedEnergy);
    const auto limitedMetrics = acoustics::RoomMetrics::Compute(limited);
    Expect(limitedMetrics.right.bands[0].rt60FromT30.status == acoustics::MetricStatus::InsufficientDynamicRange,
           "floor only 20 dB down does not support T30");
}

void TestChannelIndependence() {
    constexpr double kRate = 8000.0;
    auto echogram = MakeEchogram(1.2f, static_cast<float>(kRate));
    FillBand(echogram.left, 1, ExponentialEnergy(static_cast<std::size_t>(kRate * 1.2), kRate, 0.40));
    FillBand(echogram.right, 1, ExponentialEnergy(static_cast<std::size_t>(kRate * 1.2), kRate, 0.80));
    const auto metrics = acoustics::RoomMetrics::Compute(echogram);

    std::cout << "  L=" << metrics.left.bands[1].rt60FromT30.value
              << " R=" << metrics.right.bands[1].rt60FromT30.value << "\n";
    Expect(metrics.left.bands[1].rt60FromT30.IsValid() && metrics.right.bands[1].rt60FromT30.IsValid(),
           "both ears produce an RT60");
    Expect(Near(metrics.left.bands[1].rt60FromT30.value, 0.40, 0.03), "left RT60 is 0.40 s");
    Expect(Near(metrics.right.bands[1].rt60FromT30.value, 0.80, 0.03), "right RT60 is 0.80 s");
    Expect(std::abs(metrics.left.bands[1].rt60FromT30.value - metrics.right.bands[1].rt60FromT30.value) > 0.25,
           "left and right RT60 stay independent");
    Expect(Near(metrics.left.bands[1].edt.value, 0.40, 0.03) && Near(metrics.right.bands[1].edt.value, 0.80, 0.03),
           "EDT is also independent per ear");
}

void TestBroadbandMetrics() {
    constexpr double kRate = 4000.0;
    auto echogram = MakeEchogram(1.2f, static_cast<float>(kRate));
    const auto energy1 = ExponentialEnergy(static_cast<std::size_t>(kRate * 1.2), kRate, 0.50);
    const auto energy2 = ExponentialEnergy(static_cast<std::size_t>(kRate * 1.2), kRate, 0.50);
    FillBand(echogram.left, 2, energy1);
    FillBand(echogram.left, 3, energy2);

    const auto metrics = acoustics::RoomMetrics::Compute(echogram);
    const auto& overall = metrics.left.overall;
    Expect(overall.decay.IsValid(), "broadband decay is valid");
    Expect(MonotoneNonIncreasing(overall.decay.levelDb), "broadband Schroeder curve is monotone");
    Expect(overall.rt60FromT30.IsValid(), "broadband RT60 is valid");
    Expect(Near(overall.rt60FromT30.value, 0.50, 0.03), "broadband RT60 matches exponential");
    Expect(overall.rt60FromT20.IsValid(), "broadband T20 is valid");
    Expect(Near(overall.rt60FromT20.value, 0.50, 0.03), "broadband T20 matches exponential");
    Expect(metrics.left.MidFrequencyRt60().IsValid(), "mid-frequency RT60 is valid");
    Expect(Near(metrics.left.MidFrequencyRt60().value, 0.50, 0.03), "mid-frequency RT60 matches exponential");
}

void TestBandpassStability() {
    const float centers[] = {125.0f, 250.0f, 500.0f, 1000.0f, 2000.0f, 4000.0f, 8000.0f};
    for (float center : centers) {
        acoustics::ButterworthBandpass filter(48000.0f, center);
        Expect(filter.IsValid(), std::string("bandpass valid at ") + std::to_string(static_cast<int>(center)) + " Hz");
        if (!filter.IsValid()) continue;

        double early = 0.0;
        double late = 0.0;
        bool finite = true;
        constexpr int kCount = 8192;
        for (int n = 0; n < kCount; ++n) {
            const float y = filter.Process(n == 0 ? 1.0f : 0.0f);
            if (!std::isfinite(y)) finite = false;
            const double energy = static_cast<double>(y) * static_cast<double>(y);
            if (n < kCount / 2) early += energy;
            else late += energy;
        }
        Expect(finite, std::string("bandpass impulse is finite at ") + std::to_string(static_cast<int>(center)) + " Hz");
        Expect(early > late && early > 0.0,
               std::string("bandpass impulse decays at ") + std::to_string(static_cast<int>(center)) + " Hz");
    }

    acoustics::ButterworthBandpass invalid(16000.0f, 8000.0f);
    Expect(!invalid.IsValid(), "8 kHz band is rejected when it reaches Nyquist");
    Expect(invalid.Process(1.0f) == 0.0f, "invalid bandpass emits silence");
}

void TestImpulseResponseContract() {
    auto echogram = MakeEchogram(128.0f / 48000.0f, 48000.0f);
    for (OctaveBands& bin : echogram.left) bin = OctaveBands(0.02f);
    for (OctaveBands& bin : echogram.right) bin = OctaveBands(0.02f);

    acoustics::AudioEngine::SynthOptions options;
    const auto first = acoustics::AudioEngine::SynthesizeImpulseResponse(echogram, options);
    const auto second = acoustics::AudioEngine::SynthesizeImpulseResponse(echogram, options);
    Expect(first.ok && second.ok, "synthesis accepts a synthetic echogram");
    Expect(first.audio.left.size() == echogram.left.size() && first.audio.right.size() == echogram.right.size(),
           "impulse response keeps the echogram length");
    Expect(first.audio.IsFinite(), "impulse response is finite");

    float peak = 0.0f;
    bool differentEars = false;
    bool identicalRepeat = first.audio.left.size() == second.audio.left.size();
    for (std::size_t n = 0; n < first.audio.left.size(); ++n) {
        peak = std::max(peak, std::abs(first.audio.left[n]));
        peak = std::max(peak, std::abs(first.audio.right[n]));
        if (first.audio.left[n] != first.audio.right[n]) differentEars = true;
        if (first.audio.left[n] != second.audio.left[n] || first.audio.right[n] != second.audio.right[n]) {
            identicalRepeat = false;
        }
    }
    Expect(identicalRepeat, "the same seeds reproduce the same impulse response");
    Expect(differentEars, "left and right noise sequences differ");
    Expect(peak > 0.01f && peak <= 0.991f, "peak normalization stays in range");

    options.leftSeed = 7u;
    options.rightSeed = 7u;
    const auto sameSeed = acoustics::AudioEngine::SynthesizeImpulseResponse(echogram, options);
    bool earsMatch = sameSeed.ok && sameSeed.audio.left.size() == sameSeed.audio.right.size();
    for (std::size_t n = 0; earsMatch && n < sameSeed.audio.left.size(); ++n) {
        earsMatch = sameSeed.audio.left[n] == sameSeed.audio.right[n];
    }
    Expect(earsMatch, "equal seeds and equal echograms give equal ears");

    auto empty = MakeEchogram(0.01f, 48000.0f);
    empty.left.clear();
    empty.right.clear();
    Expect(!acoustics::AudioEngine::SynthesizeImpulseResponse(empty).ok, "empty echogram is rejected");
}

std::vector<float> NaiveConvolve(const std::vector<float>& input, const std::vector<float>& impulse) {
    std::vector<float> output(input.size() + impulse.size() - 1, 0.0f);
    for (std::size_t i = 0; i < input.size(); ++i) {
        for (std::size_t j = 0; j < impulse.size(); ++j) {
            output[i + j] += input[i] * impulse[j];
        }
    }
    return output;
}

bool CloseSequence(const std::vector<float>& actual, const std::vector<float>& expected, float tolerance) {
    if (actual.size() != expected.size()) return false;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!Near(actual[i], expected[i], tolerance)) return false;
    }
    return true;
}

void TestConvolution() {
    acoustics::MonoAudio dry;
    dry.sampleRate = 48000.0f;
    dry.samples = {1.0f, 2.0f, 3.0f};

    acoustics::StereoAudio impulse;
    impulse.sampleRate = 48000.0f;
    impulse.left = {1.0f, 0.0f, -1.0f};
    impulse.right = {0.5f};

    const auto wet = acoustics::AudioEngine::Convolve(dry, impulse, 8);
    Expect(wet.ok, "convolution succeeds");
    Expect(wet.audio.left.size() == 5 && wet.audio.right.size() == 5, "output length is Nx + Nh - 1, padded");
    Expect(CloseSequence(wet.audio.left, {1.0f, 2.0f, 2.0f, -2.0f, -3.0f}, 1e-4f),
           "known short left convolution");
    Expect(Near(wet.audio.right[0], 0.5, 1e-4) && Near(wet.audio.right[1], 1.0, 1e-4)
               && Near(wet.audio.right[2], 1.5, 1e-4) && Near(wet.audio.right[3], 0.0, 1e-4),
           "known short right convolution");

    acoustics::MonoAudio click;
    click.sampleRate = 48000.0f;
    click.samples = {1.0f};
    acoustics::StereoAudio longImpulse = impulse;
    longImpulse.left = {0.25f, -0.5f, 0.125f, 0.0625f, -0.03125f, 0.015625f};
    longImpulse.right = longImpulse.left;
    const auto echoed = acoustics::AudioEngine::Convolve(click, longImpulse, 4);
    Expect(echoed.ok && CloseSequence(echoed.audio.left, longImpulse.left, 1e-5f),
           "unit impulse reproduces the impulse response");
    Expect(echoed.audio.left.size() == longImpulse.left.size(), "unit impulse keeps the impulse length");

    std::vector<float> input(37);
    std::vector<float> kernel(23);
    for (std::size_t i = 0; i < input.size(); ++i) input[i] = std::sin(0.37f * static_cast<float>(i));
    for (std::size_t i = 0; i < kernel.size(); ++i) kernel[i] = std::cos(0.21f * static_cast<float>(i)) * 0.1f;
    const auto reference = NaiveConvolve(input, kernel);
    acoustics::MonoAudio signal;
    signal.sampleRate = 16000.0f;
    signal.samples = input;
    acoustics::StereoAudio kernelAudio;
    kernelAudio.sampleRate = 16000.0f;
    kernelAudio.left = kernel;
    kernelAudio.right = kernel;
    const auto fast = acoustics::AudioEngine::Convolve(signal, kernelAudio, 8);
    Expect(fast.ok && fast.audio.left.size() == input.size() + kernel.size() - 1,
           "overlap-add output length");
    Expect(CloseSequence(fast.audio.left, reference, 1e-4f) && CloseSequence(fast.audio.right, reference, 1e-4f),
           "overlap-add matches direct convolution");

    // A ringtone-like burst, then the same burst again 2.0 s later. The delay
    // is not a multiple of the 2048-sample overlap-add block.
    std::vector<float> tone(180);
    for (std::size_t n = 0; n < tone.size(); ++n) {
        tone[n] = static_cast<float>((n * 17) % 11) / 11.0f - 0.5f;
    }
    constexpr std::size_t kDelay = 96000;
    std::vector<float> oneImpulse(kDelay + 1, 0.0f);
    oneImpulse[100] = 0.4f;
    std::vector<float> twoImpulses = oneImpulse;
    twoImpulses[kDelay] = -0.25f;
    const auto oneReference = NaiveConvolve(tone, oneImpulse);
    const auto twoReference = NaiveConvolve(tone, twoImpulses);

    acoustics::MonoAudio toneAudio;
    toneAudio.sampleRate = 48000.0f;
    toneAudio.samples = tone;
    acoustics::StereoAudio oneAudio;
    oneAudio.sampleRate = 48000.0f;
    oneAudio.left = oneImpulse;
    oneAudio.right = oneImpulse;
    acoustics::StereoAudio twoAudio;
    twoAudio.sampleRate = 48000.0f;
    twoAudio.left = twoImpulses;
    twoAudio.right = twoImpulses;

    const auto oneFast = acoustics::AudioEngine::Convolve(toneAudio, oneAudio, 2048);
    const auto twoFast = acoustics::AudioEngine::Convolve(toneAudio, twoAudio, 2048);
    const auto twoSmallBlocks = acoustics::AudioEngine::Convolve(toneAudio, twoAudio, 8);
    const std::size_t oneLength = tone.size() + oneImpulse.size() - 1;
    const std::size_t twoLength = tone.size() + twoImpulses.size() - 1;
    Expect(oneFast.ok && oneFast.audio.left.size() == oneLength &&
               CloseSequence(oneFast.audio.left, oneReference, 1e-4f) &&
               CloseSequence(oneFast.audio.right, oneReference, 1e-4f),
           "one delayed impulse is only a scaled shifted copy");
    Expect(twoFast.ok && twoFast.audio.left.size() == twoLength &&
               CloseSequence(twoFast.audio.left, twoReference, 1e-4f) &&
               CloseSequence(twoFast.audio.right, twoReference, 1e-4f),
           "two impulses 2 s apart match direct convolution");
    Expect(twoSmallBlocks.ok && CloseSequence(twoSmallBlocks.audio.left, twoReference, 1e-4f),
           "two impulses match direct convolution on 8-sample blocks");

    bool secondCopyOnlyAtDelay = true;
    for (std::size_t n = 100 + tone.size(); n < kDelay && secondCopyOnlyAtDelay; ++n) {
        secondCopyOnlyAtDelay = std::abs(twoFast.audio.left[n]) <= 1e-4f;
    }
    Expect(secondCopyOnlyAtDelay, "no extra copy appears between the two impulses");

    dry.samples = {1.0f, std::numeric_limits<float>::quiet_NaN()};
    Expect(!acoustics::AudioEngine::Convolve(dry, impulse).ok, "non-finite audio is rejected");
}

void TestResampleAndRateConversion() {
    acoustics::MonoAudio input;
    input.sampleRate = 1000.0f;
    input.samples = {0.0f, 2.0f, 4.0f, 6.0f};
    acoustics::MonoAudio output;
    const auto status = acoustics::AudioEngine::ResampleLinear(input, 2000.0f, output);
    const std::vector<float> expected = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 6.0f};
    Expect(status.ok && CloseSequence(output.samples, expected, 1e-5f), "linear resample doubles the rate");

    acoustics::StereoAudio impulse;
    impulse.sampleRate = 48000.0f;
    impulse.left = {1.0f};
    impulse.right = {1.0f};
    acoustics::MonoAudio dry;
    dry.sampleRate = 24000.0f;
    dry.samples = {0.0f, 1.0f, 0.0f, -1.0f};
    const auto wet = acoustics::AudioEngine::Convolve(dry, impulse, 16);
    const std::vector<float> resampled = {0.0f, 0.5f, 1.0f, 0.5f, 0.0f, -0.5f, -1.0f, -1.0f};
    Expect(wet.ok && CloseSequence(wet.audio.left, resampled, 1e-4f),
           "convolution resamples the dry signal onto the impulse rate");
}

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

void TestWavRoundTrip(const std::filesystem::path& directory) {
    acoustics::StereoAudio audio;
    audio.sampleRate = 48000.0f;
    audio.left = {0.25f, -0.5f, 0.0f, 0.125f};
    audio.right = {-0.25f, 0.5f, 0.75f, -0.125f};
    const auto path = directory / "roundtrip.wav";
    const auto written = acoustics::AudioEngine::WriteStereoWav(path, audio);
    const auto loaded = acoustics::AudioEngine::ReadWav(path);
    Expect(written.ok && loaded.ok, "stereo WAV writes and reads back");
    Expect(loaded.channels == 2 && loaded.bitsPerSample == 16 && Near(loaded.sampleRate, 48000.0, 0.1),
           "WAV header is stereo PCM16 at 48 kHz");
    Expect(loaded.interleaved.size() == 8, "WAV frame count matches");
    bool close = loaded.interleaved.size() == 8;
    const float expected[8] = {0.25f, -0.25f, -0.5f, 0.5f, 0.0f, 0.75f, 0.125f, -0.125f};
    for (std::size_t i = 0; close && i < 8; ++i) {
        close = Near(loaded.interleaved[i], expected[i], 1.0 / 32767.0);
    }
    Expect(close, "PCM16 samples survive the round trip");

    acoustics::MonoAudio mono;
    Expect(!acoustics::AudioEngine::ReadMonoWav(path, mono).ok, "stereo file is rejected as a dry mono source");

    acoustics::StereoAudio hot;
    hot.sampleRate = 44100.0f;
    hot.left = {2.0f, -2.0f};
    hot.right = {2.0f, -2.0f};
    const auto hotPath = directory / "limited.wav";
    Expect(acoustics::AudioEngine::WriteStereoWav(hotPath, hot).ok, "hot signal is still written");
    const auto limited = acoustics::AudioEngine::ReadWav(hotPath);
    Expect(limited.ok && limited.interleaved.size() == 4, "limited WAV has two stereo frames");
    bool limitedOk = limited.interleaved.size() == 4;
    for (float sample : limited.interleaved) {
        limitedOk = limitedOk && std::abs(sample) <= 0.991f && std::abs(std::abs(sample) - 0.99f) < 0.002f;
    }
    Expect(limitedOk, "both channels share the attenuation that avoids clipping");

    const auto monoPath = directory / "dry.wav";
    {
        std::ofstream out(monoPath, std::ios::binary | std::ios::trunc);
        const float samples[4] = {0.5f, -0.25f, 1.0f, 0.0f};
        out.write("RIFF", 4);
        WriteU32(out, 36u + sizeof(samples));
        out.write("WAVE", 4);
        out.write("fmt ", 4);
        WriteU32(out, 16u);
        WriteU16(out, 3u);
        WriteU16(out, 1u);
        WriteU32(out, 16000u);
        WriteU32(out, 16000u * 4u);
        WriteU16(out, 4u);
        WriteU16(out, 32u);
        out.write("data", 4);
        WriteU32(out, sizeof(samples));
        out.write(reinterpret_cast<const char*>(samples), sizeof(samples));
    }
    acoustics::MonoAudio dry;
    const auto monoStatus = acoustics::AudioEngine::ReadMonoWav(monoPath, dry);
    Expect(monoStatus.ok && dry.samples.size() == 4 && Near(dry.sampleRate, 16000.0, 0.1),
           "mono float32 WAV loads");
    Expect(dry.samples.size() == 4 && Near(dry.samples[0], 0.5, 1e-6) && Near(dry.samples[1], -0.25, 1e-6)
               && Near(dry.samples[2], 1.0, 1e-6),
           "mono float samples are exact");

    const auto missing = acoustics::AudioEngine::PlayWavFile(directory / "missing-file.wav");
    Expect(missing.status == acoustics::PlaybackStatus::Failed, "missing WAV does not pretend to play");
}

void TestLateSpikeAndFewRays() {
    auto echogram = MakeEchogram(2.0f, 48000.0f);
    echogram.left[10] = OctaveBands(1.0f);
    echogram.right[10] = OctaveBands(1.0f);
    constexpr std::size_t kLate = 48000 * 2 - 10;
    echogram.left[kLate] = OctaveBands(0.01f);
    echogram.right[kLate] = OctaveBands(0.01f);

    acoustics::AudioEngine::SynthOptions raw;
    raw.peakNormalize = false;
    const auto normalized = acoustics::AudioEngine::SynthesizeImpulseResponse(echogram);
    const auto unscaled = acoustics::AudioEngine::SynthesizeImpulseResponse(echogram, raw);
    const auto peakNear = [](const std::vector<float>& samples, std::size_t begin, std::size_t end) {
        float peak = 0.0f;
        for (std::size_t n = begin; n < end && n < samples.size(); ++n) {
            peak = std::max(peak, std::abs(samples[n]));
        }
        return peak;
    };
    const float normEarly = peakNear(normalized.audio.left, 0, 4000);
    const float normLate = peakNear(normalized.audio.left, kLate, kLate + 4000);
    const float rawEarly = peakNear(unscaled.audio.left, 0, 4000);
    const float rawLate = peakNear(unscaled.audio.left, kLate, kLate + 4000);
    const double normRatio = normLate / normEarly;
    const double rawRatio = rawLate / rawEarly;
    std::cout << "  late/early peak ratio normalized=" << normRatio << " raw=" << rawRatio
              << "  raw late peak=" << rawLate << " normalized late peak=" << normLate << "\n";
    Expect(normalized.ok && unscaled.ok && rawEarly > 0.0f && rawLate > 0.0f,
           "sparse echogram synthesizes an early and a late burst");
    Expect(std::abs(normRatio - rawRatio) / rawRatio < 0.05,
           "peak normalization keeps the late-to-early ratio");
    Expect(normLate > rawLate, "peak normalization raises the whole response, including the late burst");

    const auto& materials = AcousticMaterialLibrary::Instance();
    const auto wall = materials.GetMaterialId("Uniform02");
    AcousticSceneGeometry room;
    room.AddShoeboxRoom({10.0f, 3.0f, 8.0f}, wall, wall, wall, {0.0f, 1.5f, 0.0f});
    BVH bvh;
    bvh.Build(room);
    const acoustics::Simulator simulator(bvh, materials);
    acoustics::Receiver listener;
    listener.headCenter = {0.0f, 1.5f, 0.5f};
    listener.headShadow = false;
    acoustics::Source source;
    source.position = listener.headCenter + 2.0f * listener.forward;
    source.power = OctaveBands(3.0f);

    const auto reportRays = [&](std::uint32_t rays, std::uint32_t seed) {
        acoustics::Settings settings;
        settings.raysPerSource = rays;
        settings.maxBounces = 50;
        settings.maxDuration = 2.0f;
        settings.randomSeed = seed;
        settings.airAbsorption = OctaveBands::Zero();
        acoustics::Echogram echo(settings.maxDuration);
        const auto stats = simulator.Run({source}, listener, echo, settings);
        struct Cluster { std::size_t begin = 0; double energy = 0.0; };
        std::vector<Cluster> clusters;
        for (std::size_t n = 0; n < echo.left.size(); ++n) {
            const double energy = static_cast<double>(echo.left[n].Average() + echo.right[n].Average());
            if (!(energy > 0.0)) continue;
            if (!clusters.empty() && n <= clusters.back().begin + 64) clusters.back().energy += energy;
            else clusters.push_back({n, energy});
        }
        std::cout << "  rays=" << rays << " seed=" << seed
                  << " earHits=" << stats.earHits
                  << " escaped=" << stats.escaped
                  << " clusters=" << clusters.size() << "\n";
        for (const Cluster& cluster : clusters) {
            std::cout << "    t=" << static_cast<double>(cluster.begin) / echo.sampleRate
                      << " s energy=" << cluster.energy << "\n";
        }
        const bool finalBinSilent = echo.left.back().Average() == 0.0f && echo.right.back().Average() == 0.0f;
        std::cout << "    final bin silent=" << (finalBinSilent ? "yes" : "no") << "\n";
        return clusters.size();
    };
    std::size_t runsWithTwoArrivals = 0;
    std::size_t runsWithAnyArrival = 0;
    std::size_t runsWritingFinalBin = 0;
    for (std::uint32_t rays = 1; rays <= 2; ++rays) {
        for (std::uint32_t seed = 0; seed < 64; ++seed) {
            acoustics::Settings settings;
            settings.raysPerSource = rays;
            settings.maxBounces = 50;
            settings.maxDuration = 2.0f;
            settings.randomSeed = seed;
            settings.airAbsorption = OctaveBands::Zero();
            acoustics::Echogram echo(settings.maxDuration);
            const auto stats = simulator.Run({source}, listener, echo, settings);
            std::size_t clusters = 0;
            std::size_t previous = 0;
            bool any = false;
            for (std::size_t n = 0; n < echo.left.size(); ++n) {
                const double energy = static_cast<double>(echo.left[n].Average() + echo.right[n].Average());
                if (!(energy > 0.0)) continue;
                if (!any || n > previous + 64) ++clusters;
                previous = n;
                any = true;
            }
            if (stats.earHits > 0) ++runsWithAnyArrival;
            if (stats.earHits > 0 && !any) {
                std::cout << "    energy scan missed ear hits rays=" << rays << " seed=" << seed
                          << " earHits=" << stats.earHits << "\n";
            }
            if (clusters >= 2) {
                ++runsWithTwoArrivals;
                std::cout << "    multiple arrivals rays=" << rays << " seed=" << seed
                          << " clusters=" << clusters << " earHits=" << stats.earHits << "\n";
            }
            if (echo.left.back().Average() > 0.0f || echo.right.back().Average() > 0.0f) ++runsWritingFinalBin;
        }
    }
    std::cout << "  128 runs of 1 or 2 rays: arrivals=" << runsWithAnyArrival
              << " two-or-more clusters=" << runsWithTwoArrivals
              << " final-bin writes=" << runsWritingFinalBin << "\n";
    reportRays(1, 42);
    reportRays(2, 42);
    Expect(runsWritingFinalBin == 0, "these 1-2 ray runs do not write an impulse into the final bin");

    acoustics::Receiver outside;
    outside.headCenter = {0.0f, 80.0f, 0.0f};
    outside.headShadow = false;
    acoustics::Settings missed;
    missed.raysPerSource = 2;
    missed.maxBounces = 50;
    missed.maxDuration = 2.0f;
    missed.randomSeed = 42;
    acoustics::Echogram silent(missed.maxDuration);
    const auto missedStats = simulator.Run({source}, outside, silent, missed);
    bool anyEnergy = false;
    for (const OctaveBands& bin : silent.left) anyEnergy = anyEnergy || bin.Average() > 0.0f;
    for (const OctaveBands& bin : silent.right) anyEnergy = anyEnergy || bin.Average() > 0.0f;
    Expect(missedStats.earHits == 0 && !anyEnergy,
           "a ray that never meets an ear leaves the echogram empty at 2 s");
}

void TestSyntheticDemo(const std::filesystem::path& directory) {
    const auto path = directory / "synthetic-demo.wav";
    const auto demo = acoustics::AudioEngine::RunSyntheticDemo(path, false);
    std::cout << "  demo: " << demo.message << "\n";
    Expect(demo.ok, "synthetic demo completes");
    Expect(std::filesystem::is_regular_file(path), "demo writes a WAV file");
    Expect(Near(demo.sampleRate, 48000.0, 0.1) && demo.frameCount > 0, "demo reports a non-empty 48 kHz result");

    const auto loaded = acoustics::AudioEngine::ReadWav(path);
    Expect(loaded.ok && loaded.channels == 2 && loaded.interleaved.size() == demo.frameCount * 2,
           "demo WAV is stereo and has the reported length");
    Expect(demo.metrics.left.bands[0].decay.IsValid() && demo.metrics.right.bands[0].decay.IsValid(),
           "demo metrics read both ears of the synthetic echogram");
    Expect(demo.playback.status == acoustics::PlaybackStatus::Unavailable,
           "demo does not play unless playback is requested");
}

} // namespace

int main() {
    std::cout << "Acoustics steps 5 and 6\n";
    TestBackwardIntegration();
    TestLeadingSilenceDoesNotChangeIntegral();
    TestSchroederMonotoneOnExponential();
    TestRt60AndEdt();
    TestClarityAndDefinition();
    TestFractionalClarityWindow();
    TestDirectSoundOnly();
    TestEmptyAndInvalidInput();
    TestInsufficientDynamicRange();
    TestNoiseFloor();
    TestChannelIndependence();
    TestBroadbandMetrics();
    TestBandpassStability();
    TestImpulseResponseContract();
    TestConvolution();
    TestLateSpikeAndFewRays();
    TestResampleAndRateConversion();

    const auto directory = std::filesystem::temp_directory_path() / "ga-ray-tracing-step56";
    std::filesystem::create_directories(directory);
    TestWavRoundTrip(directory);
    TestSyntheticDemo(directory);

    std::error_code error;
    std::filesystem::remove_all(directory, error);

    if (g_failed != 0) {
        std::cout << g_failed << " test(s) failed\n";
        return 1;
    }
    std::cout << "All acoustics step tests passed\n";
    return 0;
}
