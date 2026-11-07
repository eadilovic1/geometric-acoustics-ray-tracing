#pragma once

namespace acoustics {

// Butterworth pojasni filter cetvrtog reda za jedan ISO oktavni opseg.
// Realizacija su dvije kaskadne biquad sekcije (DF2T):
//   highpass 2. reda na fc/sqrt(2), Q = 1/sqrt(2)
//   lowpass  2. reda na fc*sqrt(2), Q = 1/sqrt(2)
// Pojacanje je normirano tako da je |H(fc)| = 1.
class ButterworthBandpass {
public:
    explicit ButterworthBandpass(float sampleRate, float centerFrequencyHz);

    [[nodiscard]] bool IsValid() const noexcept { return valid_; }
    [[nodiscard]] float LowCutoffHz() const noexcept { return lowHz_; }
    [[nodiscard]] float HighCutoffHz() const noexcept { return highHz_; }

    void Reset() noexcept;
    [[nodiscard]] float Process(float sample) noexcept;

private:
    struct Section {
        double b0 = 1.0;
        double b1 = 0.0;
        double b2 = 0.0;
        double a1 = 0.0;
        double a2 = 0.0;
        double z1 = 0.0;
        double z2 = 0.0;

        [[nodiscard]] double Process(double x) noexcept;
    };

    Section highpass_{};
    Section lowpass_{};
    double outputGain_ = 1.0;
    float lowHz_ = 0.0f;
    float highHz_ = 0.0f;
    bool valid_ = false;
};

} // namespace acoustics
