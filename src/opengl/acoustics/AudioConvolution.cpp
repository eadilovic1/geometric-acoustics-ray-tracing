#include "acoustics/AudioEngine.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>

namespace acoustics {

namespace {

constexpr double kPi = 3.14159265358979323846;

std::size_t NextPowerOfTwo(std::size_t value) {
    std::size_t power = 1;
    while (power < value && power < (std::size_t{1} << 20)) power <<= 1;
    return power;
}

std::size_t NormalizeBlockSize(std::size_t requested) {
    std::size_t block = requested < 8 ? 8 : requested;
    if (block > 65536) block = 65536;
    return NextPowerOfTwo(block);
}

void Fft(std::vector<std::complex<double>>& data, bool inverse) {
    const std::size_t n = data.size();
    if (n < 2) return;

    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; (j & bit) != 0; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(data[i], data[j]);
    }

    for (std::size_t length = 2; length <= n; length <<= 1) {
        const double angle = (inverse ? 2.0 : -2.0) * kPi / static_cast<double>(length);
        const std::complex<double> wlen(std::cos(angle), std::sin(angle));
        for (std::size_t start = 0; start < n; start += length) {
            std::complex<double> w(1.0, 0.0);
            const std::size_t half = length / 2;
            for (std::size_t j = 0; j < half; ++j) {
                const std::complex<double> u = data[start + j];
                const std::complex<double> v = data[start + j + half] * w;
                data[start + j] = u + v;
                data[start + j + half] = u - v;
                w *= wlen;
            }
        }
    }

    if (inverse) {
        const double inv = 1.0 / static_cast<double>(n);
        for (std::complex<double>& sample : data) sample *= inv;
    }
}

std::vector<float> ConvolveChannel(
    const std::vector<float>& input,
    const std::vector<float>& impulse,
    std::size_t blockSize
) {
    if (input.empty() || impulse.empty()) return {};

    const std::size_t block = blockSize;
    const std::size_t fftSize = block * 2;
    const std::size_t partitions = (impulse.size() + block - 1) / block;
    const std::size_t inputBlocks = (input.size() + block - 1) / block;
    const std::size_t outputLength = input.size() + impulse.size() - 1;

    std::vector<std::vector<std::complex<double>>> spectra(partitions);
    for (std::size_t part = 0; part < partitions; ++part) {
        std::vector<std::complex<double>> spectrum(fftSize);
        const std::size_t offset = part * block;
        const std::size_t count = std::min(block, impulse.size() - offset);
        for (std::size_t n = 0; n < count; ++n) {
            spectrum[n] = static_cast<double>(impulse[offset + n]);
        }
        Fft(spectrum, false);
        spectra[part] = std::move(spectrum);
    }

    std::vector<double> accumulator(outputLength, 0.0);
    for (std::size_t blockIndex = 0; blockIndex < inputBlocks; ++blockIndex) {
        std::vector<std::complex<double>> spectrum(fftSize);
        const std::size_t offset = blockIndex * block;
        const std::size_t count = std::min(block, input.size() - offset);
        for (std::size_t n = 0; n < count; ++n) {
            spectrum[n] = static_cast<double>(input[offset + n]);
        }
        Fft(spectrum, false);

        for (std::size_t part = 0; part < partitions; ++part) {
            std::vector<std::complex<double>> product(fftSize);
            for (std::size_t bin = 0; bin < fftSize; ++bin) {
                product[bin] = spectrum[bin] * spectra[part][bin];
            }
            Fft(product, true);

            const std::size_t dest = (blockIndex + part) * block;
            const std::size_t write = std::min(fftSize, outputLength - std::min(outputLength, dest));
            for (std::size_t n = 0; n < write; ++n) {
                accumulator[dest + n] += product[n].real();
            }
        }
    }

    std::vector<float> output(outputLength, 0.0f);
    for (std::size_t n = 0; n < outputLength; ++n) {
        output[n] = static_cast<float>(accumulator[n]);
    }
    return output;
}

} // namespace

ConvolveResult AudioEngine::Convolve(
    const MonoAudio& dry,
    const StereoAudio& impulseResponse,
    std::size_t blockSize
) {
    ConvolveResult result;
    if (!(impulseResponse.sampleRate > 0.0f) || !std::isfinite(impulseResponse.sampleRate)) {
        result.message = "Frekvencija impulsnog odziva nije ispravna.";
        return result;
    }
    if (!dry.IsFinite() || !impulseResponse.IsFinite()) {
        result.message = "Ulaz konvolucije nije konacan.";
        return result;
    }
    if (dry.samples.empty() || (impulseResponse.left.empty() && impulseResponse.right.empty())) {
        result.audio.sampleRate = impulseResponse.sampleRate;
        result.ok = true;
        result.message = "Prazan ulaz daje prazan izlaz.";
        return result;
    }

    MonoAudio aligned = dry;
    if (!(dry.sampleRate > 0.0f) || !std::isfinite(dry.sampleRate)) {
        result.message = "Frekvencija suhog signala nije ispravna.";
        return result;
    }
    if (std::abs(dry.sampleRate - impulseResponse.sampleRate) > 0.01f) {
        const WavIoStatus resampled = ResampleLinear(dry, impulseResponse.sampleRate, aligned);
        if (!resampled.ok) {
            result.message = resampled.message;
            return result;
        }
    }

    const std::size_t block = NormalizeBlockSize(blockSize);
    result.audio.sampleRate = impulseResponse.sampleRate;
    result.audio.left = ConvolveChannel(aligned.samples, impulseResponse.left, block);
    result.audio.right = ConvolveChannel(aligned.samples, impulseResponse.right, block);

    const std::size_t frames = std::max(result.audio.left.size(), result.audio.right.size());
    result.audio.left.resize(frames, 0.0f);
    result.audio.right.resize(frames, 0.0f);

    if (!result.audio.IsFinite()) {
        result.audio.left.clear();
        result.audio.right.clear();
        result.message = "Konvolucija nije ostala konacna.";
        return result;
    }

    result.ok = true;
    result.message = "Overlap-add konvolucija je zavrsena.";
    return result;
}

bool MonoAudio::IsFinite() const {
    for (float sample : samples) {
        if (!std::isfinite(sample)) return false;
    }
    return true;
}

} // namespace acoustics
