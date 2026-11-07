#include "acoustics/RoomMetrics.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace acoustics {

namespace {

constexpr double kRt60StartDb = -5.0;
constexpr double kRt60EndDb = -35.0;
constexpr double kEdtStartDb = 0.0;
constexpr double kEdtEndDb = -10.0;
constexpr double kExtrapolationDb = 60.0;

struct ExtractedBand {
    MetricStatus status = MetricStatus::NoEnergy;
    std::vector<double> energy;
};

void FailBand(BandResult& band, MetricStatus status) {
    band.decay.status = status;
    band.rt60FromT30.status = status;
    band.edt.status = status;
    band.c50Db.status = status;
    band.c80Db.status = status;
    band.d50Percent.status = status;
    band.rt60FromT30.value = 0.0;
    band.edt.value = 0.0;
    band.c50Db.value = 0.0;
    band.c80Db.value = 0.0;
    band.d50Percent.value = 0.0;
}

double Mean(const std::vector<double>& values, std::size_t begin, std::size_t end) {
    if (end <= begin) return 0.0;
    double sum = 0.0;
    for (std::size_t i = begin; i < end; ++i) sum += values[i];
    return sum / static_cast<double>(end - begin);
}

double StdDev(const std::vector<double>& values, std::size_t begin, std::size_t end, double mean) {
    if (end <= begin) return 0.0;
    double acc = 0.0;
    for (std::size_t i = begin; i < end; ++i) {
        const double d = values[i] - mean;
        acc += d * d;
    }
    return std::sqrt(acc / static_cast<double>(end - begin));
}

void TrimSilence(std::vector<double>& energy, double onsetLevel) {
    std::size_t first = 0;
    while (first < energy.size() && energy[first] <= onsetLevel) ++first;

    std::size_t last = energy.size();
    while (last > first && energy[last - 1] <= 0.0) --last;

    if (first == 0 && last == energy.size()) return;
    energy.assign(energy.begin() + static_cast<std::ptrdiff_t>(first),
                  energy.begin() + static_cast<std::ptrdiff_t>(last));
}

// Pad u dB izmedju prve i zadnje cetvrtine prozora. Pozitivan broj znaci opadanje.
double WindowDbDrop(const std::vector<double>& energy, std::size_t begin, std::size_t end) {
    if (end < begin + 4) return 0.0;
    const std::size_t quarter = std::max<std::size_t>(1, (end - begin) / 4);
    const double first = Mean(energy, begin, begin + quarter);
    const double last = Mean(energy, end - quarter, end);
    if (!(first > 0.0) || !(last > 0.0)) return 0.0;
    return 10.0 * std::log10(first / last);
}

// Pod se skida samo kada rep vise ne prati nagib pocetka.
// Eksponencijalni rep ima isti nagib kao pocetak, pa se ne dira.
void SubtractFlatNoiseFloor(std::vector<double>& energy) {
    if (energy.size() < 20) return;

    const std::size_t window = std::max<std::size_t>(8, energy.size() / 10);
    const double head = Mean(energy, 0, window);
    const std::size_t tailBegin = energy.size() - window;
    const double tail = Mean(energy, tailBegin, energy.size());
    if (!(tail > 0.0) || !(head > 4.0 * tail)) return;

    const double headDrop = WindowDbDrop(energy, 0, window);
    const double tailDrop = WindowDbDrop(energy, tailBegin, energy.size());
    if (!(headDrop > 6.0) || !(tailDrop < 0.25 * headDrop)) return;

    const double deviation = StdDev(energy, tailBegin, energy.size(), tail);
    if (deviation / tail >= 0.35) return;

    // Energija ispod poda je potopljena u sum. Ne rekonstruise se oduzimanjem
    // konstante sve do kraja zapisa, jer bi to izmislilo dinamicki raspon.
    const double cutoff = tail * 4.0;
    std::size_t last = energy.size();
    while (last > 0 && energy[last - 1] <= cutoff) --last;
    energy.resize(last);

    for (double& sample : energy) {
        sample = std::max(0.0, sample - tail);
    }
}

ExtractedBand ExtractBand(
    const std::vector<OctaveBands>& bins,
    std::size_t band,
    const RoomMetrics::Options& options
) {
    ExtractedBand extracted;
    extracted.energy.reserve(bins.size());

    for (const OctaveBands& bin : bins) {
        const float value = bin[band];
        if (!std::isfinite(value)) {
            extracted.status = MetricStatus::NonFiniteInput;
            extracted.energy.clear();
            return extracted;
        }
        extracted.energy.push_back(static_cast<double>(std::max(0.0f, value)));
    }

    double peak = 0.0;
    for (double sample : extracted.energy) peak = std::max(peak, sample);
    if (!(peak > 0.0)) {
        extracted.status = MetricStatus::NoEnergy;
        extracted.energy.clear();
        return extracted;
    }

    const double onset = peak * std::max(0.0, options.onsetFraction);
    TrimSilence(extracted.energy, onset);
    SubtractFlatNoiseFloor(extracted.energy);

    double cleanedPeak = 0.0;
    for (double sample : extracted.energy) cleanedPeak = std::max(cleanedPeak, sample);
    const double dust = cleanedPeak * std::max(0.0, options.dustFraction);
    if (dust > 0.0) {
        for (double& sample : extracted.energy) {
            if (sample < dust) sample = 0.0;
        }
    }
    TrimSilence(extracted.energy, 0.0);

    double total = 0.0;
    for (double sample : extracted.energy) total += sample;
    if (extracted.energy.empty() || !(total > 0.0)) {
        extracted.status = MetricStatus::NoEnergy;
        extracted.energy.clear();
        return extracted;
    }

    extracted.status = MetricStatus::Ok;
    return extracted;
}

// Raspon koji je stvarno vidljiv prije zavrsnog odsjecanja zapisa.
// Zadnjih nekoliko binova Schroederove krive padaju i kada energija jos nije
// pala za trazenih 35 dB, pa se taj kraj ne racuna kao dokaz dinamike.
double UsableRangeDb(const std::vector<double>& levelDb) {
    if (levelDb.size() < 4) return 0.0;
    const std::size_t margin = std::max<std::size_t>(levelDb.size() / 20, 8);
    if (margin >= levelDb.size()) return 0.0;
    const double level = levelDb[levelDb.size() - margin];
    if (!std::isfinite(level)) return 0.0;
    return std::max(0.0, -level);
}

void SplitEarlyLate(
    const std::vector<double>& energy,
    double sampleRate,
    double windowSec,
    double& early,
    double& late
) {
    early = 0.0;
    late = 0.0;
    const double limitSamples = windowSec * sampleRate;
    for (std::size_t k = 0; k < energy.size(); ++k) {
        const double sample = energy[k];
        const double start = static_cast<double>(k);
        const double end = static_cast<double>(k + 1);
        if (end <= limitSamples) {
            early += sample;
        } else if (start >= limitSamples) {
            late += sample;
        } else {
            const double fraction = std::clamp(limitSamples - start, 0.0, 1.0);
            early += sample * fraction;
            late += sample * (1.0 - fraction);
        }
    }
}

ScalarMetric ClarityDb(double early, double late) {
    ScalarMetric metric;
    if (!(early > 0.0) || !(late > 0.0)) {
        metric.status = MetricStatus::UndefinedRatio;
        return metric;
    }
    const double value = 10.0 * std::log10(early / late);
    if (!std::isfinite(value)) {
        metric.status = MetricStatus::UndefinedRatio;
        return metric;
    }
    metric.status = MetricStatus::Ok;
    metric.value = value;
    return metric;
}

ScalarMetric DefinitionPercent(double early, double total) {
    ScalarMetric metric;
    if (!(total > 0.0)) {
        metric.status = MetricStatus::NoEnergy;
        return metric;
    }
    metric.status = MetricStatus::Ok;
    metric.value = 100.0 * early / total;
    return metric;
}

bool FindCrossing(const std::vector<double>& levelDb, double thresholdDb, std::size_t& index) {
    for (std::size_t i = 0; i < levelDb.size(); ++i) {
        if (std::isfinite(levelDb[i]) && levelDb[i] <= thresholdDb) {
            index = i;
            return true;
        }
    }
    return false;
}

ScalarMetric Regress(
    const std::vector<double>& levelDb,
    double sampleRate,
    double startDb,
    double endDb,
    double requiredRangeDb,
    const RoomMetrics::Options& options
) {
    ScalarMetric metric;
    if (UsableRangeDb(levelDb) + 1e-6 < requiredRangeDb) {
        metric.status = MetricStatus::InsufficientDynamicRange;
        return metric;
    }

    std::size_t iStart = 0;
    std::size_t iEnd = 0;
    if (!FindCrossing(levelDb, startDb, iStart) || !FindCrossing(levelDb, endDb, iEnd) || iEnd < iStart) {
        metric.status = MetricStatus::InsufficientDynamicRange;
        return metric;
    }

    const std::size_t count = iEnd - iStart + 1;
    if (count < std::max<std::size_t>(2, options.minimumRegressionSamples)) {
        metric.status = MetricStatus::InsufficientDynamicRange;
        return metric;
    }

    double sumT = 0.0;
    double sumL = 0.0;
    double sumTT = 0.0;
    double sumLL = 0.0;
    double sumTL = 0.0;
    std::size_t used = 0;
    const double invRate = 1.0 / sampleRate;

    for (std::size_t i = iStart; i <= iEnd; ++i) {
        const double level = levelDb[i];
        if (!std::isfinite(level)) continue;
        const double time = static_cast<double>(i) * invRate;
        sumT += time;
        sumL += level;
        sumTT += time * time;
        sumLL += level * level;
        sumTL += time * level;
        ++used;
    }

    if (used < std::max<std::size_t>(2, options.minimumRegressionSamples)) {
        metric.status = MetricStatus::InsufficientDynamicRange;
        return metric;
    }

    const double n = static_cast<double>(used);
    const double denom = n * sumTT - sumT * sumT;
    if (!(std::abs(denom) > 1e-18)) {
        metric.status = MetricStatus::PoorFit;
        return metric;
    }

    const double slope = (n * sumTL - sumT * sumL) / denom;
    if (!(slope < -1e-9) || !std::isfinite(slope)) {
        metric.status = MetricStatus::PoorFit;
        return metric;
    }

    const double varT = sumTT - sumT * sumT / n;
    const double varL = sumLL - sumL * sumL / n;
    if (!(varT > 1e-18) || !(varL > 1e-12)) {
        metric.status = MetricStatus::PoorFit;
        return metric;
    }
    const double cov = sumTL - sumT * sumL / n;
    const double determination = (cov * cov) / (varT * varL);
    if (!std::isfinite(determination) || determination < options.minimumDetermination) {
        metric.status = MetricStatus::PoorFit;
        return metric;
    }

    const double rt60 = -kExtrapolationDb / slope;
    if (!(rt60 > 0.0) || !std::isfinite(rt60)) {
        metric.status = MetricStatus::PoorFit;
        return metric;
    }

    metric.status = MetricStatus::Ok;
    metric.value = rt60;
    return metric;
}

void AnalyzeExtracted(
    const ExtractedBand& extracted,
    double sampleRate,
    const RoomMetrics::Options& options,
    BandResult& band
) {
    if (extracted.status != MetricStatus::Ok) {
        FailBand(band, extracted.status);
        return;
    }

    const std::vector<double>& energy = extracted.energy;
    band.decay.backwardEnergy.assign(energy.size(), 0.0);
    double accumulated = 0.0;
    for (std::size_t index = energy.size(); index-- > 0; ) {
        accumulated += energy[index];
        band.decay.backwardEnergy[index] = accumulated;
    }

    if (!(band.decay.backwardEnergy.front() > 0.0) || !std::isfinite(accumulated)) {
        FailBand(band, MetricStatus::NoEnergy);
        return;
    }

    band.decay.levelDb.resize(energy.size());
    const double reference = band.decay.backwardEnergy.front();
    for (std::size_t k = 0; k < energy.size(); ++k) {
        const double ratio = band.decay.backwardEnergy[k] / reference;
        if (!(ratio > 0.0) || !std::isfinite(ratio)) {
            FailBand(band, MetricStatus::NonFiniteInput);
            return;
        }
        band.decay.levelDb[k] = 10.0 * std::log10(ratio);
    }
    band.decay.status = MetricStatus::Ok;

    double early50 = 0.0;
    double late50 = 0.0;
    double early80 = 0.0;
    double late80 = 0.0;
    SplitEarlyLate(energy, sampleRate, 0.050, early50, late50);
    SplitEarlyLate(energy, sampleRate, 0.080, early80, late80);

    band.c50Db = ClarityDb(early50, late50);
    band.c80Db = ClarityDb(early80, late80);
    band.d50Percent = DefinitionPercent(early50, early50 + late50);

    band.rt60FromT30 = Regress(
        band.decay.levelDb, sampleRate,
        kRt60StartDb, kRt60EndDb, -kRt60EndDb, options
    );
    band.edt = Regress(
        band.decay.levelDb, sampleRate,
        kEdtStartDb, kEdtEndDb, -kEdtEndDb, options
    );
}

void AnalyzeChannel(
    const std::vector<OctaveBands>& bins,
    double sampleRate,
    const RoomMetrics::Options& options,
    ChannelMetrics& channel
) {
    if (bins.empty()) {
        for (BandResult& band : channel.bands) FailBand(band, MetricStatus::EmptyInput);
        return;
    }
    if (!(sampleRate > 0.0) || !std::isfinite(sampleRate)) {
        for (BandResult& band : channel.bands) FailBand(band, MetricStatus::InvalidSampleRate);
        return;
    }

    for (std::size_t bandIndex = 0; bandIndex < OctaveBands::Count; ++bandIndex) {
        const ExtractedBand extracted = ExtractBand(bins, bandIndex, options);
        AnalyzeExtracted(extracted, sampleRate, options, channel.bands[bandIndex]);
    }
}

} // namespace

RoomMetricsResult RoomMetrics::Compute(const Echogram& echogram, const Options& options) {
    RoomMetricsResult result;
    result.sampleRate = echogram.sampleRate;
    const double sampleRate = static_cast<double>(echogram.sampleRate);
    AnalyzeChannel(echogram.left, sampleRate, options, result.left);
    AnalyzeChannel(echogram.right, sampleRate, options, result.right);
    return result;
}

} // namespace acoustics
