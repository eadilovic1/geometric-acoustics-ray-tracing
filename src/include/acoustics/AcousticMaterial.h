#pragma once

#include <array>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>

/// 7 standard octave bands defined by ISO 3382-1:
/// 125 Hz, 250 Hz, 500 Hz, 1000 Hz, 2000 Hz, 4000 Hz, 8000 Hz.
struct OctaveBands {
    static constexpr size_t Count = 7;
    static constexpr std::array<float, Count> Frequencies = {
        125.0f, 250.0f, 500.0f, 1000.0f, 2000.0f, 4000.0f, 8000.0f
    };

    std::array<float, Count> values = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    constexpr OctaveBands() = default;
    
    constexpr explicit OctaveBands(float all) {
        values.fill(all);
    }
    
    constexpr explicit OctaveBands(const std::array<float, Count>& vals) : values(vals) {}
    
    constexpr OctaveBands(float b0, float b1, float b2, float b3, float b4, float b5, float b6)
        : values{b0, b1, b2, b3, b4, b5, b6} {}

    [[nodiscard]] static constexpr OctaveBands Zero() {
        return OctaveBands(0.0f);
    }

    [[nodiscard]] static constexpr OctaveBands One() {
        return OctaveBands(1.0f);
    }

    /// Wavelength λ = c / f for standard speed of sound in air (c = 343 m/s at 20°C).
    [[nodiscard]] static float GetWavelength(size_t bandIndex, float speedOfSound = 343.0f) {
        if (bandIndex >= Count) return 0.0f;
        return speedOfSound / Frequencies[bandIndex];
    }

    float& operator[](size_t index) { return values[index]; }
    const float& operator[](size_t index) const { return values[index]; }

    OctaveBands operator+(const OctaveBands& other) const {
        OctaveBands res;
        for (size_t i = 0; i < Count; ++i) res.values[i] = values[i] + other.values[i];
        return res;
    }

    OctaveBands operator-(const OctaveBands& other) const {
        OctaveBands res;
        for (size_t i = 0; i < Count; ++i) res.values[i] = values[i] - other.values[i];
        return res;
    }

    /// Component-wise energy multiplication.
    OctaveBands operator*(const OctaveBands& other) const {
        OctaveBands res;
        for (size_t i = 0; i < Count; ++i) res.values[i] = values[i] * other.values[i];
        return res;
    }

    OctaveBands operator*(float scalar) const {
        OctaveBands res;
        for (size_t i = 0; i < Count; ++i) res.values[i] = values[i] * scalar;
        return res;
    }

    friend OctaveBands operator*(float scalar, const OctaveBands& bands) {
        return bands * scalar;
    }

    OctaveBands operator/(float scalar) const {
        const float inv = 1.0f / scalar;
        return *this * inv;
    }

    OctaveBands& operator+=(const OctaveBands& other) {
        for (size_t i = 0; i < Count; ++i) values[i] += other.values[i];
        return *this;
    }

    OctaveBands& operator-=(const OctaveBands& other) {
        for (size_t i = 0; i < Count; ++i) values[i] -= other.values[i];
        return *this;
    }

    OctaveBands& operator*=(const OctaveBands& other) {
        for (size_t i = 0; i < Count; ++i) values[i] *= other.values[i];
        return *this;
    }

    OctaveBands& operator*=(float scalar) {
        for (size_t i = 0; i < Count; ++i) values[i] *= scalar;
        return *this;
    }

    [[nodiscard]] float Average() const {
        float sum = 0.0f;
        for (float v : values) sum += v;
        return sum / static_cast<float>(Count);
    }

    [[nodiscard]] OctaveBands Clamped(float minVal = 0.0f, float maxVal = 1.0f) const {
        OctaveBands res;
        for (size_t i = 0; i < Count; ++i) {
            res.values[i] = std::clamp(values[i], minVal, maxVal);
        }
        return res;
    }
};

/// Acoustic material properties.
/// Direct physical analog to optical PBR:
///   - absorption (α_b) is analogous to albedo (fraction of absorbed vs reflected energy: 1 - α_b)
///   - scattering (s) is analogous to roughness (fraction of diffuse vs specular reflection)
struct AcousticMaterial {
    std::string name = "Default";
    OctaveBands absorption = OctaveBands(0.1f);   // α_b ∈ [0, 1] across 7 octave bands
    float scattering = 0.1f;                    // s ∈ [0, 1] (diffuse vs specular reflection)
    OctaveBands transmission = OctaveBands(0.0f); // Optional transmission through thin partitions

    AcousticMaterial() = default;
    AcousticMaterial(std::string materialName, const OctaveBands& absCoeffs, float scatterCoeff = 0.1f, const OctaveBands& transCoeffs = OctaveBands(0.0f))
        : name(std::move(materialName)), absorption(absCoeffs), scattering(scatterCoeff), transmission(transCoeffs) {}

    /// Returns reflected energy fraction per band: (1 - α_b - τ_b).
    [[nodiscard]] OctaveBands ReflectedEnergy() const {
        OctaveBands res;
        for (size_t i = 0; i < OctaveBands::Count; ++i) {
            const float rem = 1.0f - absorption.values[i] - transmission.values[i];
            res.values[i] = std::max(0.0f, rem);
        }
        return res;
    }
};

/// Predefined standard acoustic material library (based on published architectural acoustics literature).
class AcousticMaterialLibrary {
public:
    AcousticMaterialLibrary();

    /// Global singleton instance.
    static AcousticMaterialLibrary& Instance();

    /// Registers a new material and returns its integer ID.
    uint32_t RegisterMaterial(const AcousticMaterial& material);

    /// Gets a material by ID. Returns default material if ID is invalid.
    [[nodiscard]] const AcousticMaterial& GetMaterial(uint32_t id) const;

    /// Finds material by name (case-insensitive). Returns nullptr if not found.
    [[nodiscard]] const AcousticMaterial* FindByName(const std::string& name) const;

    /// Gets material ID by name. Returns 0 (Default) if not found.
    [[nodiscard]] uint32_t GetMaterialId(const std::string& name) const;

    /// Returns all registered materials.
    [[nodiscard]] const std::vector<AcousticMaterial>& GetAllMaterials() const { return m_materials; }

    /// Total count of registered materials.
    [[nodiscard]] size_t Count() const { return m_materials.size(); }

private:
    void RegisterDefaultPresets();
    std::vector<AcousticMaterial> m_materials;
};
