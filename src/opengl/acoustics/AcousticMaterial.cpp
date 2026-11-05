#include "acoustics/AcousticMaterial.h"
#include <algorithm>
#include <cctype>

namespace {

std::string ToLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

} // namespace

AcousticMaterialLibrary::AcousticMaterialLibrary() {
    RegisterDefaultPresets();
}

AcousticMaterialLibrary& AcousticMaterialLibrary::Instance() {
    static AcousticMaterialLibrary s_instance;
    return s_instance;
}

void AcousticMaterialLibrary::RegisterDefaultPresets() {
    m_materials.clear();

    // 0: Default
    RegisterMaterial(AcousticMaterial(
        "Default",
        OctaveBands(0.10f, 0.10f, 0.10f, 0.10f, 0.10f, 0.10f, 0.10f),
        0.10f
    ));

    // 1: Concrete (smooth unpainted concrete) - highly reflective, low absorption
    RegisterMaterial(AcousticMaterial(
        "Concrete",
        OctaveBands(0.01f, 0.01f, 0.02f, 0.02f, 0.02f, 0.03f, 0.03f),
        0.05f
    ));

    // 2: WoodPanels (standard wood paneling on battens)
    RegisterMaterial(AcousticMaterial(
        "WoodPanels",
        OctaveBands(0.28f, 0.22f, 0.17f, 0.09f, 0.10f, 0.11f, 0.11f),
        0.15f
    ));

    // 3: Glass (large window glass panes) - low absorption across all frequencies
    RegisterMaterial(AcousticMaterial(
        "Glass",
        OctaveBands(0.18f, 0.06f, 0.04f, 0.03f, 0.02f, 0.02f, 0.02f),
        0.05f
    ));

    // 4: Carpet (heavy carpet on concrete) - strong high frequency absorption
    RegisterMaterial(AcousticMaterial(
        "Carpet",
        OctaveBands(0.02f, 0.06f, 0.14f, 0.37f, 0.60f, 0.65f, 0.70f),
        0.35f
    ));

    // 5: Plasterboard (gypsum board / drywall)
    RegisterMaterial(AcousticMaterial(
        "Plasterboard",
        OctaveBands(0.29f, 0.10f, 0.05f, 0.04f, 0.07f, 0.09f, 0.09f),
        0.10f
    ));

    // 6: HeadSkinHair (human head tissue for binaural ILD / head shadowing)
    RegisterMaterial(AcousticMaterial(
        "HeadSkinHair",
        OctaveBands(0.10f, 0.15f, 0.25f, 0.40f, 0.65f, 0.80f, 0.85f),
        0.50f
    ));

    // 7: Brickwork (exposed brickwork, unpointed/coarse)
    RegisterMaterial(AcousticMaterial(
        "Brickwork",
        OctaveBands(0.02f, 0.02f, 0.03f, 0.04f, 0.05f, 0.07f, 0.07f),
        0.25f
    ));

    // 8: CeramicTile (glazed ceramic or marble tile)
    RegisterMaterial(AcousticMaterial(
        "CeramicTile",
        OctaveBands(0.01f, 0.01f, 0.01f, 0.01f, 0.02f, 0.02f, 0.02f),
        0.05f
    ));

    // 9: CurtainHeavy (draped velour curtains)
    RegisterMaterial(AcousticMaterial(
        "CurtainHeavy",
        OctaveBands(0.14f, 0.35f, 0.55f, 0.72f, 0.70f, 0.65f, 0.60f),
        0.40f
    ));

    // 10: Uniform02 (α = 0.2 across all bands - matches the exact validation room in Section 10 of PRS_Projekat.pdf!)
    RegisterMaterial(AcousticMaterial(
        "Uniform02",
        OctaveBands(0.20f, 0.20f, 0.20f, 0.20f, 0.20f, 0.20f, 0.20f),
        0.15f
    ));

    // 11: AnechoicAbsorber (100% absorption, 0% reflection - perfect free-field / anechoic chamber test)
    RegisterMaterial(AcousticMaterial(
        "AnechoicAbsorber",
        OctaveBands(1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f),
        0.00f
    ));

    // 12: PerfectReflector (0% absorption, 100% reflection - lossless acoustic boundary test)
    RegisterMaterial(AcousticMaterial(
        "PerfectReflector",
        OctaveBands(0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f),
        0.00f
    ));
}

uint32_t AcousticMaterialLibrary::RegisterMaterial(const AcousticMaterial& material) {
    const uint32_t id = static_cast<uint32_t>(m_materials.size());
    m_materials.push_back(material);
    return id;
}

const AcousticMaterial& AcousticMaterialLibrary::GetMaterial(uint32_t id) const {
    if (id < m_materials.size()) {
        return m_materials[id];
    }
    static const AcousticMaterial s_fallback("Fallback", OctaveBands(0.1f), 0.1f);
    return s_fallback;
}

const AcousticMaterial* AcousticMaterialLibrary::FindByName(const std::string& name) const {
    const std::string target = ToLower(name);
    for (const auto& mat : m_materials) {
        if (ToLower(mat.name) == target) {
            return &mat;
        }
    }
    return nullptr;
}

uint32_t AcousticMaterialLibrary::GetMaterialId(const std::string& name) const {
    const std::string target = ToLower(name);
    for (size_t i = 0; i < m_materials.size(); ++i) {
        if (ToLower(m_materials[i].name) == target) {
            return static_cast<uint32_t>(i);
        }
    }
    return 0; // Default
}
