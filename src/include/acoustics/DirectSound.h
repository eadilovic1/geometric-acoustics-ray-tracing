
#pragma once

#include "acoustics/AcousticSimulation.h"

namespace acoustics {

// Racuna direktni zvuk od izvora do oba uha.
// Implementaciju cemo dodati u DirectSound.cpp.
void DepositDirectSound(
    const Source& source,
    const Receiver& receiver,
    const BVH& bvh,
    const Settings& settings,
    Echogram& echogram
);

} // namespace acoustics
