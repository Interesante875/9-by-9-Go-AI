// Go Text Protocol front-end, so the engine can be used from Sabaki, Lizzie, GoGui, etc.
// Supports lz-analyze / kata-analyze for live win-rate overlays in those GUIs.
#pragma once

#include <memory>

#include "play/engine.h"

namespace gtp {

int run_gtp(std::shared_ptr<play::PlayEngine> engine, int visits);

}  // namespace gtp
