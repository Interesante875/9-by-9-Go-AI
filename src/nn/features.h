// Neural-network input encoding and the 8 dihedral board symmetries.
#pragma once

#include <array>

#include "go/board.h"

namespace nn {

// Input planes (all from the perspective of the player to move):
//   0..7   own / opponent stones for the current and 3 previous positions
//   8..10  own groups with 1, 2, >=3 liberties
//   11..13 opponent groups with 1, 2, >=3 liberties
//   14     legal moves for the player to move
//   15     last move
//   16     ones (marks the on-board area for zero-padded convolutions)
//   17     1 if Black to move
//   18     komi from the mover's point of view / 15
constexpr int kInputPlanes = 19;
constexpr int kInputSize = kInputPlanes * go::NN;

// Symmetry s in [0, 8): maps an on-board action through rotation/reflection.
int sym_action(int sym, int action);
// Inverse symmetry index such that sym_action(inverse_sym(s), sym_action(s, a)) == a.
int inverse_sym(int sym);

// Writes kInputSize floats for `snap` transformed by symmetry `sym`.
void encode(const go::Snapshot& snap, int sym, float* out);

// Recomputes liberties from a raw board array (row-major NN cells).
void liberty_map_from_cells(const std::array<int8_t, go::NN>& cells, uint8_t* out, int cap = 3);

}  // namespace nn
