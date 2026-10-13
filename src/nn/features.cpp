#include "nn/features.h"

#include <algorithm>
#include <cstring>
#include <vector>

namespace nn {

using namespace go;

namespace {

struct SymTables {
  int fwd[8][NUM_ACTIONS];
  SymTables() {
    for (int s = 0; s < 8; ++s) {
      for (int a = 0; a < NN; ++a) {
        int x = a % N, y = a / N;
        int tx = x, ty = y;
        if (s & 1) tx = N - 1 - tx;         // horizontal flip
        if (s & 2) ty = N - 1 - ty;         // vertical flip
        if (s & 4) std::swap(tx, ty);       // transpose
        fwd[s][a] = ty * N + tx;
      }
      fwd[s][PASS] = PASS;
    }
  }
};

const SymTables& tables() {
  static const SymTables t;
  return t;
}

}  // namespace

int sym_action(int sym, int action) {
  if (action < 0) return action;
  return tables().fwd[sym][action];
}

int inverse_sym(int sym) {
  // Flips and transposes: transpose-after-flip needs the flips swapped.
  if (!(sym & 4)) return sym;
  int fx = sym & 1, fy = (sym >> 1) & 1;
  return 4 | (fy) | (fx << 1);
}

void liberty_map_from_cells(const std::array<int8_t, NN>& cells, uint8_t* out, int cap) {
  std::fill(out, out + NN, uint8_t(0));
  std::array<uint8_t, NN> done{};
  std::vector<int> stones;
  stones.reserve(NN);
  for (int a = 0; a < NN; ++a) {
    if (done[a] || (cells[a] != BLACK && cells[a] != WHITE)) continue;
    const int8_t color = cells[a];
    std::array<uint8_t, NN> seen{};
    stones.clear();
    int stack[NN];
    int sp = 0;
    stack[sp++] = a;
    seen[a] = 1;
    int libs = 0;
    while (sp) {
      int q = stack[--sp];
      stones.push_back(q);
      int x = q % N, y = q / N;
      int nbr[4];
      int k = 0;
      if (x > 0) nbr[k++] = q - 1;
      if (x < N - 1) nbr[k++] = q + 1;
      if (y > 0) nbr[k++] = q - N;
      if (y < N - 1) nbr[k++] = q + N;
      for (int i = 0; i < k; ++i) {
        int r = nbr[i];
        if (seen[r]) continue;
        if (cells[r] == EMPTY) {
          seen[r] = 1;
          ++libs;
        } else if (cells[r] == color) {
          seen[r] = 1;
          stack[sp++] = r;
        }
      }
    }
    uint8_t v = uint8_t(std::min(libs, cap));
    for (int q : stones) {
      done[q] = 1;
      out[q] = v;
    }
  }
}

void encode(const Snapshot& snap, int sym, float* out) {
  std::memset(out, 0, sizeof(float) * kInputSize);
  const int8_t own = snap.to_move;
  const int8_t opp = own == BLACK ? WHITE : BLACK;
  const int* map = tables().fwd[sym];
  auto plane = [&](int p) { return out + p * NN; };

  for (int h = 0; h < HISTORY; ++h) {
    float* po = plane(2 * h);
    float* pp = plane(2 * h + 1);
    for (int a = 0; a < NN; ++a) {
      int8_t c = snap.boards[h][a];
      if (c == own) po[map[a]] = 1.f;
      else if (c == opp) pp[map[a]] = 1.f;
    }
  }
  uint8_t libs[NN];
  liberty_map_from_cells(snap.boards[0], libs, 3);
  for (int a = 0; a < NN; ++a) {
    int8_t c = snap.boards[0][a];
    if (c != BLACK && c != WHITE) continue;
    int base = c == own ? 8 : 11;
    plane(base + libs[a] - 1)[map[a]] = 1.f;
  }
  float* legal = plane(14);
  for (int a = 0; a < NN; ++a)
    if (snap.legal[a]) legal[map[a]] = 1.f;
  if (snap.last_move >= 0 && snap.last_move < NN) plane(15)[map[snap.last_move]] = 1.f;
  float* ones = plane(16);
  float* black = plane(17);
  float* komi = plane(18);
  const float k = (own == WHITE ? snap.komi : -snap.komi) / 15.f;
  float* passed = plane(19);
  const float p = snap.last_move == PASS ? 1.f : 0.f;
  for (int a = 0; a < NN; ++a) {
    ones[a] = 1.f;
    black[a] = own == BLACK ? 1.f : 0.f;
    komi[a] = k;
    passed[a] = p;
  }
}

}  // namespace nn
