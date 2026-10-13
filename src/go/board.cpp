#include "go/board.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>

namespace go {

namespace {

const char* kColumns = "ABCDEFGHJ";
const int kDirs[4] = {-W, W, -1, 1};

struct Zobrist {
  uint64_t table[PADDED][3];
  Zobrist() {
    uint64_t s = 0x9E3779B97F4A7C15ULL;
    auto next = [&s]() {
      // splitmix64
      uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
      z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
      z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
      return z ^ (z >> 31);
    };
    for (int i = 0; i < PADDED; ++i)
      for (int c = 0; c < 3; ++c) table[i][c] = next();
  }
};

const Zobrist& zobrist() {
  static const Zobrist z;
  return z;
}

}  // namespace

std::string action_to_gtp(int action) {
  if (action == PASS) return "pass";
  if (action < 0 || action >= NN) return "invalid";
  int x = action % N, y = action / N;
  return std::string(1, kColumns[x]) + std::to_string(N - y);
}

int gtp_to_action(const std::string& vertex) {
  std::string v;
  for (char ch : vertex) v += char(std::toupper(static_cast<unsigned char>(ch)));
  if (v == "PASS") return PASS;
  if (v.size() < 2) return -1;
  int x = -1;
  for (int i = 0; i < N; ++i)
    if (kColumns[i] == v[0]) x = i;
  if (x < 0) return -1;
  int row = 0;
  for (size_t i = 1; i < v.size(); ++i) {
    if (!std::isdigit(static_cast<unsigned char>(v[i]))) return -1;
    row = row * 10 + (v[i] - '0');
  }
  if (row < 1 || row > N) return -1;
  return action_xy(x, N - row);
}

std::string action_to_sgf(int action) {
  if (action == PASS || action < 0 || action >= NN) return "";
  int x = action % N, y = action / N;
  return std::string(1, char('a' + x)) + std::string(1, char('a' + y));
}

int sgf_to_action(const std::string& s) {
  if (s.empty() || s == "tt") return PASS;
  if (s.size() != 2) return -1;
  int x = s[0] - 'a', y = s[1] - 'a';
  if (x < 0 || x >= N || y < 0 || y >= N) return -1;
  return action_xy(x, y);
}

Board::Board(float komi, KoRule ko, int max_moves) : komi_(komi), ko_rule_(ko), max_moves_(max_moves) {
  reset();
}

void Board::reset() {
  cells_.fill(BORDER);
  for (int y = 0; y < N; ++y)
    for (int x = 0; x < N; ++x) cells_[(y + 1) * W + x + 1] = EMPTY;
  for (auto& b : prev_) b.fill(EMPTY);
  to_move_ = BLACK;
  ko_point_ = -1;
  passes_ = 0;
  move_num_ = 0;
  last_move_ = -1;
  captures_[0] = captures_[1] = captures_[2] = captures_[3] = 0;
  hash_ = 0;
  history_.clear();
  history_.reserve(128);
  history_.push_back(hash_);
}

int Board::count_liberties(int p, int limit) const {
  // Flood fill the group containing p and count distinct liberties, stopping
  // early once `limit` liberties have been found.
  const int8_t color = cells_[p];
  std::array<uint8_t, PADDED> seen{};
  int stack[PADDED];
  int sp = 0;
  stack[sp++] = p;
  seen[p] = 1;
  int libs = 0;
  while (sp) {
    int q = stack[--sp];
    for (int d : kDirs) {
      int r = q + d;
      if (seen[r]) continue;
      if (cells_[r] == EMPTY) {
        seen[r] = 1;
        if (++libs >= limit) return libs;
      } else if (cells_[r] == color) {
        seen[r] = 1;
        stack[sp++] = r;
      }
    }
  }
  return libs;
}

void Board::remove_group(int p) {
  const int8_t color = cells_[p];
  const auto& z = zobrist();
  int stack[PADDED];
  int sp = 0;
  stack[sp++] = p;
  cells_[p] = EMPTY;
  hash_ ^= z.table[p][color];
  int removed = 1;
  while (sp) {
    int q = stack[--sp];
    for (int d : kDirs) {
      int r = q + d;
      if (cells_[r] == color) {
        cells_[r] = EMPTY;
        hash_ ^= z.table[r][color];
        ++removed;
        stack[sp++] = r;
      }
    }
  }
  captures_[opponent(Color(color))] += removed;
}

uint64_t Board::hash_after(int p, Color c) const {
  // Hash of the position after c plays at p (assumes the move is not suicide).
  const auto& z = zobrist();
  uint64_t h = hash_ ^ z.table[p][c];
  const Color o = opponent(c);
  std::array<uint8_t, PADDED> removed{};
  for (int d : kDirs) {
    int n = p + d;
    if (cells_[n] != o || removed[n]) continue;
    if (count_liberties(n, 2) != 1) continue;
    // Group at n is captured: xor out all its stones.
    int stack[PADDED];
    int sp = 0;
    stack[sp++] = n;
    removed[n] = 1;
    while (sp) {
      int q = stack[--sp];
      h ^= z.table[q][o];
      for (int dd : kDirs) {
        int r = q + dd;
        if (cells_[r] == o && !removed[r]) {
          removed[r] = 1;
          stack[sp++] = r;
        }
      }
    }
  }
  return h;
}

bool Board::superko_violation(uint64_t h) const {
  return std::find(history_.begin(), history_.end(), h) != history_.end();
}

bool Board::is_over() const {
  return passes_ >= 2 || (max_moves_ > 0 && move_num_ >= max_moves_);
}

bool Board::is_legal(int action) const {
  if (is_over()) return false;
  if (action == PASS) return true;
  if (action < 0 || action >= NN) return false;
  const int p = to_padded(action);
  if (cells_[p] != EMPTY) return false;
  if (ko_rule_ == KoRule::Simple && p == ko_point_) return false;
  const Color c = to_move_, o = opponent(c);
  bool has_liberty = false, captures = false;
  for (int d : kDirs) {
    int n = p + d;
    if (cells_[n] == EMPTY) {
      has_liberty = true;
    } else if (cells_[n] == c) {
      // Friendly group keeps a liberty other than p?
      if (count_liberties(n, 2) >= 2) has_liberty = true;
    } else if (cells_[n] == o) {
      if (count_liberties(n, 2) == 1) captures = true;
    }
  }
  if (!has_liberty && !captures) return false;  // suicide
  if (ko_rule_ == KoRule::PositionalSuperko) {
    if (superko_violation(hash_after(p, c))) return false;
  }
  return true;
}

bool Board::play(int action) {
  if (!is_legal(action)) return false;
  // Shift feature history.
  for (int i = HISTORY - 2; i > 0; --i) prev_[i] = prev_[i - 1];
  for (int a = 0; a < NN; ++a) prev_[0][a] = cells_[to_padded(a)];

  const Color c = to_move_, o = opponent(c);
  if (action == PASS) {
    ++passes_;
    ko_point_ = -1;
  } else {
    passes_ = 0;
    const int p = to_padded(action);
    cells_[p] = c;
    hash_ ^= zobrist().table[p][c];
    int captured_total = 0, captured_point = -1;
    for (int d : kDirs) {
      int n = p + d;
      if (cells_[n] == o && count_liberties(n, 1) == 0) {
        int before = captures_[c];
        remove_group(n);
        captured_total += captures_[c] - before;
        captured_point = n;
      }
    }
    // Simple ko: a single stone captured a single stone and is now in atari.
    ko_point_ = -1;
    if (captured_total == 1) {
      bool lone = true;
      for (int d : kDirs)
        if (cells_[p + d] == c) lone = false;
      if (lone && count_liberties(p, 2) == 1) ko_point_ = captured_point;
    }
  }
  history_.push_back(hash_);
  last_move_ = action;
  ++move_num_;
  to_move_ = o;
  return true;
}

std::vector<int> Board::legal_actions() const {
  std::vector<int> out;
  out.reserve(NUM_ACTIONS);
  for (int a = 0; a < NUM_ACTIONS; ++a)
    if (is_legal(a)) out.push_back(a);
  return out;
}

void Board::legal_mask(uint8_t* mask) const {
  for (int a = 0; a < NUM_ACTIONS; ++a) mask[a] = is_legal(a) ? 1 : 0;
}

int Board::stone_count(Color c) const {
  int n = 0;
  for (int a = 0; a < NN; ++a) n += cells_[to_padded(a)] == c;
  return n;
}

void Board::area_ownership(int8_t* out) const {
  std::array<uint8_t, PADDED> seen{};
  for (int a = 0; a < NN; ++a) {
    int p = to_padded(a);
    if (cells_[p] == BLACK) out[a] = 1;
    else if (cells_[p] == WHITE) out[a] = -1;
  }
  for (int a = 0; a < NN; ++a) {
    int p = to_padded(a);
    if (cells_[p] != EMPTY || seen[p]) continue;
    // Flood fill empty region and record which colours it touches.
    std::vector<int> region;
    int stack[PADDED];
    int sp = 0;
    stack[sp++] = p;
    seen[p] = 1;
    bool touches_black = false, touches_white = false;
    while (sp) {
      int q = stack[--sp];
      region.push_back(q);
      for (int d : kDirs) {
        int r = q + d;
        if (cells_[r] == BLACK) touches_black = true;
        else if (cells_[r] == WHITE) touches_white = true;
        else if (cells_[r] == EMPTY && !seen[r]) {
          seen[r] = 1;
          stack[sp++] = r;
        }
      }
    }
    int8_t owner = (touches_black && !touches_white) ? 1 : (touches_white && !touches_black) ? -1 : 0;
    for (int q : region) out[to_action(q)] = owner;
  }
}

float Board::score() const {
  int8_t own[NN];
  area_ownership(own);
  int s = 0;
  for (int a = 0; a < NN; ++a) s += own[a];
  return float(s) - komi_;
}

Color Board::winner() const {
  float s = score();
  if (s > 0) return BLACK;
  if (s < 0) return WHITE;
  return EMPTY;
}

int Board::group_liberties(int action) const {
  int p = to_padded(action);
  if (cells_[p] != BLACK && cells_[p] != WHITE) return 0;
  return count_liberties(p, PADDED);
}

void Board::liberty_map(uint8_t* out, int cap) const {
  std::array<uint8_t, PADDED> done{};
  for (int a = 0; a < NN; ++a) out[a] = 0;
  for (int a = 0; a < NN; ++a) {
    int p = to_padded(a);
    if (done[p] || (cells_[p] != BLACK && cells_[p] != WHITE)) continue;
    const int8_t color = cells_[p];
    // Collect group and its liberties.
    std::vector<int> stones;
    std::array<uint8_t, PADDED> seen{};
    int stack[PADDED];
    int sp = 0;
    stack[sp++] = p;
    seen[p] = 1;
    int libs = 0;
    while (sp) {
      int q = stack[--sp];
      stones.push_back(q);
      for (int d : kDirs) {
        int r = q + d;
        if (seen[r]) continue;
        if (cells_[r] == EMPTY) {
          seen[r] = 1;
          ++libs;
        } else if (cells_[r] == color) {
          seen[r] = 1;
          stack[sp++] = r;
        }
      }
    }
    uint8_t v = uint8_t(std::min(libs, cap));
    for (int q : stones) {
      done[q] = 1;
      out[to_action(q)] = v;
    }
  }
}

Snapshot Board::snapshot() const {
  Snapshot s;
  for (int a = 0; a < NN; ++a) s.boards[0][a] = cells_[to_padded(a)];
  for (int i = 1; i < HISTORY; ++i) s.boards[i] = prev_[i - 1];
  legal_mask(s.legal);
  s.to_move = to_move_;
  s.last_move = int16_t(last_move_);
  s.komi = komi_;
  return s;
}

uint64_t Board::compute_hash() const {
  uint64_t h = 0;
  for (int a = 0; a < NN; ++a) {
    int p = to_padded(a);
    if (cells_[p] == BLACK || cells_[p] == WHITE) h ^= zobrist().table[p][cells_[p]];
  }
  return h;
}

std::string Board::to_string() const {
  std::ostringstream os;
  os << "   ";
  for (int x = 0; x < N; ++x) os << kColumns[x] << ' ';
  os << '\n';
  for (int y = 0; y < N; ++y) {
    int row = N - y;
    os << (row < 10 ? " " : "") << row << ' ';
    for (int x = 0; x < N; ++x) {
      int a = action_xy(x, y);
      Color c = at(a);
      char ch = c == BLACK ? 'X' : c == WHITE ? 'O' : '.';
      if (c == EMPTY && (x == 2 || x == 6 || x == 4) && (y == 2 || y == 6 || y == 4) &&
          !((x == 4) != (y == 4)))
        ch = '+';
      os << ch << (a == last_move_ ? ')' : ' ');
    }
    os << row << '\n';
  }
  os << "   ";
  for (int x = 0; x < N; ++x) os << kColumns[x] << ' ';
  os << "\nTo move: " << (to_move_ == BLACK ? "Black (X)" : "White (O)") << "  Move: " << move_num_
     << "  Captures B:" << captures_[BLACK] << " W:" << captures_[WHITE] << "  Komi: " << komi_ << '\n';
  return os.str();
}

}  // namespace go
