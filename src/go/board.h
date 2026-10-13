// 9x9 Go rules engine.
//
// Rules implemented (configurable where noted):
//   * Area scoring (Tromp-Taylor / Chinese style): stones + empty regions that
//     reach only one colour. Komi is added to White (default 7.0).
//   * Suicide is illegal.
//   * Ko: positional superko (default) or simple ko.
//   * The game ends after two consecutive passes or when max_moves is hit.
//
// Coordinates:
//   action  a in [0, 81): x = a % 9 (column, left->right), y = a / 9 (row, top->bottom)
//   action 81 is PASS.
//   GTP vertex: column letter "ABCDEFGHJ"[x] (no 'I'), row number 9 - y.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace go {

constexpr int N = 9;
constexpr int NN = N * N;
constexpr int W = N + 2;  // padded width (one border cell on each side)
constexpr int PADDED = W * W;
constexpr int PASS = NN;
constexpr int NUM_ACTIONS = NN + 1;
constexpr int HISTORY = 4;  // current board + 3 previous boards kept for NN features

enum Color : int8_t { EMPTY = 0, BLACK = 1, WHITE = 2, BORDER = 3 };
inline Color opponent(Color c) { return c == BLACK ? WHITE : BLACK; }

enum class KoRule { Simple, PositionalSuperko };

inline int to_padded(int action) { return (action / N + 1) * W + (action % N + 1); }
inline int to_action(int padded) { return (padded / W - 1) * N + (padded % W - 1); }
inline int action_xy(int x, int y) { return y * N + x; }

std::string action_to_gtp(int action);         // "D5", "pass"
int gtp_to_action(const std::string& vertex);  // -1 if invalid
std::string action_to_sgf(int action);         // "dd", "" for pass (we write "tt"-free empty)
int sgf_to_action(const std::string& s);       // -1 if invalid

// Compact, history-free description of a position that is enough to build
// neural-network input features and to store in the replay buffer.
struct Snapshot {
  std::array<int8_t, NN> boards[HISTORY];  // [0] = current, [1] = one move ago, ...
  uint8_t legal[NUM_ACTIONS];              // legal-move mask for the player to move
  int8_t to_move;                          // BLACK or WHITE
  int16_t last_move;                       // action, -1 if none
  float komi;
};

class Board {
 public:
  explicit Board(float komi = 7.0f, KoRule ko = KoRule::PositionalSuperko, int max_moves = 0);

  void reset();

  // Legality / play ---------------------------------------------------------
  bool is_legal(int action) const;
  // Plays an action for the side to move. Returns false (and leaves the board
  // unchanged) if the move is illegal or the game is over.
  bool play(int action);
  std::vector<int> legal_actions() const;
  void legal_mask(uint8_t* mask) const;  // NUM_ACTIONS entries

  // State -------------------------------------------------------------------
  bool is_over() const;
  Color to_move() const { return to_move_; }
  Color at(int action) const { return Color(cells_[to_padded(action)]); }
  Color at(int x, int y) const { return at(action_xy(x, y)); }
  int move_number() const { return move_num_; }
  int last_move() const { return last_move_; }
  int consecutive_passes() const { return passes_; }
  int captures(Color c) const { return captures_[c]; }  // stones captured BY colour c
  float komi() const { return komi_; }
  void set_komi(float k) { komi_ = k; }
  KoRule ko_rule() const { return ko_rule_; }
  int max_moves() const { return max_moves_; }
  uint64_t hash() const { return hash_; }
  int simple_ko_point() const { return ko_point_ < 0 ? -1 : to_action(ko_point_); }
  int stone_count(Color c) const;

  // Scoring -------------------------------------------------------------------
  // Tromp-Taylor area score: black_area - white_area - komi (positive = Black wins).
  float score() const;
  // Winner by area score (BLACK or WHITE; with integer komi a tie returns EMPTY).
  Color winner() const;
  // Area ownership: +1 black, -1 white, 0 neutral (dame / seki shared liberties).
  void area_ownership(int8_t* out) const;  // NN entries

  // Group information ---------------------------------------------------------
  int group_liberties(int action) const;            // 0 for empty points
  void liberty_map(uint8_t* out, int cap = 3) const;  // per-point liberties of its group (capped)

  Snapshot snapshot() const;
  std::string to_string() const;  // ASCII diagram

  // Recomputes the Zobrist hash from scratch (used by tests).
  uint64_t compute_hash() const;

 private:
  int count_liberties(int p, int limit) const;
  void remove_group(int p);
  bool superko_violation(uint64_t h) const;
  uint64_t hash_after(int p, Color c) const;

  std::array<int8_t, PADDED> cells_;
  std::array<std::array<int8_t, NN>, HISTORY - 1> prev_;
  std::vector<uint64_t> history_;  // positional hashes of all positions so far
  Color to_move_;
  int ko_point_;  // padded index of simple-ko forbidden point or -1
  int passes_;
  int move_num_;
  int last_move_;
  int captures_[4];
  uint64_t hash_;
  float komi_;
  KoRule ko_rule_;
  int max_moves_;
};

}  // namespace go
