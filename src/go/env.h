// Gym-style 9x9 Go environment built on go::Board.
//
//   GoEnv env;                       // komi 7, positional superko, 2-pass end
//   auto obs = env.reset();
//   while (true) {
//     auto legal = env.legal_actions();
//     auto r = env.step(legal[i]);   // reward from the perspective of the player who moved
//     if (r.done) break;
//   }
//
// Observation = nn::kInputPlanes x 9 x 9 float planes (see nn/features.h).
#pragma once

#include <string>
#include <vector>

#include "go/board.h"

namespace go {

struct StepResult {
  std::vector<float> observation;  // for the NEXT player to move
  float reward = 0.f;              // +1 / -1 / 0 for the player that just moved (non-zero only at the end)
  bool done = false;
  bool illegal = false;            // the action was illegal; state unchanged
};

class GoEnv {
 public:
  explicit GoEnv(float komi = 7.0f, KoRule ko = KoRule::PositionalSuperko, int max_moves = 3 * NN);

  std::vector<float> reset();
  StepResult step(int action);
  // Undo the last move by replaying the game (cheap on 9x9).
  bool undo();

  std::vector<int> legal_actions() const { return board_.legal_actions(); }
  std::vector<float> observation() const;
  bool done() const { return board_.is_over() || resigned_ != EMPTY; }
  Color to_move() const { return board_.to_move(); }
  // Final result from Black's perspective (score > 0 = Black wins). Valid when done().
  float final_score() const;
  Color winner() const;
  std::string result_string() const;  // "B+3.0", "W+R", ...
  void resign(Color who);
  Color resigned() const { return resigned_; }

  const Board& board() const { return board_; }
  const std::vector<int>& moves() const { return moves_; }
  float komi() const { return komi_; }
  void set_komi(float k);
  std::string render() const { return board_.to_string(); }

  std::string to_sgf(const std::string& black = "Black", const std::string& white = "White") const;
  // Loads the main line of an SGF game record (board size must be 9). Returns false on error.
  bool load_sgf(const std::string& sgf, std::string* error = nullptr);

 private:
  float komi_;
  KoRule ko_;
  int max_moves_;
  Board board_;
  std::vector<int> moves_;
  Color resigned_ = EMPTY;
};

// "B+3.5" / "W+0.5" / "0" from a Black-perspective score.
std::string score_to_result(float score);

// Minimal SGF writer used by self-play logging too.
std::string make_sgf(const std::vector<int>& moves, float komi, const std::string& result,
                     const std::string& black = "Black", const std::string& white = "White",
                     const std::string& comment = "");

}  // namespace go
