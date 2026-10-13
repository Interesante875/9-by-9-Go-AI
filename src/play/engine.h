// Game + search wrapper shared by the web GUI and the GTP front-end.
#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "go/env.h"
#include "mcts/mcts.h"
#include "nn/evaluator.h"

namespace play {

struct Analysis {
  int visits = 0;
  go::Color to_move = go::BLACK;
  float winrate_black = 0.5f;  // root winrate from Black's perspective
  float score_black = 0.f;     // root score lead from Black's perspective
  float nn_value_black = 0.f;  // raw network value (no search)
  std::vector<mcts::MoveStat> moves;  // winrate/score from the perspective of the player to move
  std::array<float, go::NUM_ACTIONS> policy{};
  std::array<float, go::NN> ownership_black{};  // +1 Black owns, -1 White owns
};

class PlayEngine {
 public:
  PlayEngine(std::shared_ptr<nn::Evaluator> eval, const std::string& model_path, float komi = 7.0f);

  std::mutex& mutex() { return mu_; }

  // All methods below assume the caller holds mutex() when used from several threads.
  const go::GoEnv& env() const { return env_; }
  void new_game(float komi);
  bool play(int action);
  bool undo();
  void resign(go::Color who);
  bool load_sgf(const std::string& sgf, std::string* err);

  // Adds `visits` playouts to the current search tree (tree is reused between calls).
  Analysis analyze(int visits, int max_total = 200000);
  // Searches `visits` playouts (fresh-ish: reuses tree) and returns the chosen action
  // (or -2 if the engine resigns). Does NOT play it.
  int choose_move(int visits, bool allow_resign, Analysis* out);

  const std::vector<float>& winrate_history() const { return winrate_hist_; }  // Black's perspective, NaN = unknown
  const std::vector<float>& score_history() const { return score_hist_; }

  bool reload(const std::string& path, std::string* err);
  const std::string& model_path() const { return model_path_; }
  nn::Evaluator& evaluator() { return *eval_; }
  mcts::Search& search() { return *search_; }

 private:
  void sync_history();
  void record_quick_eval();
  Analysis make_analysis();

  std::shared_ptr<nn::Evaluator> eval_;
  std::string model_path_;
  go::GoEnv env_;
  std::unique_ptr<mcts::Search> search_;
  mcts::Config mcfg_;
  std::vector<float> winrate_hist_, score_hist_;
  std::mutex mu_;
};

}  // namespace play
