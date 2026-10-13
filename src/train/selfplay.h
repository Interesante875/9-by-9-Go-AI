// Batched multi-threaded self-play game generation.
#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "nn/evaluator.h"
#include "train/config.h"
#include "train/replay.h"

namespace train {

struct GameResult {
  std::vector<Sample> samples;  // training samples from this game
  std::vector<int> moves;
  float score = 0;              // final area score, Black's perspective (0 if resigned)
  go::Color winner = go::EMPTY;
  bool resigned = false;
  bool resign_allowed = true;
  bool false_resign = false;    // no-resign game in which the eventual winner would have resigned
  std::string sgf;
};

struct SelfPlayStats {
  int games = 0;
  int moves = 0;
  int samples = 0;
  int black_wins = 0;
  int resigned = 0;
  int noresign_games = 0;
  int false_resigns = 0;
  double seconds = 0;
  int64_t nn_evals = 0;
};

class SelfPlay {
 public:
  SelfPlay(const TrainConfig& cfg, std::shared_ptr<nn::Evaluator> eval);
  // Plays `n` games (MCTS self-play, or raw-policy rollouts when algo == "ppo").
  std::vector<GameResult> play(int n, uint64_t seed, const std::atomic<bool>* stop, SelfPlayStats* stats);
  void set_resign_enabled(bool on) { resign_enabled_ = on; }

 private:
  void worker_mcts(int thread_id, int n, uint64_t seed, const std::atomic<bool>* stop,
                   std::atomic<int>* started, std::vector<GameResult>* out, std::mutex* out_mu);
  void worker_policy(int thread_id, int n, uint64_t seed, const std::atomic<bool>* stop,
                     std::atomic<int>* started, std::vector<GameResult>* out, std::mutex* out_mu);

  TrainConfig cfg_;
  std::shared_ptr<nn::Evaluator> eval_;
  bool resign_enabled_ = true;
};

// Per-move record collected during a game, turned into Samples at the end.
struct MoveRecord {
  go::Snapshot snap;
  std::array<float, go::NUM_ACTIONS> pi;
  float root_q = 0;
  float nn_value = 0;
  float nn_logp = 0;
  int action = 0;
  bool train = true;
};

// Builds training samples (outcome, ownership, score, GAE advantages) for a finished game.
void finalize_game(const std::vector<MoveRecord>& recs, const go::Board& final_board, go::Color winner,
                   bool resigned, float gae_lambda, GameResult* out);

}  // namespace train
