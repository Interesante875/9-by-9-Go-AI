// Head-to-head matches between two networks (or a network and a random player).
#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "nn/evaluator.h"

namespace train {

struct MatchConfig {
  int games = 40;
  int sims = 64;            // MCTS visits per move (0 = play the raw policy argmax)
  int parallel_games = 20;
  float komi = 7.0f;
  int max_moves = 200;
  int temp_moves = 4;       // opening randomisation for game diversity
  float temperature = 0.5f;
  uint64_t seed = 1;
};

struct MatchResult {
  int a_wins = 0, b_wins = 0, draws = 0;
  int a_black_wins = 0, a_white_wins = 0;
  double avg_length = 0;
  std::vector<std::string> sgfs;
  double a_score() const { return games() ? (a_wins + 0.5 * draws) / games() : 0.5; }
  int games() const { return a_wins + b_wins + draws; }
};

// A null evaluator plays uniformly random legal moves (never filling its own single eyes).
MatchResult play_match(std::shared_ptr<nn::Evaluator> a, std::shared_ptr<nn::Evaluator> b, const MatchConfig& cfg,
                       const std::atomic<bool>* stop = nullptr, const std::string& name_a = "A",
                       const std::string& name_b = "B");

// Elo difference implied by an expected score (clamped to avoid infinities).
double elo_from_score(double score);

}  // namespace train
