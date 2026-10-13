// Self-play reinforcement-learning loop with checkpoint/resume.
//
//   iteration:  self-play games  ->  prioritised replay  ->  SGD steps  ->  log / save / evaluate
//
// Everything needed to resume lives in the run directory:
//   config.ini, model_latest.pt, optimizer_latest.pt, state.txt, replay.bin,
//   metrics.csv, eval.csv, train.log, snapshots/, sgf/
#pragma once

#include <atomic>
#include <map>
#include <memory>
#include <random>
#include <string>

#include "nn/evaluator.h"
#include "train/config.h"
#include "train/replay.h"

namespace train {

struct LossStats {
  double total = 0, policy = 0, value = 0, ownership = 0, score = 0, ppo = 0, entropy = 0, kl = 0, value_mse = 0;
  double clip_frac = 0;
  int steps = 0;
};

class Trainer {
 public:
  explicit Trainer(const TrainConfig& cfg);
  // Runs until max_iters or until *stop becomes true. Returns 0 on success.
  int run(std::atomic<bool>* stop);

  // Exposed for tests.
  LossStats train_steps(int steps);
  ReplayBuffer& replay() { return *replay_; }
  nn::GoNet& net() { return net_; }
  void save_checkpoint(bool with_replay);
  bool load_checkpoint();
  int iteration() const { return iter_; }

 private:
  void setup_optimizer();
  void set_lr(double lr);
  double current_lr() const;
  void evaluate(std::atomic<bool>* stop);
  void write_state() const;
  bool read_state();
  std::string path(const std::string& name) const;

  TrainConfig cfg_;
  torch::Device device_;
  nn::GoNet net_{nullptr};
  std::shared_ptr<nn::Evaluator> evaluator_;
  std::unique_ptr<torch::optim::Optimizer> opt_;
  std::unique_ptr<ReplayBuffer> replay_;
  std::mt19937_64 rng_;

  int iter_ = 0;
  int64_t total_steps_ = 0;
  int64_t total_games_ = 0;
  int64_t total_samples_ = 0;
  double elo_ = 0;
  double train_hours_ = 0;
};

}  // namespace train
