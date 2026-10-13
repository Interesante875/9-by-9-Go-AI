// Batched, thread-safe neural-network inference.
#pragma once

#include <array>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <vector>

#include "nn/model.h"

namespace nn {

struct Evaluation {
  std::array<float, go::NUM_ACTIONS> policy;  // probabilities, illegal moves = 0
  float value = 0.f;                          // [-1, 1] for player to move
  float score = 0.f;                          // score lead for player to move (points)
  std::array<float, go::NN> ownership;        // +1 = player to move owns the point
};

class Evaluator {
 public:
  Evaluator(GoNet net, torch::Device device);
  // Loads a model file, or creates a randomly initialised net if path is empty.
  static std::shared_ptr<Evaluator> create(const std::string& model_path, const torch::Device& device,
                                           const NetConfig& fallback = NetConfig());

  // sym < 0 picks a random symmetry for every position; sym in [0,8) uses that one.
  std::vector<Evaluation> evaluate(const std::vector<go::Snapshot>& positions, int sym = -1);
  // Averages over all 8 symmetries (slower, stronger; used for analysis).
  Evaluation evaluate_all_syms(const go::Snapshot& position);

  GoNet& net() { return net_; }
  const torch::Device& device() const { return device_; }
  // Replaces the weights (e.g. reload a newer checkpoint) safely.
  void set_net(GoNet net);
  int64_t evaluations() const { return evals_; }

 private:
  GoNet net_;
  torch::Device device_;
  std::mutex mu_;
  std::mt19937 rng_{12345};
  int64_t evals_ = 0;
};

torch::Device pick_device(const std::string& name);  // "auto", "cpu", "cuda", "cuda:1"

}  // namespace nn
