// Prioritized experience replay (Schaul et al. 2015) backed by a sum tree.
#pragma once

#include <array>
#include <cstdint>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "go/board.h"

namespace train {

struct Sample {
  go::Snapshot pos;
  float policy[go::NUM_ACTIONS];  // search policy target (visit distribution)
  int8_t ownership[go::NN];       // final area ownership, +1 = player to move
  float z = 0;                    // game outcome for player to move (+1 / -1 / 0)
  float score = 0;                // final score lead for player to move (points)
  float root_q = 0;               // MCTS root value at the time
  float old_logp = 0;             // log pi_net(action) of the behaviour network (for PPO)
  float old_value = 0;            // V_net(s) of the behaviour network
  float advantage = 0;            // GAE advantage for the played action
  float policy_weight = 1;        // 0 for positions without a full search
  float aux_weight = 1;           // ownership/score weight (0 when the game ended by resignation)
  int16_t action = 0;             // action played in the game
};
static_assert(std::is_trivially_copyable<Sample>::value, "Sample must be POD for fast (de)serialisation");

class SumTree {
 public:
  explicit SumTree(size_t capacity = 1);
  void set(size_t i, double p);
  double get(size_t i) const { return tree_[leaf0_ + i]; }
  double total() const { return tree_[1]; }
  // Finds the leaf whose cumulative range contains `mass` (0 <= mass < total).
  size_t find(double mass) const;
  size_t capacity() const { return capacity_; }

 private:
  size_t capacity_, leaf0_;
  std::vector<double> tree_;
};

class ReplayBuffer {
 public:
  ReplayBuffer(size_t capacity, float alpha = 0.6f, float eps = 1e-3f);

  void add(const Sample& s);
  size_t size() const { return size_; }
  size_t capacity() const { return capacity_; }
  uint64_t total_added() const { return total_added_; }

  struct Batch {
    std::vector<size_t> indices;
    std::vector<float> weights;  // normalised importance-sampling weights
  };
  // Stratified proportional sampling. beta controls IS correction (0 = none, 1 = full).
  Batch sample(size_t batch_size, float beta, std::mt19937_64& rng) const;
  void update_priorities(const std::vector<size_t>& indices, const std::vector<float>& td_errors);
  const Sample& at(size_t i) const { return data_[i]; }
  float priority(size_t i) const { return float(tree_.get(i)); }
  float max_priority() const { return max_priority_; }

  bool save(const std::string& path) const;
  bool load(const std::string& path);

 private:
  size_t capacity_;
  float alpha_, eps_;
  std::vector<Sample> data_;
  SumTree tree_;
  size_t next_ = 0, size_ = 0;
  uint64_t total_added_ = 0;
  float max_priority_ = 1.f;
};

}  // namespace train
