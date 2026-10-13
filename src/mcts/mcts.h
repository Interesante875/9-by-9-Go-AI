// PUCT Monte-Carlo Tree Search (AlphaZero / KataGo style) with
//   * batched leaf evaluation via virtual loss,
//   * first-play-urgency reduction,
//   * Dirichlet noise at the root for self-play exploration,
//   * subtree reuse between moves,
//   * exact evaluation of terminal positions with Tromp-Taylor scoring.
#pragma once

#include <array>
#include <memory>
#include <random>
#include <vector>

#include "go/board.h"
#include "nn/evaluator.h"

namespace mcts {

struct Config {
  float c_puct_init = 1.1f;
  float c_puct_base = 19652.f;
  float fpu_reduction = 0.2f;       // non-root unvisited children: parent_q - r * sqrt(visited prior mass)
  float root_fpu_reduction = 0.0f;  // at the root (exploration-friendly)
  bool root_noise = false;
  float noise_eps = 0.25f;
  float noise_alpha_total = 10.83f;  // alpha = total / num_legal (KataGo: 0.03 * 361)
  int virtual_loss = 1;
  bool keep_ownership = false;       // keep NN ownership per node (analysis)
};

struct Node;

struct Edge {
  int16_t move = 0;
  float prior = 0.f;      // prior used in search (may include noise)
  float prior_raw = 0.f;  // network prior
  std::unique_ptr<Node> child;
};

struct Node {
  std::vector<Edge> edges;
  int n = 0;            // visits (including the node's own evaluation)
  int vl = 0;           // in-flight virtual losses
  double w = 0;         // sum of values from the perspective of the player to move here
  double score_sum = 0; // sum of score leads, same perspective
  float nn_value = 0.f, nn_score = 0.f;
  bool expanded = false, pending = false, terminal = false;
  std::unique_ptr<std::array<float, go::NN>> ownership;  // optional, player-to-move perspective
  float q() const { return n > 0 ? float(w / n) : 0.f; }
};

struct Leaf {
  std::vector<Node*> path;  // root ... leaf
  go::Board board;
};

struct MoveStat {
  int move = 0;
  int visits = 0;
  float winrate = 0.5f;  // for the player to move at the root, in [0, 1]
  float score_lead = 0;  // for the player to move at the root
  float prior = 0;
  float lcb = 0;
  std::vector<int> pv;
};

class Search {
 public:
  Search(const Config& cfg, const go::Board& root, uint64_t seed = 1);

  const go::Board& root_board() const { return root_board_; }
  const Node& root() const { return *root_; }
  Config& config() { return cfg_; }

  void reset(const go::Board& root);
  // Enables/disables Dirichlet noise at the root (applied immediately if the root is expanded).
  void set_root_noise(bool on);
  // Moves the root to the child reached by `action`, reusing its subtree.
  void advance(int action);

  // Selects up to k leaves that need a network evaluation. Terminal leaves are
  // scored exactly and backed up immediately (they still count as visits).
  std::vector<Leaf> gather(int k);
  void apply(std::vector<Leaf>& leaves, const std::vector<nn::Evaluation>& evals);

  // Convenience: search synchronously with one evaluator until the root has `visits` visits.
  void run(nn::Evaluator& eval, int visits, int batch = 8);

  int root_visits() const { return root_->n; }
  float root_value() const { return root_->q(); }  // player to move at root, [-1, 1]
  float root_score() const { return root_->n ? float(root_->score_sum / root_->n) : 0.f; }

  std::array<float, go::NUM_ACTIONS> visit_policy() const;
  std::vector<MoveStat> move_stats(int max_pv = 10) const;  // sorted by visits (desc)
  int best_move() const;
  int sample_move(float temperature);

 private:
  Edge* select_child(Node* node, bool is_root);
  void backup(const std::vector<Node*>& path, float value, float score);
  void expand(Node* node, const go::Board& board, const nn::Evaluation& e);
  void apply_noise();

  Config cfg_;
  go::Board root_board_;
  std::unique_ptr<Node> root_;
  std::mt19937_64 rng_;
};

// Value/score of a finished game from the perspective of the player to move.
void terminal_eval(const go::Board& b, float* value, float* score);

}  // namespace mcts
