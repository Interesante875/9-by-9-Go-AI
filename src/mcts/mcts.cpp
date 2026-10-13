#include "mcts/mcts.h"

#include <algorithm>
#include <cmath>

namespace mcts {

using namespace go;

void terminal_eval(const Board& b, float* value, float* score) {
  float s = b.score();  // Black's perspective
  if (b.to_move() == WHITE) s = -s;
  *score = s;
  *value = s > 0 ? 1.f : (s < 0 ? -1.f : 0.f);
}

Search::Search(const Config& cfg, const Board& root, uint64_t seed) : cfg_(cfg), root_board_(root), rng_(seed) {
  root_ = std::make_unique<Node>();
}

void Search::reset(const Board& root) {
  root_board_ = root;
  root_ = std::make_unique<Node>();
}

void Search::set_root_noise(bool on) {
  cfg_.root_noise = on;
  if (!root_->expanded) return;
  if (on) apply_noise();
  else
    for (auto& e : root_->edges) e.prior = e.prior_raw;
}

void Search::advance(int action) {
  Board next = root_board_;
  if (!next.play(action)) {
    reset(root_board_);
    return;
  }
  std::unique_ptr<Node> child;
  for (auto& e : root_->edges)
    if (e.move == action) child = std::move(e.child);
  root_board_ = next;
  root_ = child ? std::move(child) : std::make_unique<Node>();
  // Pending/virtual-loss state never survives a completed search, but be safe.
  root_->pending = false;
  root_->vl = 0;
  if (root_->expanded && cfg_.root_noise) apply_noise();
  else if (root_->expanded)
    for (auto& e : root_->edges) e.prior = e.prior_raw;
}

void Search::apply_noise() {
  auto& edges = root_->edges;
  if (edges.empty()) return;
  const double alpha = std::max(0.03, double(cfg_.noise_alpha_total) / double(edges.size()));
  std::gamma_distribution<double> gamma(alpha, 1.0);
  std::vector<double> noise(edges.size());
  double sum = 0;
  for (auto& x : noise) sum += (x = gamma(rng_));
  if (sum <= 0) return;
  for (size_t i = 0; i < edges.size(); ++i)
    edges[i].prior = float((1 - cfg_.noise_eps) * edges[i].prior_raw + cfg_.noise_eps * noise[i] / sum);
}

Edge* Search::select_child(Node* node, bool is_root) {
  // Returns the best edge by PUCT score, creating its child node if needed.
  const double parent_n = double(node->n + node->vl);
  const double c = cfg_.c_puct_init + std::log((1.0 + parent_n + cfg_.c_puct_base) / cfg_.c_puct_base);
  const double sqrt_n = std::sqrt(std::max(1.0, parent_n));

  double visited_prior = 0;
  for (auto& e : node->edges)
    if (e.child && e.child->n + e.child->vl > 0) visited_prior += e.prior;
  const double parent_q = node->q();  // from this node's mover perspective
  const double fpu = parent_q - (is_root ? cfg_.root_fpu_reduction : cfg_.fpu_reduction) * std::sqrt(visited_prior);

  Edge* best = nullptr;
  double best_score = -1e30;
  for (auto& e : node->edges) {
    double q, n_child = 0;
    if (e.child && e.child->n + e.child->vl > 0) {
      const Node& ch = *e.child;
      n_child = ch.n + ch.vl;
      // child stores values for the opponent; virtual losses count as losses for us.
      q = (-ch.w - double(ch.vl)) / n_child;
    } else {
      q = fpu;
    }
    double u = c * e.prior * sqrt_n / (1.0 + n_child);
    double s = q + u;
    if (s > best_score) {
      best_score = s;
      best = &e;
    }
  }
  if (!best->child) best->child = std::make_unique<Node>();
  return best;
}

std::vector<Leaf> Search::gather(int k) {
  std::vector<Leaf> leaves;
  for (int i = 0; i < k; ++i) {
    Leaf leaf;
    leaf.board = root_board_;
    Node* node = root_.get();
    leaf.path.push_back(node);
    bool is_root = true;
    while (node->expanded && !node->terminal) {
      Edge* edge = select_child(node, is_root);
      leaf.board.play(edge->move);
      node = edge->child.get();
      leaf.path.push_back(node);
      is_root = false;
    }
    if (!node->expanded && !node->terminal && leaf.board.is_over()) node->terminal = true;
    if (node->terminal) {
      float v, s;
      terminal_eval(leaf.board, &v, &s);
      node->nn_value = v;
      node->nn_score = s;
      backup(leaf.path, v, s);
      continue;
    }
    if (node->pending) break;  // another leaf in this batch already waits on this node
    node->pending = true;
    for (Node* p : leaf.path) p->vl += cfg_.virtual_loss;
    leaves.push_back(std::move(leaf));
  }
  return leaves;
}

void Search::expand(Node* node, const Board& board, const nn::Evaluation& e) {
  uint8_t legal[NUM_ACTIONS];
  board.legal_mask(legal);
  node->edges.clear();
  double total = 0;
  for (int a = 0; a < NUM_ACTIONS; ++a)
    if (legal[a]) total += e.policy[a];
  for (int a = 0; a < NUM_ACTIONS; ++a) {
    if (!legal[a]) continue;
    Edge ed;
    ed.move = int16_t(a);
    ed.prior_raw = total > 0 ? float(e.policy[a] / total) : 0.f;
    ed.prior = ed.prior_raw;
    node->edges.push_back(std::move(ed));
  }
  if (total <= 0)
    for (auto& ed : node->edges) ed.prior = ed.prior_raw = 1.f / float(node->edges.size());
  std::sort(node->edges.begin(), node->edges.end(),
            [](const Edge& a, const Edge& b) { return a.prior_raw > b.prior_raw; });
  node->nn_value = e.value;
  node->nn_score = e.score;
  if (cfg_.keep_ownership) node->ownership = std::make_unique<std::array<float, NN>>(e.ownership);
  node->expanded = true;
  if (node == root_.get() && cfg_.root_noise) apply_noise();
}

void Search::backup(const std::vector<Node*>& path, float value, float score) {
  // value/score are from the perspective of the player to move at the leaf.
  float v = value, s = score;
  for (auto it = path.rbegin(); it != path.rend(); ++it) {
    Node* n = *it;
    n->n += 1;
    n->w += v;
    n->score_sum += s;
    v = -v;
    s = -s;
  }
}

void Search::apply(std::vector<Leaf>& leaves, const std::vector<nn::Evaluation>& evals) {
  for (size_t i = 0; i < leaves.size(); ++i) {
    Leaf& leaf = leaves[i];
    Node* node = leaf.path.back();
    for (Node* p : leaf.path) p->vl -= cfg_.virtual_loss;
    node->pending = false;
    if (!node->expanded) expand(node, leaf.board, evals[i]);
    backup(leaf.path, evals[i].value, evals[i].score);
  }
}

void Search::run(nn::Evaluator& eval, int visits, int batch) {
  int guard = 0;
  while (root_->n < visits) {
    auto leaves = gather(root_->n == 0 ? 1 : batch);
    if (leaves.empty()) {
      if (++guard > 100000) break;  // only terminal positions reachable
      continue;
    }
    std::vector<Snapshot> snaps;
    snaps.reserve(leaves.size());
    for (auto& l : leaves) snaps.push_back(l.board.snapshot());
    auto evals = eval.evaluate(snaps);
    apply(leaves, evals);
  }
}

int tt_safe_move(const Search& search, int chosen) {
  if (chosen != PASS) return chosen;
  const Board& b = search.root_board();
  float v, s;
  terminal_eval(b, &v, &s);  // score of the board as it stands, for the player to move
  if (s > 0) return PASS;
  const Color me = b.to_move();
  int best = PASS, best_n = 0;
  for (auto& e : search.root().edges) {
    if (e.move == PASS || !e.child || e.child->n <= best_n) continue;
    // Skip filling our own single-point eye.
    const int x = e.move % N, y = e.move / N;
    bool own_eye = true;
    const int dx[4] = {-1, 1, 0, 0}, dy[4] = {0, 0, -1, 1};
    for (int d = 0; d < 4; ++d) {
      int nx = x + dx[d], ny = y + dy[d];
      if (nx >= 0 && ny >= 0 && nx < N && ny < N && b.at(nx, ny) != me) own_eye = false;
    }
    if (own_eye) continue;
    best = e.move;
    best_n = e.child->n;
  }
  return best;
}

std::array<float, NUM_ACTIONS> Search::visit_policy() const {
  std::array<float, NUM_ACTIONS> pi{};
  double total = 0;
  for (auto& e : root_->edges)
    if (e.child) total += e.child->n;
  for (auto& e : root_->edges)
    if (e.child && total > 0) pi[e.move] = float(e.child->n / total);
  if (total == 0)
    for (auto& e : root_->edges) pi[e.move] = e.prior_raw;
  return pi;
}

std::vector<MoveStat> Search::move_stats(int max_pv) const {
  std::vector<MoveStat> out;
  for (auto& e : root_->edges) {
    MoveStat m;
    m.move = e.move;
    m.prior = e.prior_raw;
    if (e.child && e.child->n > 0) {
      const Node& ch = *e.child;
      m.visits = ch.n;
      double q = -ch.w / ch.n;  // root mover perspective
      m.winrate = float((q + 1) / 2);
      m.score_lead = float(-ch.score_sum / ch.n);
      // Crude lower confidence bound on the winrate.
      m.lcb = float(std::max(0.0, (q + 1) / 2 - 0.5 / std::sqrt(double(ch.n))));
      // Principal variation.
      m.pv.push_back(e.move);
      const Node* cur = &ch;
      while (cur->expanded && int(m.pv.size()) < max_pv) {
        const Edge* best = nullptr;
        int bn = 0;
        for (auto& ce : cur->edges)
          if (ce.child && ce.child->n > bn) {
            bn = ce.child->n;
            best = &ce;
          }
        if (!best) break;
        m.pv.push_back(best->move);
        cur = best->child.get();
      }
    }
    out.push_back(std::move(m));
  }
  std::stable_sort(out.begin(), out.end(), [](const MoveStat& a, const MoveStat& b) {
    if (a.visits != b.visits) return a.visits > b.visits;
    return a.prior > b.prior;
  });
  return out;
}

int Search::best_move() const {
  int best = PASS;
  int bn = -1;
  double bq = -2;
  for (auto& e : root_->edges) {
    int n = e.child ? e.child->n : 0;
    double q = (e.child && e.child->n) ? -e.child->w / e.child->n : -2;
    if (n > bn || (n == bn && q > bq)) {
      bn = n;
      bq = q;
      best = e.move;
    }
  }
  return best;
}

int Search::sample_move(float temperature) {
  if (temperature <= 1e-3f) return best_move();
  std::vector<double> w;
  std::vector<int> moves;
  for (auto& e : root_->edges) {
    int n = e.child ? e.child->n : 0;
    if (n == 0) continue;
    w.push_back(std::pow(double(n), 1.0 / temperature));
    moves.push_back(e.move);
  }
  if (moves.empty()) return best_move();
  std::discrete_distribution<int> dist(w.begin(), w.end());
  return moves[dist(rng_)];
}

}  // namespace mcts
