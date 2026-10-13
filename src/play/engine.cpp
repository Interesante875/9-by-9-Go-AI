#include "play/engine.h"

#include <cmath>
#include <limits>

namespace play {

using namespace go;

PlayEngine::PlayEngine(std::shared_ptr<nn::Evaluator> eval, const std::string& model_path, float komi)
    : eval_(std::move(eval)), model_path_(model_path), env_(komi, KoRule::PositionalSuperko, 0) {
  mcfg_.keep_ownership = true;
  search_ = std::make_unique<mcts::Search>(mcfg_, env_.board(), 42);
  sync_history();
}

void PlayEngine::sync_history() {
  const size_t n = env_.moves().size() + 1;
  winrate_hist_.resize(n, std::numeric_limits<float>::quiet_NaN());
  score_hist_.resize(n, std::numeric_limits<float>::quiet_NaN());
  record_quick_eval();
}

void PlayEngine::record_quick_eval() {
  const size_t i = env_.moves().size();
  if (!std::isnan(winrate_hist_[i])) return;
  const Board& b = env_.board();
  float v, s;
  if (b.is_over()) {
    mcts::terminal_eval(b, &v, &s);
  } else {
    auto e = eval_->evaluate({b.snapshot()})[0];
    v = e.value;
    s = e.score;
  }
  const float sign = b.to_move() == BLACK ? 1.f : -1.f;
  winrate_hist_[i] = (sign * v + 1.f) / 2.f;
  score_hist_[i] = sign * s;
}

void PlayEngine::new_game(float komi) {
  env_ = GoEnv(komi, KoRule::PositionalSuperko, 0);
  env_.reset();
  search_->reset(env_.board());
  winrate_hist_.clear();
  score_hist_.clear();
  sync_history();
}

bool PlayEngine::play(int action) {
  if (env_.done()) return false;
  auto r = env_.step(action);
  if (r.illegal) return false;
  search_->advance(action);
  if (search_->root_board().hash() != env_.board().hash() ||
      search_->root_board().move_number() != env_.board().move_number())
    search_->reset(env_.board());
  sync_history();
  return true;
}

bool PlayEngine::undo() {
  if (env_.resigned() != EMPTY) {
    // Undoing a resignation just clears it.
    std::vector<int> moves = env_.moves();
    float komi = env_.komi();
    env_ = GoEnv(komi, KoRule::PositionalSuperko, 0);
    env_.reset();
    for (int a : moves) env_.step(a);
  } else if (!env_.undo()) {
    return false;
  }
  search_->reset(env_.board());
  winrate_hist_.resize(env_.moves().size() + 1);
  score_hist_.resize(env_.moves().size() + 1);
  sync_history();
  return true;
}

void PlayEngine::resign(Color who) { env_.resign(who); }

bool PlayEngine::load_sgf(const std::string& sgf, std::string* err) {
  GoEnv e(env_.komi(), KoRule::PositionalSuperko, 0);
  if (!e.load_sgf(sgf, err)) return false;
  env_ = e;
  search_->reset(env_.board());
  winrate_hist_.clear();
  score_hist_.clear();
  winrate_hist_.resize(env_.moves().size() + 1, std::numeric_limits<float>::quiet_NaN());
  score_hist_.resize(env_.moves().size() + 1, std::numeric_limits<float>::quiet_NaN());
  sync_history();
  return true;
}

Analysis PlayEngine::make_analysis() {
  Analysis a;
  const Board& b = env_.board();
  a.to_move = b.to_move();
  const float sign = a.to_move == BLACK ? 1.f : -1.f;
  a.visits = search_->root_visits();
  a.moves = search_->move_stats(12);
  const auto& root = search_->root();
  if (a.visits > 0) {
    a.winrate_black = (sign * search_->root_value() + 1.f) / 2.f;
    a.score_black = sign * search_->root_score();
  }
  a.nn_value_black = sign * root.nn_value;
  a.policy.fill(0.f);
  for (auto& e : root.edges) a.policy[e.move] = e.prior_raw;
  if (root.ownership)
    for (int i = 0; i < NN; ++i) a.ownership_black[i] = sign * (*root.ownership)[i];
  // Refine the win-rate graph with the searched value.
  const size_t i = env_.moves().size();
  if (a.visits > 0 && i < winrate_hist_.size()) {
    winrate_hist_[i] = a.winrate_black;
    score_hist_[i] = a.score_black;
  }
  return a;
}

Analysis PlayEngine::analyze(int visits, int max_total) {
  if (!env_.done()) {
    search_->set_root_noise(false);
    int target = std::min(max_total, search_->root_visits() + std::max(1, visits));
    search_->run(*eval_, target, 16);
  }
  return make_analysis();
}

int PlayEngine::choose_move(int visits, bool allow_resign, Analysis* out) {
  search_->set_root_noise(false);
  search_->run(*eval_, std::max(1, visits), 16);
  Analysis a = make_analysis();
  int move = search_->best_move();
  const float my_wr = a.to_move == BLACK ? a.winrate_black : 1.f - a.winrate_black;
  if (allow_resign && env_.board().move_number() > 30 && a.visits >= 50 && my_wr < 0.03f) move = -2;
  if (out) *out = a;
  return move;
}

bool PlayEngine::reload(const std::string& path, std::string* err) {
  try {
    auto net = nn::load_model(path, eval_->device());
    eval_->set_net(net);
    model_path_ = path;
    search_->reset(env_.board());
    std::fill(winrate_hist_.begin(), winrate_hist_.end(), std::numeric_limits<float>::quiet_NaN());
    std::fill(score_hist_.begin(), score_hist_.end(), std::numeric_limits<float>::quiet_NaN());
    record_quick_eval();
    return true;
  } catch (const std::exception& e) {
    if (err) *err = e.what();
    return false;
  }
}

}  // namespace play
