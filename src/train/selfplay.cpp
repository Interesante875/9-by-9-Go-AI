#include "train/selfplay.h"

#include <algorithm>
#include <chrono>
#include <iterator>
#include <cmath>
#include <mutex>
#include <random>
#include <thread>

#include "go/env.h"
#include "mcts/mcts.h"

namespace train {

using namespace go;

SelfPlay::SelfPlay(const TrainConfig& cfg, std::shared_ptr<nn::Evaluator> eval) : cfg_(cfg), eval_(std::move(eval)) {}

void finalize_game(const std::vector<MoveRecord>& recs, const Board& final_board, Color winner, bool resigned,
                   float gae_lambda, GameResult* out) {
  int8_t own_black[NN];
  final_board.area_ownership(own_black);
  const float score_black = resigned ? 0.f : final_board.score();
  const int T = int(recs.size());

  // GAE over the alternating-perspective trajectory (gamma = 1).
  // delta_t = r_t - V_{t+1} - V_t   (V_{t+1} is from the opponent's point of view)
  // A_t     = delta_t - lambda * A_{t+1}
  std::vector<float> adv(T, 0.f);
  float next_adv = 0.f;
  for (int t = T - 1; t >= 0; --t) {
    const Color mover = Color(recs[t].snap.to_move);
    float r = 0.f;
    if (t == T - 1) r = winner == EMPTY ? 0.f : (winner == mover ? 1.f : -1.f);
    const float v_next = (t + 1 < T) ? recs[t + 1].nn_value : 0.f;
    const float delta = r - v_next - recs[t].nn_value;
    adv[t] = delta - gae_lambda * next_adv;
    next_adv = adv[t];
  }

  for (int t = 0; t < T; ++t) {
    const MoveRecord& r = recs[t];
    if (!r.train) continue;
    const Color mover = Color(r.snap.to_move);
    const float sign = mover == BLACK ? 1.f : -1.f;
    Sample s{};
    s.pos = r.snap;
    for (int a = 0; a < NUM_ACTIONS; ++a) s.policy[a] = r.pi[a];
    for (int a = 0; a < NN; ++a) s.ownership[a] = int8_t(own_black[a] * int(sign));
    s.z = winner == EMPTY ? 0.f : (winner == mover ? 1.f : -1.f);
    s.score = score_black * sign;
    s.root_q = r.root_q;
    s.old_logp = r.nn_logp;
    s.old_value = r.nn_value;
    s.advantage = adv[t];
    s.policy_weight = 1.f;
    s.aux_weight = resigned ? 0.f : 1.f;
    s.action = int16_t(r.action);
    out->samples.push_back(s);
  }
}

std::vector<GameResult> SelfPlay::play(int n, uint64_t seed, const std::atomic<bool>* stop, SelfPlayStats* stats) {
  auto t0 = std::chrono::steady_clock::now();
  const int64_t evals0 = eval_->evaluations();
  std::vector<GameResult> results;
  std::mutex mu;
  std::atomic<int> started{0};
  const int threads = std::max(1, cfg_.selfplay_threads);
  std::vector<std::thread> pool;
  for (int t = 0; t < threads; ++t) {
    pool.emplace_back([&, t] {
      if (cfg_.algo == "ppo") worker_policy(t, n, seed + 7919ULL * t, stop, &started, &results, &mu);
      else worker_mcts(t, n, seed + 7919ULL * t, stop, &started, &results, &mu);
    });
  }
  for (auto& th : pool) th.join();

  if (stats) {
    *stats = SelfPlayStats{};
    for (auto& g : results) {
      stats->games++;
      stats->moves += int(g.moves.size());
      stats->samples += int(g.samples.size());
      stats->black_wins += g.winner == BLACK;
      stats->resigned += g.resigned;
      if (!g.resign_allowed) {
        stats->noresign_games++;
        stats->false_resigns += g.false_resign;
      }
    }
    stats->seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    stats->nn_evals = eval_->evaluations() - evals0;
  }
  return results;
}

namespace {

struct McSlot {
  bool active = false;
  Board board;
  std::unique_ptr<mcts::Search> search;
  int target = 0;
  bool full = false;
  bool allow_resign = true;
  bool would_resign[3] = {false, false, false};
  int below[3] = {0, 0, 0};  // consecutive moves with root value under the resign threshold
  std::vector<MoveRecord> recs;
  std::vector<int> moves;
};

}  // namespace

void SelfPlay::worker_mcts(int thread_id, int n, uint64_t seed, const std::atomic<bool>* stop,
                           std::atomic<int>* started, std::vector<GameResult>* out, std::mutex* out_mu) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> uni(0.f, 1.f);
  mcts::Config mc;
  mc.c_puct_init = cfg_.c_puct;
  mc.fpu_reduction = cfg_.fpu_reduction;
  mc.noise_eps = cfg_.noise_eps;
  mc.noise_alpha_total = cfg_.noise_alpha_total;

  std::vector<McSlot> slots(std::max(1, cfg_.parallel_games));

  auto begin_search = [&](McSlot& s) {
    // Playout cap randomisation: only full searches (with noise) produce policy targets.
    s.full = uni(rng) < cfg_.full_prob;
    s.target = s.full ? cfg_.full_sims : cfg_.fast_sims;
    s.search->set_root_noise(s.full);
  };
  auto start_game = [&](McSlot& s) {
    s.active = true;
    s.board = Board(cfg_.komi, KoRule::PositionalSuperko, cfg_.max_moves);
    s.search = std::make_unique<mcts::Search>(mc, s.board, rng());
    s.allow_resign = resign_enabled_ && cfg_.resign_threshold != 0.f && uni(rng) >= cfg_.noresign_frac;
    s.would_resign[BLACK] = s.would_resign[WHITE] = false;
    s.below[BLACK] = s.below[WHITE] = 0;
    s.recs.clear();
    s.moves.clear();
    begin_search(s);
  };
  auto finish_game = [&](McSlot& s, Color winner, bool resigned) {
    GameResult g;
    g.moves = s.moves;
    g.winner = winner;
    g.resigned = resigned;
    g.resign_allowed = s.allow_resign;
    g.false_resign = !s.allow_resign && winner != EMPTY && s.would_resign[winner];
    g.score = resigned ? 0.f : s.board.score();
    finalize_game(s.recs, s.board, winner, resigned, cfg_.gae_lambda, &g);
    std::string result = resigned ? (winner == BLACK ? "B+R" : "W+R")
                                  : (g.score == 0 ? "0" : score_to_result(g.score));
    g.sgf = make_sgf(s.moves, cfg_.komi, result, "go9 self-play", "go9 self-play");
    {
      std::lock_guard<std::mutex> lock(*out_mu);
      out->push_back(std::move(g));
    }
    s.active = false;
  };

  for (auto& s : slots)
    if (started->fetch_add(1) < n) start_game(s);
    else break;

  const int k = std::max(1, cfg_.leaves_per_search);
  std::vector<mcts::Leaf> all_leaves;
  std::vector<std::pair<size_t, size_t>> owner;  // (slot, count)
  while (true) {
    if (stop && stop->load()) break;
    bool any = false;
    all_leaves.clear();
    owner.clear();
    for (size_t i = 0; i < slots.size(); ++i) {
      McSlot& s = slots[i];
      if (!s.active) continue;
      any = true;
      auto leaves = s.search->gather(s.search->root_visits() == 0 ? 1 : k);
      owner.emplace_back(i, leaves.size());
      for (auto& l : leaves) all_leaves.push_back(std::move(l));
    }
    if (!any) break;
    if (!all_leaves.empty()) {
      std::vector<Snapshot> snaps;
      snaps.reserve(all_leaves.size());
      for (auto& l : all_leaves) snaps.push_back(l.board.snapshot());
      auto evals = eval_->evaluate(snaps);
      size_t off = 0;
      for (auto& [slot, cnt] : owner) {
        if (cnt == 0) continue;
        std::vector<mcts::Leaf> mine(std::make_move_iterator(all_leaves.begin() + off),
                                     std::make_move_iterator(all_leaves.begin() + off + cnt));
        std::vector<nn::Evaluation> ev(evals.begin() + off, evals.begin() + off + cnt);
        slots[slot].search->apply(mine, ev);
        off += cnt;
      }
    }
    // Make moves for slots whose search is complete.
    for (auto& s : slots) {
      if (!s.active || s.search->root_visits() < s.target) continue;
      const mcts::Node& root = s.search->root();
      const Color mover = s.board.to_move();
      const float rv = s.search->root_value();
      s.below[mover] = rv < cfg_.resign_threshold ? s.below[mover] + 1 : 0;
      if (resign_enabled_ && cfg_.resign_threshold != 0.f && s.board.move_number() >= cfg_.resign_min_moves &&
          s.below[mover] >= std::max(1, cfg_.resign_consecutive)) {
        s.would_resign[mover] = true;
        if (s.allow_resign) {
          finish_game(s, opponent(mover), true);
          if (started->fetch_add(1) < n) start_game(s);
          continue;
        }
      }
      MoveRecord r;
      r.snap = s.board.snapshot();
      r.pi = s.search->visit_policy();
      r.root_q = rv;
      r.nn_value = root.nn_value;
      const float temp = s.board.move_number() < cfg_.temp_moves ? cfg_.temperature : 0.f;
      // Area-scoring cleanup: no passing while the current count loses (dead stones must be captured).
      int action = mcts::tt_safe_move(*s.search, s.search->sample_move(temp));
      r.action = action;
      float prior = 1e-6f;
      for (auto& e : root.edges)
        if (e.move == action) prior = std::max(prior, e.prior_raw);
      r.nn_logp = std::log(prior);
      r.train = s.full;
      s.recs.push_back(r);
      s.board.play(action);
      s.moves.push_back(action);
      s.search->advance(action);
      if (s.board.is_over()) {
        finish_game(s, s.board.winner(), false);
        if (started->fetch_add(1) < n) start_game(s);
        continue;
      }
      begin_search(s);
    }
  }
  (void)thread_id;
}

void SelfPlay::worker_policy(int thread_id, int n, uint64_t seed, const std::atomic<bool>* stop,
                             std::atomic<int>* started, std::vector<GameResult>* out, std::mutex* out_mu) {
  // Pure PPO rollouts: sample every move from the raw network policy.
  std::mt19937_64 rng(seed);
  struct Slot {
    bool active = false;
    Board board;
    std::vector<MoveRecord> recs;
    std::vector<int> moves;
  };
  std::vector<Slot> slots(std::max(1, cfg_.parallel_games));
  auto start_game = [&](Slot& s) {
    s.active = true;
    s.board = Board(cfg_.komi, KoRule::PositionalSuperko, cfg_.max_moves);
    s.recs.clear();
    s.moves.clear();
  };
  for (auto& s : slots)
    if (started->fetch_add(1) < n) start_game(s);
    else break;

  while (true) {
    if (stop && stop->load()) break;
    std::vector<Snapshot> snaps;
    std::vector<size_t> idx;
    for (size_t i = 0; i < slots.size(); ++i)
      if (slots[i].active) {
        snaps.push_back(slots[i].board.snapshot());
        idx.push_back(i);
      }
    if (idx.empty()) break;
    auto evals = eval_->evaluate(snaps);
    for (size_t j = 0; j < idx.size(); ++j) {
      Slot& s = slots[idx[j]];
      const auto& e = evals[j];
      std::discrete_distribution<int> dist(e.policy.begin(), e.policy.end());
      int action = dist(rng);
      if (!snaps[j].legal[action]) action = PASS;
      MoveRecord r;
      r.snap = snaps[j];
      r.pi = e.policy;
      r.root_q = e.value;
      r.nn_value = e.value;
      r.nn_logp = std::log(std::max(1e-6f, e.policy[action]));
      r.action = action;
      r.train = true;
      s.recs.push_back(r);
      s.board.play(action);
      s.moves.push_back(action);
      if (s.board.is_over()) {
        GameResult g;
        g.moves = s.moves;
        g.winner = s.board.winner();
        g.score = s.board.score();
        g.resign_allowed = false;
        finalize_game(s.recs, s.board, g.winner, false, cfg_.gae_lambda, &g);
        for (auto& smp : g.samples) smp.policy_weight = 0.f;  // no search targets in pure PPO
        g.sgf = make_sgf(s.moves, cfg_.komi, g.score == 0 ? "0" : score_to_result(g.score), "go9 ppo", "go9 ppo");
        {
          std::lock_guard<std::mutex> lock(*out_mu);
          out->push_back(std::move(g));
        }
        s.active = false;
        if (started->fetch_add(1) < n) start_game(s);
      }
    }
  }
  (void)thread_id;
}

}  // namespace train
