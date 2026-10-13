#include "train/arena.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <random>

#include "go/env.h"
#include "mcts/mcts.h"

namespace train {

using namespace go;

double elo_from_score(double s) {
  s = std::min(0.99, std::max(0.01, s));
  return 400.0 * std::log10(s / (1.0 - s));
}

namespace {

bool is_own_eye(const Board& b, int action, Color c) {
  int x = action % N, y = action / N;
  const int dx[4] = {-1, 1, 0, 0}, dy[4] = {0, 0, -1, 1};
  for (int d = 0; d < 4; ++d) {
    int nx = x + dx[d], ny = y + dy[d];
    if (nx < 0 || ny < 0 || nx >= N || ny >= N) continue;
    if (b.at(nx, ny) != c) return false;
  }
  return true;
}

int random_move(const Board& b, std::mt19937_64& rng) {
  std::vector<int> cand;
  for (int a : b.legal_actions())
    if (a != PASS && !is_own_eye(b, a, b.to_move())) cand.push_back(a);
  if (cand.empty()) return PASS;
  return cand[rng() % cand.size()];
}

struct Slot {
  bool active = false;
  int game = 0;
  Board board;
  std::unique_ptr<mcts::Search> search[2];  // [0] = player A, [1] = player B
  std::vector<int> moves;
};

}  // namespace

MatchResult play_match(std::shared_ptr<nn::Evaluator> a, std::shared_ptr<nn::Evaluator> b, const MatchConfig& cfg,
                       const std::atomic<bool>* stop, const std::string& name_a, const std::string& name_b) {
  MatchResult res;
  std::mt19937_64 rng(cfg.seed);
  std::shared_ptr<nn::Evaluator> evs[2] = {a, b};
  mcts::Config mc;
  int next_game = 0;
  double total_len = 0;
  std::vector<Slot> slots(std::max(1, std::min(cfg.parallel_games, cfg.games)));

  auto start = [&](Slot& s) {
    s.active = true;
    s.game = next_game++;
    s.board = Board(cfg.komi, KoRule::PositionalSuperko, cfg.max_moves);
    for (int p = 0; p < 2; ++p) s.search[p] = std::make_unique<mcts::Search>(mc, s.board, rng());
    s.moves.clear();
  };
  // Player index (0 = A, 1 = B) to move in slot s. A is Black in even games.
  auto player_to_move = [](const Slot& s) {
    bool a_black = s.game % 2 == 0;
    bool black_to_move = s.board.to_move() == BLACK;
    return (a_black == black_to_move) ? 0 : 1;
  };
  for (auto& s : slots)
    if (next_game < cfg.games) start(s);

  while (true) {
    if (stop && stop->load()) break;
    bool any = false;
    // Gather leaves per evaluator.
    std::vector<mcts::Leaf> leaves[2];
    std::vector<std::pair<Slot*, size_t>> owners[2];
    std::vector<Slot*> ready;
    for (auto& s : slots) {
      if (!s.active) continue;
      any = true;
      int p = player_to_move(s);
      if (!evs[p] || cfg.sims <= 0) {
        ready.push_back(&s);
        continue;
      }
      auto& search = *s.search[p];
      if (search.root_visits() >= cfg.sims) {
        ready.push_back(&s);
        continue;
      }
      auto l = search.gather(search.root_visits() == 0 ? 1 : 4);
      owners[p].emplace_back(&s, l.size());
      for (auto& x : l) leaves[p].push_back(std::move(x));
    }
    if (!any) break;
    for (int p = 0; p < 2; ++p) {
      if (leaves[p].empty()) continue;
      std::vector<Snapshot> snaps;
      for (auto& l : leaves[p]) snaps.push_back(l.board.snapshot());
      auto evals = evs[p]->evaluate(snaps);
      size_t off = 0;
      for (auto& [s, cnt] : owners[p]) {
        if (!cnt) continue;
        std::vector<mcts::Leaf> mine(std::make_move_iterator(leaves[p].begin() + off),
                                     std::make_move_iterator(leaves[p].begin() + off + cnt));
        std::vector<nn::Evaluation> ev(evals.begin() + off, evals.begin() + off + cnt);
        s->search[player_to_move(*s)]->apply(mine, ev);
        off += cnt;
      }
    }
    // Raw-policy players (sims == 0) are evaluated here in one batch per evaluator.
    for (Slot* s : ready) {
      int p = player_to_move(*s);
      int action;
      if (!evs[p]) {
        action = random_move(s->board, rng);
      } else if (cfg.sims <= 0) {
        auto e = evs[p]->evaluate({s->board.snapshot()})[0];
        action = int(std::max_element(e.policy.begin(), e.policy.end()) - e.policy.begin());
      } else {
        float t = int(s->moves.size()) < cfg.temp_moves ? cfg.temperature : 0.f;
        action = s->search[p]->sample_move(t);
      }
      if (!s->board.play(action)) {
        action = PASS;
        s->board.play(PASS);
      }
      s->moves.push_back(action);
      for (int q = 0; q < 2; ++q) s->search[q]->advance(action);
      if (s->board.is_over()) {
        Color w = s->board.winner();
        bool a_black = s->game % 2 == 0;
        Color a_color = a_black ? BLACK : WHITE;
        if (w == EMPTY) res.draws++;
        else if (w == a_color) {
          res.a_wins++;
          (a_black ? res.a_black_wins : res.a_white_wins)++;
        } else
          res.b_wins++;
        total_len += double(s->moves.size());
        res.sgfs.push_back(make_sgf(s->moves, cfg.komi, score_to_result(s->board.score()),
                                    a_black ? name_a : name_b, a_black ? name_b : name_a));
        s->active = false;
        if (next_game < cfg.games) start(*s);
      }
    }
  }
  res.avg_length = res.games() ? total_len / res.games() : 0;
  return res;
}

}  // namespace train
