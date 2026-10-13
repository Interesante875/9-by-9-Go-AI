#include <cmath>

#include "mcts/mcts.h"
#include "test_framework.h"

using namespace go;

namespace {

// Runs the search with a "know-nothing" evaluator: uniform policy, value 0.
// Only exactly-scored terminal positions carry information.
void run_uniform(mcts::Search& s, int visits, int batch = 4) {
  int guard = 0;
  while (s.root_visits() < visits && guard++ < 100000) {
    auto leaves = s.gather(s.root_visits() == 0 ? 1 : batch);
    std::vector<nn::Evaluation> evs(leaves.size());
    for (size_t i = 0; i < leaves.size(); ++i) {
      uint8_t legal[NUM_ACTIONS];
      leaves[i].board.legal_mask(legal);
      int n = 0;
      for (int a = 0; a < NUM_ACTIONS; ++a) n += legal[a];
      for (int a = 0; a < NUM_ACTIONS; ++a) evs[i].policy[a] = legal[a] ? 1.f / n : 0.f;
      evs[i].value = 0.f;
      evs[i].score = 0.f;
      evs[i].ownership.fill(0.f);
    }
    s.apply(leaves, evs);
  }
}

}  // namespace

TEST(mcts_visit_accounting) {
  Board b;
  mcts::Config cfg;
  mcts::Search s(cfg, b, 1);
  run_uniform(s, 300);
  CHECK(s.root_visits() >= 300);
  int child_sum = 0;
  for (auto& e : s.root().edges)
    if (e.child) child_sum += e.child->n;
  CHECK_EQ(child_sum, s.root_visits() - 1);  // root's own evaluation is the extra visit
  auto pi = s.visit_policy();
  double total = 0;
  for (float p : pi) total += p;
  CHECK_NEAR(total, 1.0, 1e-5);
  CHECK_EQ(s.root().vl, 0);
}

TEST(mcts_passes_to_win_when_ahead) {
  // Black has one stone (owns the board), White passed. Passing now ends the game with a Black win.
  Board b(7.0f);
  b.play(gtp_to_action("E5"));
  b.play(PASS);
  mcts::Config cfg;
  mcts::Search s(cfg, b, 3);
  run_uniform(s, 400);
  CHECK_EQ(s.best_move(), PASS);
  CHECK(s.root_value() > 0.5f);
  auto stats = s.move_stats();
  CHECK_EQ(stats[0].move, PASS);
  CHECK(stats[0].winrate > 0.99f);
  CHECK_NEAR(stats[0].score_lead, 74.0, 1e-3);
}

TEST(mcts_does_not_pass_when_behind) {
  // Same, but komi is so large that passing loses for Black.
  Board b(100.0f);
  b.play(gtp_to_action("E5"));
  b.play(PASS);
  mcts::Config cfg;
  mcts::Search s(cfg, b, 5);
  run_uniform(s, 400);
  CHECK(s.best_move() != PASS);
  for (auto& m : s.move_stats())
    if (m.move == PASS) CHECK(m.winrate < 0.01f);
}

TEST(mcts_tree_reuse) {
  Board b;
  mcts::Config cfg;
  mcts::Search s(cfg, b, 7);
  run_uniform(s, 200);
  int move = s.best_move();
  int child_visits = 0;
  for (auto& e : s.root().edges)
    if (e.move == move) child_visits = e.child->n;
  s.advance(move);
  CHECK_EQ(s.root_visits(), child_visits);
  CHECK_EQ(s.root_board().move_number(), 1);
  CHECK_EQ(s.root_board().to_move(), WHITE);
  run_uniform(s, child_visits + 50);
  CHECK(s.root_visits() >= child_visits + 50);
}

TEST(mcts_root_noise_changes_priors_only_at_root) {
  Board b;
  mcts::Config cfg;
  cfg.root_noise = true;
  mcts::Search s(cfg, b, 9);
  run_uniform(s, 10);
  double diff = 0, sum = 0;
  for (auto& e : s.root().edges) {
    diff += std::fabs(e.prior - e.prior_raw);
    sum += e.prior;
  }
  CHECK(diff > 1e-3);
  CHECK_NEAR(sum, 1.0, 1e-4);
  s.set_root_noise(false);
  for (auto& e : s.root().edges) CHECK_EQ(e.prior, e.prior_raw);
}

TEST(mcts_finished_game_root) {
  Board b;
  b.play(PASS);
  b.play(PASS);
  mcts::Config cfg;
  mcts::Search s(cfg, b, 1);
  run_uniform(s, 5);
  CHECK(s.root().terminal);
  // Black is to move and lost by komi: value -1 from Black's point of view.
  CHECK_NEAR(s.root_value(), -1.0, 1e-6);
  CHECK_NEAR(s.root_score(), -7.0, 1e-6);
}
