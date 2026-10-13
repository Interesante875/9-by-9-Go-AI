#include <cmath>
#include <filesystem>

#include "nn/evaluator.h"
#include "test_framework.h"
#include "train/arena.h"
#include "train/selfplay.h"
#include "train/trainer.h"

using namespace go;
namespace fs = std::filesystem;

namespace {

nn::NetConfig tiny() {
  nn::NetConfig c;
  c.blocks = 3;  // includes one global-pooling block
  c.channels = 16;
  c.head_channels = 8;
  return c;
}

train::TrainConfig tiny_train(const std::string& dir) {
  train::TrainConfig c;
  c.run_dir = dir;
  c.device = "cpu";
  c.blocks = 3;
  c.channels = 16;
  c.games_per_iter = 4;
  c.parallel_games = 4;
  c.selfplay_threads = 1;
  c.full_sims = 16;
  c.fast_sims = 8;
  c.full_prob = 0.5f;
  c.max_moves = 60;
  c.batch_size = 32;
  c.min_replay = 16;
  c.replay_capacity = 2000;
  c.eval_every = 0;
  c.snapshot_every = 0;
  c.lr_decay_iters = 0;
  return c;
}

fs::path temp_dir(const std::string& name) {
  fs::path p = fs::temp_directory_path() / name;
  fs::remove_all(p);
  fs::create_directories(p);
  return p;
}

}  // namespace

TEST(net_forward_shapes) {
  nn::GoNet net(tiny());
  net->eval();
  auto x = torch::randn({3, nn::kInputPlanes, N, N});
  auto out = net->forward(x);
  CHECK_EQ(out.policy_logits.size(0), 3);
  CHECK_EQ(out.policy_logits.size(1), NUM_ACTIONS);
  CHECK_EQ(out.value.dim(), 1);
  CHECK_EQ(out.value.size(0), 3);
  CHECK_EQ(out.score.size(0), 3);
  CHECK_EQ(out.ownership.size(1), NN);
  CHECK(out.value.abs().max().item<float>() <= 1.f);
}

TEST(model_save_load_roundtrip) {
  auto dir = temp_dir("go9_test_model");
  nn::GoNet net(tiny());
  net->eval();
  auto path = (dir / "m.pt").string();
  nn::save_model(net, path);
  auto cfg = nn::read_model_config(path);
  CHECK_EQ(cfg.blocks, 3);
  CHECK_EQ(cfg.channels, 16);
  auto net2 = nn::load_model(path, torch::kCPU);
  net2->eval();
  auto x = torch::randn({2, nn::kInputPlanes, N, N});
  torch::NoGradGuard ng;
  auto a = net->forward(x), b = net2->forward(x);
  CHECK(torch::allclose(a.policy_logits, b.policy_logits));
  CHECK(torch::allclose(a.value, b.value));
  CHECK(torch::allclose(a.ownership, b.ownership));
  fs::remove_all(dir);
}

TEST(evaluator_masks_illegal_moves) {
  auto ev = std::make_shared<nn::Evaluator>(nn::GoNet(tiny()), torch::Device(torch::kCPU));
  Board b;
  b.play(gtp_to_action("E5"));
  b.play(gtp_to_action("D5"));
  for (int sym = 0; sym < 8; ++sym) {
    auto e = ev->evaluate({b.snapshot()}, sym)[0];
    double total = 0;
    for (float p : e.policy) total += p;
    CHECK_NEAR(total, 1.0, 1e-4);
    CHECK_EQ(e.policy[gtp_to_action("E5")], 0.f);
    CHECK_EQ(e.policy[gtp_to_action("D5")], 0.f);
    CHECK(e.value >= -1.f && e.value <= 1.f);
  }
}

TEST(selfplay_produces_valid_samples) {
  auto cfg = tiny_train("unused");
  auto ev = std::make_shared<nn::Evaluator>(nn::GoNet(tiny()), torch::Device(torch::kCPU));
  train::SelfPlay sp(cfg, ev);
  train::SelfPlayStats st;
  auto games = sp.play(4, 123, nullptr, &st);
  CHECK_EQ(int(games.size()), 4);
  CHECK_EQ(st.games, 4);
  for (auto& g : games) {
    CHECK(!g.moves.empty());
    CHECK(g.winner == BLACK || g.winner == WHITE || g.winner == EMPTY);
    for (auto& s : g.samples) {
      double total = 0;
      for (int a = 0; a < NUM_ACTIONS; ++a) {
        total += s.policy[a];
        if (s.policy[a] > 0) CHECK(s.pos.legal[a]);  // never targets illegal moves
      }
      CHECK_NEAR(total, 1.0, 1e-4);
      CHECK(s.z == 1.f || s.z == -1.f || s.z == 0.f);
      Color mover = Color(s.pos.to_move);
      if (g.winner != EMPTY) CHECK_EQ(s.z, mover == g.winner ? 1.f : -1.f);
      for (int a = 0; a < NN; ++a) CHECK(s.ownership[a] >= -1 && s.ownership[a] <= 1);
      CHECK(s.pos.legal[s.action]);
      CHECK(std::isfinite(s.advantage) && std::isfinite(s.old_logp));
    }
  }
}

TEST(ppo_rollouts) {
  auto cfg = tiny_train("unused");
  cfg.algo = "ppo";
  auto ev = std::make_shared<nn::Evaluator>(nn::GoNet(tiny()), torch::Device(torch::kCPU));
  train::SelfPlay sp(cfg, ev);
  train::SelfPlayStats st;
  auto games = sp.play(3, 5, nullptr, &st);
  CHECK_EQ(int(games.size()), 3);
  for (auto& g : games) {
    CHECK_EQ(g.samples.size(), g.moves.size());  // PPO keeps every move
    for (auto& s : g.samples) CHECK_EQ(s.policy_weight, 0.f);
  }
}

TEST(gae_advantages_match_monte_carlo_when_lambda_is_one) {
  // With lambda = 1, A_t = z_t - V_t.
  std::vector<train::MoveRecord> recs(5);
  Board b;
  for (int t = 0; t < 5; ++t) {
    recs[t].snap = b.snapshot();
    recs[t].nn_value = 0.1f * t - 0.2f;
    recs[t].pi.fill(0.f);
    recs[t].pi[PASS] = 1.f;
    recs[t].action = gtp_to_action("A" + std::to_string(t + 1));
    b.play(recs[t].action);
  }
  train::GameResult g;
  train::finalize_game(recs, b, BLACK, true, 1.0f, &g);
  CHECK_EQ(int(g.samples.size()), 5);
  for (int t = 0; t < 5; ++t) {
    float z = (t % 2 == 0) ? 1.f : -1.f;  // Black moves on even t and won
    CHECK_EQ(g.samples[t].z, z);
    CHECK_NEAR(g.samples[t].advantage, z - recs[t].nn_value, 1e-5);
    CHECK_EQ(g.samples[t].aux_weight, 0.f);  // resigned game: no ownership target
  }
}

TEST(training_reduces_loss_on_fixed_data) {
  auto dir = temp_dir("go9_test_train_fixed");
  auto cfg = tiny_train(dir.string());
  cfg.lr = 3e-3f;
  train::Trainer tr(cfg);
  auto ev = std::make_shared<nn::Evaluator>(tr.net(), torch::Device(torch::kCPU));
  train::SelfPlay sp(cfg, ev);
  auto games = sp.play(4, 1, nullptr, nullptr);
  for (auto& g : games)
    for (auto& s : g.samples) tr.replay().add(s);
  CHECK(tr.replay().size() > 20);
  auto first = tr.train_steps(5);
  train::LossStats last;
  for (int i = 0; i < 8; ++i) last = tr.train_steps(10);
  CHECK(first.steps == 5 && last.steps == 10);
  CHECK(std::isfinite(last.total));
  CHECK(last.total < first.total * 0.8);  // memorising a tiny buffer must work
  fs::remove_all(dir);
}

TEST(trainer_checkpoint_resume) {
  auto dir = temp_dir("go9_test_resume");
  auto cfg = tiny_train(dir.string());
  cfg.max_iters = 2;
  cfg.replay_save_every = 1;
  {
    train::Trainer tr(cfg);
    CHECK_EQ(tr.run(nullptr), 0);
    CHECK_EQ(tr.iteration(), 2);
  }
  CHECK(fs::exists(dir / "model_latest.pt"));
  CHECK(fs::exists(dir / "optimizer_latest.pt"));
  CHECK(fs::exists(dir / "replay.bin"));
  CHECK(fs::exists(dir / "metrics.csv"));
  CHECK(fs::exists(dir / "state.txt"));
  // Resume: picks up iteration counter, weights and replay buffer.
  train::Trainer tr2(cfg);
  CHECK(tr2.load_checkpoint());
  CHECK_EQ(tr2.iteration(), 2);
  CHECK(tr2.replay().size() > 0);
  auto saved = nn::load_model((dir / "model_latest.pt").string(), torch::kCPU);
  auto p1 = saved->parameters(), p2 = tr2.net()->parameters();
  CHECK_EQ(p1.size(), p2.size());
  for (size_t i = 0; i < p1.size(); ++i) CHECK(torch::equal(p1[i], p2[i]));
  // Continue for one more iteration.
  auto cfg3 = cfg;
  cfg3.max_iters = 3;
  train::Trainer tr3(cfg3);
  CHECK_EQ(tr3.run(nullptr), 0);
  CHECK_EQ(tr3.iteration(), 3);
  fs::remove_all(dir);
}

TEST(arena_match_vs_random) {
  auto ev = std::make_shared<nn::Evaluator>(nn::GoNet(tiny()), torch::Device(torch::kCPU));
  train::MatchConfig mc;
  mc.games = 4;
  mc.sims = 8;
  mc.max_moves = 60;
  auto r = train::play_match(ev, nullptr, mc);
  CHECK_EQ(r.games(), 4);
  CHECK_EQ(int(r.sgfs.size()), 4);
  CHECK_NEAR(train::elo_from_score(0.5), 0.0, 1e-9);
  CHECK(train::elo_from_score(0.75) > 150);
}
