#include <random>

#include "go/env.h"
#include "nn/features.h"
#include "test_framework.h"

using namespace go;

TEST(env_reset_and_observation) {
  GoEnv env;
  auto obs = env.reset();
  CHECK_EQ(int(obs.size()), nn::kInputSize);
  CHECK_EQ(env.to_move(), BLACK);
  CHECK(!env.done());
  CHECK_EQ(int(env.legal_actions().size()), NUM_ACTIONS);
}

TEST(env_two_passes_reward) {
  GoEnv env(7.0f);
  env.reset();
  auto r1 = env.step(PASS);
  CHECK(!r1.done);
  CHECK_EQ(r1.reward, 0.f);
  auto r2 = env.step(PASS);  // White passes: game over, White wins by komi
  CHECK(r2.done);
  CHECK_EQ(r2.reward, 1.f);
  CHECK_EQ(env.winner(), WHITE);
  CHECK_EQ(env.result_string(), std::string("W+7.0"));
  auto r3 = env.step(gtp_to_action("E5"));
  CHECK(r3.illegal);
}

TEST(env_illegal_move_keeps_state) {
  GoEnv env;
  env.reset();
  env.step(gtp_to_action("E5"));
  auto r = env.step(gtp_to_action("E5"));
  CHECK(r.illegal);
  CHECK_EQ(env.to_move(), WHITE);
  CHECK_EQ(int(env.moves().size()), 1);
}

TEST(env_undo) {
  GoEnv env;
  env.reset();
  env.step(gtp_to_action("E5"));
  uint64_t h = env.board().hash();
  env.step(gtp_to_action("D4"));
  CHECK(env.undo());
  CHECK_EQ(env.board().hash(), h);
  CHECK_EQ(env.to_move(), WHITE);
  CHECK(env.undo());
  CHECK(!env.undo());
}

TEST(env_resign) {
  GoEnv env;
  env.reset();
  env.resign(BLACK);
  CHECK(env.done());
  CHECK_EQ(env.winner(), WHITE);
  CHECK_EQ(env.result_string(), std::string("W+R"));
}

TEST(sgf_roundtrip) {
  std::mt19937_64 rng(3);
  for (int g = 0; g < 10; ++g) {
    GoEnv env(6.5f);
    env.reset();
    while (!env.done()) {
      auto legal = env.legal_actions();
      int a = legal[rng() % legal.size()];
      if (a == PASS && legal.size() > 1 && env.board().move_number() < 120) a = legal[0];
      env.step(a);
    }
    std::string sgf = env.to_sgf();
    GoEnv copy;
    std::string err;
    CHECK(copy.load_sgf(sgf, &err));
    CHECK_EQ(copy.moves().size(), env.moves().size());
    CHECK_EQ(copy.board().hash(), env.board().hash());
    CHECK_NEAR(copy.komi(), 6.5, 1e-6);
    CHECK_EQ(copy.result_string(), env.result_string());
  }
}

TEST(sgf_rejects_other_sizes) {
  GoEnv env;
  std::string err;
  CHECK(!env.load_sgf("(;GM[1]SZ[19];B[pd])", &err));
  CHECK(env.load_sgf("(;GM[1]SZ[9]KM[5.5];B[ee];W[dd](;B[cc])(;B[gg]))", &err));
  CHECK_EQ(int(env.moves().size()), 3);
  CHECK_EQ(env.moves()[2], sgf_to_action("cc"));
}

TEST(env_full_random_games_terminate) {
  std::mt19937_64 rng(11);
  int finished = 0;
  for (int g = 0; g < 50; ++g) {
    GoEnv env(7.0f, KoRule::PositionalSuperko, 3 * NN);
    env.reset();
    float last_reward = 0;
    Color last_mover = EMPTY;
    while (!env.done()) {
      auto legal = env.legal_actions();
      last_mover = env.to_move();
      auto r = env.step(legal[rng() % legal.size()]);
      CHECK(!r.illegal);
      last_reward = r.reward;
    }
    ++finished;
    Color w = env.winner();
    // Reward is from the perspective of whoever made the final move.
    CHECK_EQ(last_reward, w == EMPTY ? 0.f : (w == last_mover ? 1.f : -1.f));  // komi 7 allows jigo
  }
  CHECK_EQ(finished, 50);
}
