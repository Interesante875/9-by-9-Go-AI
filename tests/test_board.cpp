#include <cmath>
#include <random>
#include <set>

#include "go/board.h"
#include "nn/features.h"
#include "test_framework.h"

using namespace go;

namespace {

// Plays a sequence of GTP vertices ("pass" allowed) and checks every move is legal.
void play_all(Board& b, const std::vector<std::string>& moves) {
  for (auto& m : moves) {
    int a = gtp_to_action(m);
    CHECK(a >= 0);
    if (!b.play(a)) throw testing::Failure{"illegal move in setup: " + m + "\n" + b.to_string()};
  }
}

int A(const std::string& v) { return gtp_to_action(v); }

}  // namespace

TEST(coordinates_roundtrip) {
  for (int a = 0; a < NUM_ACTIONS; ++a) {
    CHECK_EQ(gtp_to_action(action_to_gtp(a)), a);
    if (a != PASS) CHECK_EQ(sgf_to_action(action_to_sgf(a)), a);
  }
  CHECK_EQ(action_to_gtp(0), std::string("A9"));
  CHECK_EQ(action_to_gtp(80), std::string("J1"));
  CHECK_EQ(gtp_to_action("e5"), 40);
  CHECK_EQ(gtp_to_action("I5"), -1);  // no I column in Go
  CHECK_EQ(gtp_to_action("A10"), -1);
}

TEST(empty_board) {
  Board b;
  CHECK_EQ(int(b.legal_actions().size()), NUM_ACTIONS);
  CHECK_EQ(b.to_move(), BLACK);
  CHECK_NEAR(b.score(), -7.0, 1e-6);
  CHECK_EQ(b.winner(), WHITE);
  CHECK(!b.is_over());
}

TEST(single_stone_owns_whole_board) {
  Board b;
  play_all(b, {"E5"});
  CHECK_NEAR(b.score(), 81 - 7.0, 1e-6);
}

TEST(capture_single_stone) {
  Board b;
  play_all(b, {"E5", "E4", "A1", "E6", "A2", "D5", "A3"});
  CHECK_EQ(b.group_liberties(A("E5")), 1);
  play_all(b, {"F5"});
  CHECK_EQ(b.at(A("E5")), EMPTY);
  CHECK_EQ(b.captures(WHITE), 1);
  CHECK_EQ(b.captures(BLACK), 0);
  CHECK_EQ(b.hash(), b.compute_hash());
}

TEST(capture_group_and_multiple_groups) {
  // Black group of two in the corner captured by White.
  Board b;
  play_all(b, {"A9", "C9", "B9", "A8", "J1", "B8"});
  CHECK_EQ(b.at(A("A9")), EMPTY);
  CHECK_EQ(b.at(A("B9")), EMPTY);
  CHECK_EQ(b.captures(WHITE), 2);

  // One move capturing two separate groups at once.
  Board c;
  // Black stones A9 and C9 each in atari after White surrounds; White B9 captures both? Build:
  // White: A8, C8, D9 ; Black: A9, C9 ; White plays B9 capturing both single stones.
  play_all(c, {"A9", "A8", "C9", "C8", "J1", "D9", "J2", "B9"});
  CHECK_EQ(c.at(A("A9")), EMPTY);
  CHECK_EQ(c.at(A("C9")), EMPTY);
  CHECK_EQ(c.captures(WHITE), 2);
  CHECK_EQ(c.hash(), c.compute_hash());
}

TEST(suicide_is_illegal) {
  Board b;
  play_all(b, {"E5", "A2", "E6", "B1"});
  // Black A1 would have no liberties and captures nothing.
  CHECK(!b.is_legal(A("A1")));
  CHECK(!b.play(A("A1")));
  CHECK_EQ(b.to_move(), BLACK);  // state unchanged
  // Multi-stone suicide is also illegal: White fills its own last liberty.
  Board c;
  play_all(c, {"A8", "A9", "B8", "pass", "C9"});
  // White A9 has liberty B9; White playing B9 would be suicide of the 2-stone group.
  CHECK(!c.is_legal(A("B9")));
}

TEST(suicide_that_captures_is_legal) {
  Board b;
  // White A1 in corner; Black A2, B1 would capture. White's A1 stone: liberties A2, B1.
  play_all(b, {"A2", "A1", "E5", "C1", "D1", "B2"});
  // Now Black B1: neighbours A1(W), C1(W), B2(W); captures A1 (its only liberty is B1) -> legal.
  CHECK(b.is_legal(A("B1")));
  CHECK(b.play(A("B1")));
  CHECK_EQ(b.at(A("A1")), EMPTY);
}

namespace {
// Creates a ko at B8/C8 (top-left). After this, Black has just captured at C8 and
// White may not immediately retake at B8.
Board make_ko(KoRule rule) {
  Board b(7.0f, rule);
  play_all(b, {"B9", "C9", "A8", "D8", "B7", "C7", "J1", "B8", "C8"});
  return b;
}
}  // namespace

TEST(simple_ko) {
  for (KoRule rule : {KoRule::Simple, KoRule::PositionalSuperko}) {
    Board b = make_ko(rule);
    CHECK_EQ(b.at(A("B8")), EMPTY);
    CHECK_EQ(b.at(A("C8")), BLACK);
    CHECK_EQ(b.to_move(), WHITE);
    CHECK(!b.is_legal(A("B8")));  // immediate recapture forbidden
    if (rule == KoRule::Simple) CHECK_EQ(b.simple_ko_point(), A("B8"));
    // Ko threat exchange elsewhere, then the recapture is legal.
    play_all(b, {"J5", "J6"});
    CHECK(b.is_legal(A("B8")));
    CHECK(b.play(A("B8")));
    CHECK_EQ(b.at(A("C8")), EMPTY);
    // ... and now Black may not immediately retake.
    CHECK(!b.is_legal(A("C8")));
  }
}

TEST(pass_ko_and_game_end) {
  Board b = make_ko(KoRule::Simple);
  // A pass clears the simple ko restriction for the other player only after a move by both.
  CHECK(b.play(PASS));
  CHECK(!b.is_over());
  CHECK(b.play(PASS));
  CHECK(b.is_over());
  CHECK(!b.is_legal(PASS));  // no moves after the game ended
  CHECK(b.legal_actions().empty());
}

TEST(max_moves_ends_game) {
  Board b(7.0f, KoRule::PositionalSuperko, 4);
  play_all(b, {"A1", "B1", "C1", "D1"});
  CHECK(b.is_over());
}

TEST(area_scoring_wall) {
  // Black wall on column E, White wall on column F: Black 45 points, White 36.
  Board b;
  for (int row = 1; row <= 9; ++row) play_all(b, {"E" + std::to_string(row), "F" + std::to_string(row)});
  CHECK_NEAR(b.score(), 45 - 36 - 7.0, 1e-6);
  CHECK_EQ(b.winner(), BLACK);
  int8_t own[NN];
  b.area_ownership(own);
  CHECK_EQ(int(own[A("A1")]), 1);
  CHECK_EQ(int(own[A("J9")]), -1);
  // Komi 9.5 flips the result.
  b.set_komi(9.5f);
  CHECK_EQ(b.winner(), WHITE);
}

TEST(dame_is_neutral) {
  // Black wall column D, White wall column F: column E is dame (touches both).
  Board b;
  for (int row = 1; row <= 9; ++row) play_all(b, {"D" + std::to_string(row), "F" + std::to_string(row)});
  int8_t own[NN];
  b.area_ownership(own);
  for (int row = 1; row <= 9; ++row) CHECK_EQ(int(own[A("E" + std::to_string(row))]), 0);
  CHECK_NEAR(b.score(), 36 - 36 - 7.0, 1e-6);
}

TEST(liberty_map) {
  Board b;
  play_all(b, {"E5", "A1", "E6"});
  uint8_t libs[NN];
  b.liberty_map(libs, 10);
  CHECK_EQ(int(libs[A("E5")]), 6);
  CHECK_EQ(int(libs[A("E6")]), 6);
  CHECK_EQ(int(libs[A("A1")]), 2);
  CHECK_EQ(int(libs[A("D4")]), 0);
  uint8_t libs2[NN];
  nn::liberty_map_from_cells(b.snapshot().boards[0], libs2, 10);
  for (int a = 0; a < NN; ++a) CHECK_EQ(int(libs[a]), int(libs2[a]));
}

TEST(random_games_invariants) {
  std::mt19937_64 rng(7);
  for (int g = 0; g < 60; ++g) {
    Board b(7.0f, KoRule::PositionalSuperko, 400);
    std::set<uint64_t> seen{b.hash()};
    while (!b.is_over()) {
      auto legal = b.legal_actions();
      CHECK(!legal.empty());
      int a = legal[rng() % legal.size()];
      if (a == PASS && legal.size() > 1 && b.move_number() < 250) a = legal[rng() % (legal.size() - 1)];
      CHECK(b.play(a));
      CHECK_EQ(b.hash(), b.compute_hash());
      if (a != PASS) {
        // Positional superko: a stone-placing move never recreates an earlier position.
        CHECK(seen.insert(b.hash()).second);
      }
      for (int p = 0; p < NN; ++p)
        if (b.at(p) != EMPTY) CHECK(b.group_liberties(p) > 0);
      // Superko legality is a subset of simple-ko legality (no false negatives from hashing).
    }
    int stones = b.stone_count(BLACK) + b.stone_count(WHITE);
    CHECK(stones <= NN);
  }
}

TEST(superko_stricter_than_simple_ko) {
  std::mt19937_64 rng(99);
  int checked = 0;
  for (int g = 0; g < 30; ++g) {
    Board sk(7.0f, KoRule::PositionalSuperko, 300);
    Board simple(7.0f, KoRule::Simple, 300);
    while (!sk.is_over()) {
      auto legal_sk = sk.legal_actions();
      for (int a : legal_sk) CHECK(simple.is_legal(a));
      int a = legal_sk[rng() % legal_sk.size()];
      if (a == PASS && legal_sk.size() > 1) a = legal_sk[rng() % (legal_sk.size() - 1)];
      CHECK(sk.play(a));
      CHECK(simple.play(a));
      ++checked;
    }
  }
  CHECK(checked > 1000);
}

TEST(snapshot_history) {
  Board b;
  play_all(b, {"E5", "D4", "C3"});
  Snapshot s = b.snapshot();
  CHECK_EQ(int(s.boards[0][A("C3")]), int(BLACK));
  CHECK_EQ(int(s.boards[1][A("C3")]), int(EMPTY));
  CHECK_EQ(int(s.boards[1][A("D4")]), int(WHITE));
  CHECK_EQ(int(s.boards[2][A("D4")]), int(EMPTY));
  CHECK_EQ(int(s.boards[2][A("E5")]), int(BLACK));
  CHECK_EQ(int(s.boards[3][A("E5")]), int(EMPTY));
  CHECK_EQ(int(s.to_move), int(WHITE));
  CHECK_EQ(int(s.last_move), A("C3"));
  CHECK_EQ(int(s.legal[A("C3")]), 0);
  CHECK_EQ(int(s.legal[PASS]), 1);
}

TEST(symmetries) {
  for (int s = 0; s < 8; ++s) {
    std::set<int> img;
    for (int a = 0; a < NN; ++a) {
      int t = nn::sym_action(s, a);
      CHECK(t >= 0 && t < NN);
      img.insert(t);
      CHECK_EQ(nn::sym_action(nn::inverse_sym(s), t), a);
    }
    CHECK_EQ(int(img.size()), NN);  // bijection
    CHECK_EQ(nn::sym_action(s, PASS), PASS);
    CHECK_EQ(nn::sym_action(s, 40), 40);  // centre is fixed
  }
}

TEST(feature_encoding) {
  Board b;
  play_all(b, {"E5", "D4", "C3"});
  Snapshot snap = b.snapshot();
  std::vector<float> f0(nn::kInputSize), fs(nn::kInputSize);
  nn::encode(snap, 0, f0.data());
  // White to move: own stones plane 0 contains D4, opponent plane 1 contains E5 and C3.
  CHECK_EQ(f0[0 * NN + A("D4")], 1.f);
  CHECK_EQ(f0[1 * NN + A("E5")], 1.f);
  CHECK_EQ(f0[1 * NN + A("C3")], 1.f);
  CHECK_EQ(f0[15 * NN + A("C3")], 1.f);  // last move
  CHECK_EQ(f0[17 * NN], 0.f);           // white to move
  CHECK_NEAR(f0[18 * NN], 7.0 / 15.0, 1e-6);
  // Every symmetry is a permutation of the identity encoding.
  for (int s = 0; s < 8; ++s) {
    nn::encode(snap, s, fs.data());
    for (int p = 0; p < nn::kInputPlanes; ++p)
      for (int a = 0; a < NN; ++a) CHECK_EQ(fs[p * NN + nn::sym_action(s, a)], f0[p * NN + a]);
  }
}
