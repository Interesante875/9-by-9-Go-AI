#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <random>

#include "test_framework.h"
#include "train/replay.h"

using namespace train;

TEST(sumtree_find) {
  SumTree t(5);
  t.set(0, 1);
  t.set(1, 2);
  t.set(2, 0);
  t.set(3, 3);
  t.set(4, 4);
  CHECK_NEAR(t.total(), 10, 1e-9);
  CHECK_EQ(t.find(0.5), size_t(0));
  CHECK_EQ(t.find(1.5), size_t(1));
  CHECK_EQ(t.find(2.99), size_t(1));
  CHECK_EQ(t.find(3.01), size_t(3));
  CHECK_EQ(t.find(6.5), size_t(4));
  t.set(4, 0);
  CHECK_NEAR(t.total(), 6, 1e-9);
}

TEST(per_sampling_is_proportional) {
  ReplayBuffer rb(4, /*alpha=*/1.0f, /*eps=*/0.0f);
  Sample s{};
  for (int i = 0; i < 4; ++i) {
    s.action = int16_t(i);
    rb.add(s);
  }
  rb.update_priorities({0, 1, 2, 3}, {1.f, 2.f, 3.f, 4.f});
  std::mt19937_64 rng(1);
  std::vector<int> counts(4, 0);
  const int draws = 200000;
  auto b = rb.sample(size_t(draws), 1.0f, rng);
  for (auto i : b.indices) counts[i]++;
  for (int i = 0; i < 4; ++i) CHECK_NEAR(double(counts[i]) / draws, (i + 1) / 10.0, 0.01);
  // IS weights: lowest priority gets weight 1 (max), highest gets 1/4.
  for (size_t k = 0; k < b.indices.size(); ++k) {
    double expected = 1.0 / (b.indices[k] + 1);
    CHECK_NEAR(b.weights[k], expected, 1e-4);
  }
}

TEST(replay_ring_buffer_and_new_samples_get_max_priority) {
  ReplayBuffer rb(3, 0.6f);
  Sample s{};
  for (int i = 0; i < 5; ++i) {
    s.action = int16_t(i);
    rb.add(s);
  }
  CHECK_EQ(rb.size(), size_t(3));
  CHECK_EQ(rb.total_added(), uint64_t(5));
  // Slots now hold actions 3, 4, 2.
  CHECK_EQ(int(rb.at(0).action), 3);
  CHECK_EQ(int(rb.at(1).action), 4);
  CHECK_EQ(int(rb.at(2).action), 2);
  rb.update_priorities({0}, {10.f});
  CHECK(rb.max_priority() >= 10.f);
  s.action = 9;
  rb.add(s);  // overwrites slot 2 with max priority
  CHECK_NEAR(rb.priority(2), std::pow(rb.max_priority(), 0.6f), 1e-3);
}

TEST(replay_save_load) {
  const std::string path = (std::filesystem::temp_directory_path() / "go9_test_replay.bin").string();
  ReplayBuffer rb(10, 0.6f);
  Sample s{};
  for (int i = 0; i < 14; ++i) {
    s.action = int16_t(i);
    s.z = (i % 2) ? 1.f : -1.f;
    s.policy[i % go::NUM_ACTIONS] = 0.5f;
    rb.add(s);
  }
  rb.update_priorities({1, 2}, {3.f, 0.5f});
  CHECK(rb.save(path));
  ReplayBuffer r2(10, 0.6f);
  CHECK(r2.load(path));
  CHECK_EQ(r2.size(), rb.size());
  CHECK_EQ(r2.total_added(), rb.total_added());
  // Chronological order is preserved: oldest kept sample is action 4.
  std::vector<int> a1, a2;
  for (size_t i = 0; i < rb.size(); ++i) a1.push_back(rb.at(i).action);
  for (size_t i = 0; i < r2.size(); ++i) a2.push_back(r2.at(i).action);
  std::sort(a1.begin(), a1.end());
  std::sort(a2.begin(), a2.end());
  CHECK(a1 == a2);
  CHECK_EQ(int(r2.at(0).action), 4);
  // Loading into a smaller buffer keeps the newest samples.
  ReplayBuffer small(4, 0.6f);
  CHECK(small.load(path));
  CHECK_EQ(small.size(), size_t(4));
  for (size_t i = 0; i < 4; ++i) CHECK(small.at(i).action >= 10);
  std::remove(path.c_str());
  CHECK(!r2.load(path));  // missing file
}
