#include "train/replay.h"

#include "util/log.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>

namespace train {

SumTree::SumTree(size_t capacity) : capacity_(capacity) {
  leaf0_ = 1;
  while (leaf0_ < capacity_) leaf0_ <<= 1;
  tree_.assign(2 * leaf0_, 0.0);
}

void SumTree::set(size_t i, double p) {
  size_t k = leaf0_ + i;
  double delta = p - tree_[k];
  while (k >= 1) {
    tree_[k] += delta;
    k >>= 1;
  }
}

size_t SumTree::find(double mass) const {
  size_t k = 1;
  while (k < leaf0_) {
    size_t left = 2 * k;
    if (mass < tree_[left] || tree_[left + 1] <= 0) {
      k = left;
    } else {
      mass -= tree_[left];
      k = left + 1;
    }
  }
  return std::min(k - leaf0_, capacity_ - 1);
}

ReplayBuffer::ReplayBuffer(size_t capacity, float alpha, float eps)
    : capacity_(capacity), alpha_(alpha), eps_(eps), data_(capacity), tree_(capacity) {}

void ReplayBuffer::add(const Sample& s) {
  data_[next_] = s;
  tree_.set(next_, std::pow(double(max_priority_), double(alpha_)));
  next_ = (next_ + 1) % capacity_;
  size_ = std::min(size_ + 1, capacity_);
  ++total_added_;
}

ReplayBuffer::Batch ReplayBuffer::sample(size_t batch_size, float beta, std::mt19937_64& rng) const {
  Batch b;
  if (size_ == 0) return b;
  b.indices.resize(batch_size);
  b.weights.resize(batch_size);
  const double total = tree_.total();
  const double seg = total / double(batch_size);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  double max_w = 0;
  for (size_t i = 0; i < batch_size; ++i) {
    double mass = std::min(total * (1 - 1e-12), seg * (double(i) + u(rng)));
    size_t idx = tree_.find(mass);
    if (idx >= size_) idx = size_t(rng() % size_);  // numerical safety
    double p = tree_.get(idx) / total;
    double w = std::pow(double(size_) * std::max(p, 1e-12), -double(beta));
    b.indices[i] = idx;
    b.weights[i] = float(w);
    max_w = std::max(max_w, w);
  }
  for (auto& w : b.weights) w = float(w / max_w);
  return b;
}

void ReplayBuffer::update_priorities(const std::vector<size_t>& indices, const std::vector<float>& errors) {
  for (size_t i = 0; i < indices.size(); ++i) {
    float p = std::fabs(errors[i]) + eps_;
    max_priority_ = std::max(max_priority_, p);
    tree_.set(indices[i], std::pow(double(p), double(alpha_)));
  }
}

namespace {
const char kMagic[8] = {'G', 'O', '9', 'R', 'P', 'L', 'Y', '2'};
}

bool ReplayBuffer::save(const std::string& path) const {
  const std::string tmp = path + ".tmp";
  FILE* f = std::fopen(tmp.c_str(), "wb");
  if (!f) return false;
  uint64_t header[6] = {uint64_t(sizeof(Sample)), uint64_t(capacity_), uint64_t(size_), uint64_t(next_),
                        total_added_, 0};
  float mp = max_priority_;
  bool ok = std::fwrite(kMagic, 1, 8, f) == 8;
  ok = ok && std::fwrite(header, sizeof(header), 1, f) == 1;
  ok = ok && std::fwrite(&mp, sizeof(mp), 1, f) == 1;
  // Write slots [0, size) - when full, this is the whole ring.
  for (size_t i = 0; ok && i < size_; ++i) {
    float p = float(tree_.get(i));
    ok = std::fwrite(&p, sizeof(p), 1, f) == 1 && std::fwrite(&data_[i], sizeof(Sample), 1, f) == 1;
  }
  ok = (std::fclose(f) == 0) && ok;
  if (!ok) return false;
  return util::replace_file(tmp, path);
}

bool ReplayBuffer::load(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  char magic[8];
  uint64_t header[6];
  float mp;
  bool ok = std::fread(magic, 1, 8, f) == 8 && std::memcmp(magic, kMagic, 8) == 0;
  ok = ok && std::fread(header, sizeof(header), 1, f) == 1 && header[0] == sizeof(Sample);
  ok = ok && std::fread(&mp, sizeof(mp), 1, f) == 1;
  if (!ok) {
    std::fclose(f);
    return false;
  }
  const size_t saved_cap = size_t(header[1]), saved_size = size_t(header[2]), saved_next = size_t(header[3]);
  std::vector<std::pair<float, Sample>> items(saved_size);
  for (size_t i = 0; ok && i < saved_size; ++i)
    ok = std::fread(&items[i].first, sizeof(float), 1, f) == 1 &&
         std::fread(&items[i].second, sizeof(Sample), 1, f) == 1;
  std::fclose(f);
  if (!ok) return false;
  // Re-insert in chronological order (oldest first) so a changed capacity keeps the newest data.
  data_.assign(capacity_, Sample{});
  tree_ = SumTree(capacity_);
  next_ = size_ = 0;
  size_t start = saved_size < saved_cap ? 0 : saved_next;
  for (size_t k = 0; k < saved_size; ++k) {
    auto& it = items[(start + k) % saved_size];
    data_[next_] = it.second;
    tree_.set(next_, it.first);
    next_ = (next_ + 1) % capacity_;
    size_ = std::min(size_ + 1, capacity_);
  }
  total_added_ = header[4];
  max_priority_ = mp;
  return true;
}

}  // namespace train
