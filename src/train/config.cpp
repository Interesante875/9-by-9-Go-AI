#include "train/config.h"

#include <algorithm>
#include <fstream>
#include <functional>
#include <sstream>

namespace train {

namespace {

struct Field {
  std::function<std::string(const TrainConfig&)> get;
  std::function<bool(TrainConfig&, const std::string&)> set;
};

template <typename T>
bool parse(const std::string& s, T* out);
template <>
bool parse(const std::string& s, int* out) {
  try {
    size_t pos;
    *out = std::stoi(s, &pos);
    return pos == s.size();
  } catch (...) {
    return false;
  }
}
template <>
bool parse(const std::string& s, float* out) {
  try {
    size_t pos;
    *out = std::stof(s, &pos);
    return pos == s.size();
  } catch (...) {
    return false;
  }
}
template <>
bool parse(const std::string& s, std::string* out) {
  *out = s;
  return true;
}

template <typename T>
std::string fmt(const T& v) {
  std::ostringstream os;
  os << v;
  return os.str();
}

const std::vector<std::pair<std::string, Field>>& fields() {
  static const std::vector<std::pair<std::string, Field>> f = [] {
    std::vector<std::pair<std::string, Field>> v;
#define F(name)                                                                              \
  v.push_back({#name, Field{[](const TrainConfig& c) { return fmt(c.name); },                \
                            [](TrainConfig& c, const std::string& s) { return parse(s, &c.name); }}});
    F(run_dir) F(device) F(seed) F(max_iters) F(torch_threads) F(algo) F(blocks) F(channels) F(komi)
    F(max_moves) F(games_per_iter) F(parallel_games) F(selfplay_threads) F(full_sims) F(fast_sims)
    F(full_prob) F(temp_moves) F(temperature) F(c_puct) F(fpu_reduction) F(noise_eps) F(noise_alpha_total)
    F(resign_threshold) F(noresign_frac) F(resign_min_moves) F(leaves_per_search) F(optimizer) F(lr)
    F(lr_min) F(lr_decay_iters) F(weight_decay) F(momentum) F(batch_size) F(sample_reuse) F(min_replay)
    F(max_train_steps) F(grad_clip) F(replay_capacity) F(per_alpha) F(per_beta) F(per_beta_iters)
    F(w_policy) F(w_value) F(w_ownership) F(w_score) F(w_ppo) F(w_entropy) F(ppo_clip) F(gae_lambda)
    F(save_every) F(replay_save_every) F(snapshot_every) F(eval_every) F(eval_games) F(eval_sims)
    F(sgf_every)
#undef F
    return v;
  }();
  return f;
}

std::string trim(const std::string& s) {
  size_t a = s.find_first_not_of(" \t\r\n");
  if (a == std::string::npos) return "";
  size_t b = s.find_last_not_of(" \t\r\n");
  return s.substr(a, b - a + 1);
}

}  // namespace

std::map<std::string, std::string> TrainConfig::to_map() const {
  std::map<std::string, std::string> m;
  for (auto& [k, f] : fields()) m[k] = f.get(*this);
  return m;
}

bool TrainConfig::set(const std::string& key, const std::string& value, std::string* error) {
  for (auto& [k, f] : fields()) {
    if (k == key) {
      if (!f.set(*this, value)) {
        if (error) *error = "bad value for '" + key + "': " + value;
        return false;
      }
      if (key == "algo" && algo != "alphazero" && algo != "ppo" && algo != "hybrid") {
        if (error) *error = "algo must be alphazero, ppo or hybrid";
        return false;
      }
      return true;
    }
  }
  if (error) *error = "unknown config key '" + key + "'";
  return false;
}

bool TrainConfig::load_file(const std::string& path, std::string* error) {
  std::ifstream in(path);
  if (!in) {
    if (error) *error = "cannot open " + path;
    return false;
  }
  std::string line;
  int ln = 0;
  while (std::getline(in, line)) {
    ++ln;
    auto hash = line.find('#');
    if (hash != std::string::npos) line = line.substr(0, hash);
    line = trim(line);
    if (line.empty()) continue;
    auto eq = line.find('=');
    if (eq == std::string::npos) {
      if (error) *error = path + ":" + std::to_string(ln) + ": expected key = value";
      return false;
    }
    if (!set(trim(line.substr(0, eq)), trim(line.substr(eq + 1)), error)) {
      if (error) *error = path + ":" + std::to_string(ln) + ": " + *error;
      return false;
    }
  }
  return true;
}

bool TrainConfig::save_file(const std::string& path) const {
  std::ofstream out(path);
  if (!out) return false;
  out << "# go9 training configuration (edit and pass with --config, or override with --key value)\n";
  for (auto& [k, f] : fields()) out << k << " = " << f.get(*this) << "\n";
  return bool(out);
}

bool TrainConfig::apply_args(const std::vector<std::string>& args, std::string* error) {
  for (size_t i = 0; i < args.size(); ++i) {
    std::string a = args[i];
    if (a.rfind("--", 0) != 0) {
      if (error) *error = "unexpected argument '" + a + "'";
      return false;
    }
    a = a.substr(2);
    std::string key, value;
    auto eq = a.find('=');
    if (eq != std::string::npos) {
      key = a.substr(0, eq);
      value = a.substr(eq + 1);
    } else {
      key = a;
      if (i + 1 >= args.size()) {
        if (error) *error = "missing value for --" + key;
        return false;
      }
      value = args[++i];
    }
    std::replace(key.begin(), key.end(), '-', '_');
    if (!set(key, value, error)) return false;
  }
  return true;
}

std::string TrainConfig::summary() const {
  std::ostringstream os;
  for (auto& [k, f] : fields()) os << "  " << k << " = " << f.get(*this) << "\n";
  return os.str();
}

}  // namespace train
