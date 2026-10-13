// Training configuration: plain "key = value" files plus --key value overrides.
#pragma once

#include <map>
#include <string>
#include <vector>

namespace train {

struct TrainConfig {
  // Run / hardware
  std::string run_dir = "runs/go9";
  std::string device = "auto";  // auto | cpu | cuda | cuda:N
  int seed = 1;
  int max_iters = 0;            // 0 = train forever (Ctrl+C saves and stops)
  int torch_threads = 0;        // 0 = LibTorch default

  // Algorithm: alphazero (MCTS distillation), ppo (pure PPO on network rollouts),
  // hybrid (MCTS distillation + PPO clipped policy-gradient term).
  std::string algo = "hybrid";

  // Network
  int blocks = 6;
  int channels = 96;

  // Game
  float komi = 7.0f;
  int max_moves = 200;

  // Self-play
  int games_per_iter = 64;
  int parallel_games = 32;     // games batched together per self-play thread
  int selfplay_threads = 2;
  int full_sims = 200;         // playout cap randomisation (KataGo): full search ...
  int fast_sims = 40;          // ... or cheap search
  float full_prob = 0.25f;     // probability of a full search (only full searches train the policy)
  int temp_moves = 12;         // sample moves proportional to visits for the first N moves
  float temperature = 1.0f;
  float c_puct = 1.1f;
  float fpu_reduction = 0.2f;
  float noise_eps = 0.25f;
  float noise_alpha_total = 10.83f;
  float resign_threshold = -0.92f;  // resign when root value < threshold (0 disables)
  float noresign_frac = 0.2f;       // fraction of games that never resign (calibration)
  int resign_min_moves = 20;
  int resign_start_iter = 30;       // no resignation before the value head has learnt something
  int resign_consecutive = 3;       // value must stay below the threshold for N of the player's moves
  int leaves_per_search = 2;        // virtual-loss leaves gathered per game per batch

  // Optimisation
  std::string optimizer = "adamw";  // adamw | sgd
  float lr = 1e-3f;
  float lr_min = 1e-4f;
  int lr_decay_iters = 3000;  // cosine decay length in iterations (0 = constant lr)
  float weight_decay = 1e-4f;
  float momentum = 0.9f;
  int batch_size = 256;
  float sample_reuse = 4.0f;  // average number of times each new sample is trained on
  int min_replay = 4096;
  int max_train_steps = 400;  // per iteration cap
  float grad_clip = 5.0f;

  // Prioritised replay
  int replay_capacity = 200000;
  float per_alpha = 0.6f;
  float per_beta = 0.4f;          // annealed to 1 over per_beta_iters
  int per_beta_iters = 1000;

  // Loss weights
  float w_policy = 1.0f;
  float w_value = 1.0f;
  float w_ownership = 0.5f;
  float w_score = 0.1f;
  float w_ppo = 0.5f;      // PPO clipped surrogate weight (hybrid / ppo)
  float w_entropy = 0.0f;  // entropy bonus
  float ppo_clip = 0.2f;
  float gae_lambda = 0.95f;

  // Checkpointing / evaluation / logging
  int save_every = 1;          // model + optimiser + state every N iterations
  int replay_save_every = 5;   // replay buffer (large) every N iterations
  int snapshot_every = 25;     // keep a permanent copy of the model every N iterations
  int eval_every = 10;         // play against the previous evaluation snapshot
  int eval_games = 40;
  int eval_sims = 64;
  int sgf_every = 1;           // save a sample self-play game every N iterations

  std::map<std::string, std::string> to_map() const;
  // Returns false and fills `error` on unknown keys / bad values.
  bool set(const std::string& key, const std::string& value, std::string* error);
  bool load_file(const std::string& path, std::string* error);
  bool save_file(const std::string& path) const;
  // Parses ["--key", "value", "--key2=value2", ...]
  bool apply_args(const std::vector<std::string>& args, std::string* error);
  std::string summary() const;
  std::string dump() const;  // config-file text
};

}  // namespace train
