#include "train/trainer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "go/env.h"
#include "train/arena.h"
#include "train/selfplay.h"
#include "util/log.h"

namespace fs = std::filesystem;

namespace train {

using namespace go;
using util::format;
using util::log_info;
using util::log_warn;

namespace {
const double kPi = 3.14159265358979323846;
const std::vector<std::string> kMetricColumns = {
    "iter",          "hours",         "games_total",  "samples_total",   "buffer",
    "steps_total",   "loss_total",    "loss_policy",  "loss_value",      "loss_ownership",
    "loss_score",    "loss_ppo",      "entropy",      "policy_kl",       "value_mse",
    "ppo_clip_frac", "lr",            "avg_game_len", "black_winrate",   "resign_rate",
    "false_resign_rate", "selfplay_sec", "train_sec",  "evals_per_sec",  "elo"};
}  // namespace

Trainer::Trainer(const TrainConfig& cfg)
    : cfg_(cfg), device_(nn::pick_device(cfg.device)), rng_(uint64_t(cfg.seed) * 1000003ULL + 17) {
  if (cfg_.torch_threads > 0) torch::set_num_threads(cfg_.torch_threads);
  torch::manual_seed(uint64_t(cfg_.seed));
  replay_ = std::make_unique<ReplayBuffer>(size_t(std::max(1, cfg_.replay_capacity)), cfg_.per_alpha);
  nn::NetConfig nc;
  nc.blocks = cfg_.blocks;
  nc.channels = cfg_.channels;
  net_ = nn::GoNet(nc);
  net_->to(device_);
  setup_optimizer();
}

std::string Trainer::path(const std::string& name) const { return (fs::path(cfg_.run_dir) / name).string(); }

void Trainer::setup_optimizer() {
  if (cfg_.optimizer == "sgd") {
    opt_ = std::make_unique<torch::optim::SGD>(
        net_->parameters(),
        torch::optim::SGDOptions(cfg_.lr).momentum(cfg_.momentum).weight_decay(cfg_.weight_decay).nesterov(true));
  } else {
    opt_ = std::make_unique<torch::optim::AdamW>(net_->parameters(),
                                                 torch::optim::AdamWOptions(cfg_.lr).weight_decay(cfg_.weight_decay));
  }
}

void Trainer::set_lr(double lr) {
  for (auto& g : opt_->param_groups()) g.options().set_lr(lr);
}

double Trainer::current_lr() const {
  if (cfg_.lr_decay_iters <= 0) return cfg_.lr;
  double t = std::min(1.0, double(iter_) / double(cfg_.lr_decay_iters));
  return cfg_.lr_min + 0.5 * (cfg_.lr - cfg_.lr_min) * (1 + std::cos(kPi * t));
}

void Trainer::write_state() const {
  std::ofstream out(path("state.txt.tmp"));
  out << "iter=" << iter_ << "\n"
      << "total_steps=" << total_steps_ << "\n"
      << "total_games=" << total_games_ << "\n"
      << "total_samples=" << total_samples_ << "\n"
      << "elo=" << elo_ << "\n"
      << "train_hours=" << train_hours_ << "\n";
  out.close();
  util::replace_file(path("state.txt.tmp"), path("state.txt"));
}

bool Trainer::read_state() {
  std::ifstream in(path("state.txt"));
  if (!in) return false;
  std::string line;
  while (std::getline(in, line)) {
    auto eq = line.find('=');
    if (eq == std::string::npos) continue;
    std::string k = line.substr(0, eq), v = line.substr(eq + 1);
    try {
      if (k == "iter") iter_ = std::stoi(v);
      else if (k == "total_steps") total_steps_ = std::stoll(v);
      else if (k == "total_games") total_games_ = std::stoll(v);
      else if (k == "total_samples") total_samples_ = std::stoll(v);
      else if (k == "elo") elo_ = std::stod(v);
      else if (k == "train_hours") train_hours_ = std::stod(v);
    } catch (...) {
    }
  }
  return true;
}

void Trainer::save_checkpoint(bool with_replay) {
  auto t0 = std::chrono::steady_clock::now();
  nn::save_model(net_, path("model_latest.pt"));
  {
    torch::serialize::OutputArchive ar;
    opt_->save(ar);
    ar.save_to(path("optimizer_latest.pt.tmp"));
    if (!util::replace_file(path("optimizer_latest.pt.tmp"), path("optimizer_latest.pt")))
      log_warn("could not write optimizer_latest.pt");
  }
  if (with_replay && !replay_->save(path("replay.bin"))) log_warn("failed to save replay buffer");
  write_state();
  double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  log_info(format("checkpoint saved (iter %d%s) in %.1fs", iter_, with_replay ? ", with replay buffer" : "", sec));
}

bool Trainer::load_checkpoint() {
  if (!fs::exists(path("model_latest.pt"))) return false;
  net_ = nn::load_model(path("model_latest.pt"), device_);
  cfg_.blocks = net_->cfg.blocks;
  cfg_.channels = net_->cfg.channels;
  setup_optimizer();
  if (fs::exists(path("optimizer_latest.pt"))) {
    try {
      torch::serialize::InputArchive ar;
      ar.load_from(path("optimizer_latest.pt"), device_);
      opt_->load(ar);
    } catch (const std::exception& e) {
      log_warn(std::string("could not restore optimizer state (starting fresh): ") + e.what());
      setup_optimizer();
    }
  }
  read_state();
  if (fs::exists(path("replay.bin"))) {
    if (replay_->load(path("replay.bin")))
      log_info(format("restored replay buffer: %zu samples", replay_->size()));
    else
      log_warn("replay.bin is incompatible or corrupt - starting with an empty buffer");
  }
  return true;
}

LossStats Trainer::train_steps(int steps) {
  LossStats st;
  if (steps <= 0 || replay_->size() == 0) return st;
  net_->train();
  const int B = cfg_.batch_size;
  const float beta = cfg_.per_beta + (1.f - cfg_.per_beta) * std::min(1.f, float(iter_) / float(std::max(1, cfg_.per_beta_iters)));
  const bool use_ppo = cfg_.algo != "alphazero" && cfg_.w_ppo > 0;
  const float w_policy = cfg_.algo == "ppo" ? 0.f : cfg_.w_policy;

  std::vector<float> x(size_t(B) * nn::kInputSize), pi(size_t(B) * NUM_ACTIONS), mask(size_t(B) * NUM_ACTIONS),
      own(size_t(B) * NN);
  std::vector<float> z(B), score(B), old_logp(B), adv(B), wis(B), pw(B), aw(B);
  std::vector<int64_t> action(B);

  for (int step = 0; step < steps; ++step) {
    auto batch = replay_->sample(size_t(B), beta, rng_);
    for (int i = 0; i < B; ++i) {
      const Sample& s = replay_->at(batch.indices[i]);
      const int sym = int(rng_() % 8);
      nn::encode(s.pos, sym, &x[size_t(i) * nn::kInputSize]);
      float* p = &pi[size_t(i) * NUM_ACTIONS];
      float* m = &mask[size_t(i) * NUM_ACTIONS];
      for (int a = 0; a < NUM_ACTIONS; ++a) {
        int ta = nn::sym_action(sym, a);
        p[ta] = s.policy[a];
        m[ta] = s.pos.legal[a] ? 1.f : 0.f;
      }
      for (int a = 0; a < NN; ++a) own[size_t(i) * NN + nn::sym_action(sym, a)] = float(s.ownership[a]);
      z[i] = s.z;
      score[i] = s.score / 10.f;
      old_logp[i] = s.old_logp;
      adv[i] = s.advantage;
      wis[i] = batch.weights[i];
      pw[i] = s.policy_weight;
      aw[i] = s.aux_weight;
      action[i] = nn::sym_action(sym, s.action);
    }
    auto opts = torch::TensorOptions().dtype(torch::kFloat32);
    auto tx = torch::from_blob(x.data(), {B, nn::kInputPlanes, N, N}, opts).to(device_);
    auto tpi = torch::from_blob(pi.data(), {B, NUM_ACTIONS}, opts).to(device_);
    auto tmask = torch::from_blob(mask.data(), {B, NUM_ACTIONS}, opts).to(device_);
    auto town = torch::from_blob(own.data(), {B, NN}, opts).to(device_);
    auto tz = torch::from_blob(z.data(), {B}, opts).to(device_);
    auto tscore = torch::from_blob(score.data(), {B}, opts).to(device_);
    auto tlogp_old = torch::from_blob(old_logp.data(), {B}, opts).to(device_);
    auto tadv = torch::from_blob(adv.data(), {B}, opts).to(device_);
    auto twis = torch::from_blob(wis.data(), {B}, opts).to(device_);
    auto tpw = torch::from_blob(pw.data(), {B}, opts).to(device_);
    auto taw = torch::from_blob(aw.data(), {B}, opts).to(device_);
    auto tact = torch::from_blob(action.data(), {B}, torch::TensorOptions().dtype(torch::kInt64)).to(device_);

    auto out = net_->forward(tx);
    auto logits = out.policy_logits.masked_fill(tmask < 0.5f, -1e4f);
    auto logp = torch::log_softmax(logits, 1);
    auto probs = logp.exp();

    auto l_policy = -(tpi * logp).sum(1) * tpw;                         // cross-entropy with search policy
    auto l_value = (out.value - tz).pow(2);                             // outcome MSE
    auto l_own = (out.ownership - town).pow(2).mean(1) * taw;           // ownership MSE
    auto l_score = torch::smooth_l1_loss(out.score, tscore, torch::Reduction::None) * taw;
    auto entropy = -(probs * logp).sum(1);

    auto per_sample = w_policy * l_policy + cfg_.w_value * l_value + cfg_.w_ownership * l_own +
                      cfg_.w_score * l_score - cfg_.w_entropy * entropy;
    torch::Tensor l_ppo = torch::zeros({B}, opts.device(device_));
    torch::Tensor clip_frac = torch::zeros({}, opts.device(device_));
    if (use_ppo) {
      auto a_norm = (tadv - tadv.mean()) / (tadv.std() + 1e-6f);
      auto logp_a = logp.gather(1, tact.unsqueeze(1)).squeeze(1);
      auto ratio = (logp_a - tlogp_old).clamp(-20.f, 20.f).exp();
      auto surr1 = ratio * a_norm;
      auto surr2 = ratio.clamp(1.f - cfg_.ppo_clip, 1.f + cfg_.ppo_clip) * a_norm;
      l_ppo = -torch::min(surr1, surr2);
      clip_frac = ((ratio - 1.f).abs() > cfg_.ppo_clip).to(torch::kFloat32).mean();
      per_sample = per_sample + cfg_.w_ppo * l_ppo;
    }
    auto loss = (per_sample * twis).mean();

    opt_->zero_grad();
    loss.backward();
    if (cfg_.grad_clip > 0) torch::nn::utils::clip_grad_norm_(net_->parameters(), cfg_.grad_clip);
    opt_->step();

    // New priorities: value error + policy KL (how "surprising" the sample still is).
    torch::Tensor kl;
    {
      torch::NoGradGuard ng;
      auto tpi_safe = tpi.clamp_min(1e-8f);
      kl = ((tpi * (tpi_safe.log() - logp)).sum(1) * tpw).detach();
      auto prio = ((out.value.detach() - tz).abs() * 0.5f + kl).to(torch::kCPU).contiguous();
      std::vector<float> pr(prio.data_ptr<float>(), prio.data_ptr<float>() + B);
      replay_->update_priorities(batch.indices, pr);
    }

    st.total += loss.item<double>();
    st.policy += l_policy.mean().item<double>();
    st.value += l_value.mean().item<double>();
    st.ownership += l_own.mean().item<double>();
    st.score += l_score.mean().item<double>();
    st.ppo += l_ppo.mean().item<double>();
    st.entropy += entropy.mean().item<double>();
    st.kl += kl.mean().item<double>();
    st.value_mse += l_value.mean().item<double>();
    st.clip_frac += clip_frac.item<double>();
    st.steps++;
  }
  net_->eval();
  if (st.steps) {
    double k = 1.0 / st.steps;
    st.total *= k; st.policy *= k; st.value *= k; st.ownership *= k; st.score *= k;
    st.ppo *= k; st.entropy *= k; st.kl *= k; st.value_mse *= k; st.clip_frac *= k;
  }
  total_steps_ += st.steps;
  return st;
}

void Trainer::evaluate(std::atomic<bool>* stop) {
  const std::string ref_path = path("model_eval_ref.pt");
  MatchConfig mc;
  mc.games = cfg_.eval_games;
  mc.sims = cfg_.eval_sims;
  mc.komi = cfg_.komi;
  mc.max_moves = cfg_.max_moves;
  mc.parallel_games = std::min(cfg_.eval_games, std::max(2, cfg_.parallel_games));
  mc.seed = uint64_t(iter_) * 31 + 7;
  auto current = std::make_shared<nn::Evaluator>(net_, device_);
  auto t0 = std::chrono::steady_clock::now();
  util::CsvWriter csv(path("eval.csv"), {"iter", "opponent", "games", "score", "wins", "losses", "elo_delta", "elo"});
  if (!fs::exists(ref_path)) {
    // First evaluation: sanity check against a random player and set the Elo reference.
    auto r = play_match(current, nullptr, mc, stop, "current", "random");
    log_info(format("eval vs random: %d-%d (score %.1f%%), avg length %.0f", r.a_wins, r.b_wins,
                    100 * r.a_score(), r.avg_length));
    csv.write({{"iter", iter_}, {"opponent", -1}, {"games", r.games()}, {"score", r.a_score()},
               {"wins", r.a_wins}, {"losses", r.b_wins}, {"elo_delta", 0}, {"elo", elo_}});
  } else {
    auto ref = nn::Evaluator::create(ref_path, device_);
    auto r = play_match(current, ref, mc, stop, "current", "previous");
    double delta = elo_from_score(r.a_score());
    elo_ += delta;
    log_info(format("eval vs previous checkpoint: %d-%d-%d (score %.1f%%, as black %d, as white %d) "
                    "Elo %+.0f -> %.0f",
                    r.a_wins, r.b_wins, r.draws, 100 * r.a_score(), r.a_black_wins, r.a_white_wins, delta, elo_));
    csv.write({{"iter", iter_}, {"opponent", 0}, {"games", r.games()}, {"score", r.a_score()}, {"wins", r.a_wins},
               {"losses", r.b_wins}, {"elo_delta", delta}, {"elo", elo_}});
    if (!r.sgfs.empty()) {
      std::ofstream(path("sgf/eval_latest.sgf")) << r.sgfs.front();
    }
  }
  nn::save_model(net_, ref_path);
  log_info(format("evaluation took %.1fs",
                  std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()));
}

int Trainer::run(std::atomic<bool>* stop) {
  fs::create_directories(cfg_.run_dir);
  fs::create_directories(path("snapshots"));
  fs::create_directories(path("sgf"));
  util::Logger::get().open(path("train.log"));
  bool resumed = load_checkpoint();
  if (!resumed) {
    log_info("starting a NEW run in " + cfg_.run_dir);
  } else {
    log_info(format("RESUMED run %s at iteration %d (%lld games, %lld train steps, Elo %.0f)", cfg_.run_dir.c_str(),
                    iter_, (long long)total_games_, (long long)total_steps_, elo_));
  }
  cfg_.save_file(path("config.ini"));
  log_info(format("device: %s | net: %d blocks x %d channels (%lld params) | algo: %s",
                  device_.str().c_str(), net_->cfg.blocks, net_->cfg.channels,
                  (long long)nn::count_parameters(net_), cfg_.algo.c_str()));
  net_->eval();
  evaluator_ = std::make_shared<nn::Evaluator>(net_, device_);

  util::CsvWriter metrics(path("metrics.csv"), kMetricColumns);
  SelfPlay selfplay(cfg_, evaluator_);

  while (!(stop && stop->load())) {
    if (cfg_.max_iters > 0 && iter_ >= cfg_.max_iters) break;
    auto iter_t0 = std::chrono::steady_clock::now();

    // 1) Self-play
    SelfPlayStats sp;
    selfplay.set_resign_enabled(iter_ >= cfg_.resign_start_iter);
    auto games = selfplay.play(cfg_.games_per_iter, uint64_t(cfg_.seed) * 7777 + uint64_t(iter_) * 104729, stop, &sp);
    if (stop && stop->load()) break;  // incomplete iteration: discard, checkpoint below
    int new_samples = 0;
    for (auto& g : games)
      for (auto& s : g.samples) {
        replay_->add(s);
        ++new_samples;
      }
    total_games_ += sp.games;
    total_samples_ += new_samples;
    if (cfg_.sgf_every > 0 && iter_ % cfg_.sgf_every == 0 && !games.empty()) {
      std::ofstream(path("sgf/latest_selfplay.sgf")) << games.back().sgf;
    }

    // 2) Training
    auto train_t0 = std::chrono::steady_clock::now();
    set_lr(current_lr());
    LossStats ls;
    if (int(replay_->size()) >= std::min(cfg_.min_replay, cfg_.replay_capacity)) {
      int steps = int(std::ceil(double(new_samples) * cfg_.sample_reuse / double(cfg_.batch_size)));
      steps = std::max(1, std::min(steps, cfg_.max_train_steps));
      ls = train_steps(steps);
    }
    double train_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - train_t0).count();
    ++iter_;
    double iter_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - iter_t0).count();
    train_hours_ += iter_sec / 3600.0;

    // 3) Logging
    const double avg_len = sp.games ? double(sp.moves) / sp.games : 0;
    const double bw = sp.games ? double(sp.black_wins) / sp.games : 0;
    const double eps = sp.seconds > 0 ? sp.nn_evals / sp.seconds : 0;
    if (ls.steps) {
      log_info(format("iter %d | games %d (len %.0f, B win %.0f%%, resign %d) | samples +%d buf %zu | steps %d | "
                      "loss %.3f (pol %.3f val %.3f own %.3f score %.3f ppo %.3f) ent %.2f | lr %.2e | sp %.0fs "
                      "(%.0f evals/s) train %.0fs",
                      iter_, sp.games, avg_len, 100 * bw, sp.resigned, new_samples, replay_->size(), ls.steps,
                      ls.total, ls.policy, ls.value, ls.ownership, ls.score, ls.ppo, ls.entropy, current_lr(),
                      sp.seconds, eps, train_sec));
    } else {
      log_info(format("iter %d | games %d (len %.0f) | samples +%d buf %zu / %d needed before training | sp %.0fs "
                      "(%.0f evals/s)",
                      iter_, sp.games, avg_len, new_samples, replay_->size(), cfg_.min_replay, sp.seconds, eps));
    }
    std::map<std::string, double> row = {
        {"iter", iter_}, {"hours", train_hours_}, {"games_total", double(total_games_)},
        {"samples_total", double(total_samples_)}, {"buffer", double(replay_->size())},
        {"steps_total", double(total_steps_)}, {"lr", current_lr()}, {"avg_game_len", avg_len},
        {"black_winrate", bw}, {"resign_rate", sp.games ? double(sp.resigned) / sp.games : 0},
        {"false_resign_rate", sp.noresign_games ? double(sp.false_resigns) / sp.noresign_games : 0},
        {"selfplay_sec", sp.seconds}, {"train_sec", train_sec}, {"evals_per_sec", eps}, {"elo", elo_}};
    if (ls.steps) {
      row["loss_total"] = ls.total;
      row["loss_policy"] = ls.policy;
      row["loss_value"] = ls.value;
      row["loss_ownership"] = ls.ownership;
      row["loss_score"] = ls.score;
      row["loss_ppo"] = ls.ppo;
      row["entropy"] = ls.entropy;
      row["policy_kl"] = ls.kl;
      row["value_mse"] = ls.value_mse;
      row["ppo_clip_frac"] = ls.clip_frac;
    }

    // 4) Evaluation, checkpoints, snapshots
    if (cfg_.eval_every > 0 && iter_ % cfg_.eval_every == 0 && ls.steps) {
      evaluate(stop);
      row["elo"] = elo_;
    }
    metrics.write(row);
    if (cfg_.save_every > 0 && iter_ % cfg_.save_every == 0)
      save_checkpoint(cfg_.replay_save_every > 0 && iter_ % cfg_.replay_save_every == 0);
    if (cfg_.snapshot_every > 0 && iter_ % cfg_.snapshot_every == 0) {
      char name[64];
      std::snprintf(name, sizeof(name), "snapshots/model_iter_%06d.pt", iter_);
      nn::save_model(net_, path(name));
      log_info(std::string("snapshot saved: ") + name);
    }
  }
  log_info("stopping: saving final checkpoint (including replay buffer)...");
  save_checkpoint(true);
  log_info("done. Resume any time with the same --run_dir.");
  return 0;
}

}  // namespace train
