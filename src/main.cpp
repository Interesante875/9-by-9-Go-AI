// go9 - 9x9 Go environment, self-play reinforcement learning (LibTorch) and GUI.
//
//   go9 train  [--run_dir runs/main] [--config configs/default.ini] [--key value ...]
//   go9 gui    [--run runs/main | --model model.pt] [--port 8080] [--visits 400] [--open]
//   go9 gtp    [--model model.pt] [--visits 400]
//   go9 match  --a model_a.pt --b model_b.pt|random [--games 40] [--sims 100]
//   go9 selftest
//   go9 bench  [--model model.pt]
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <fstream>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "go/env.h"
#include "gtp/gtp.h"
#include "mcts/mcts.h"
#include "nn/evaluator.h"
#include "play/engine.h"
#include "server/server.h"
#include "train/arena.h"
#include "train/trainer.h"
#include "util/log.h"

namespace fs = std::filesystem;

namespace {

std::atomic<bool> g_stop{false};

void on_signal(int) {
  if (g_stop.exchange(true)) {
    std::cerr << "\nSecond interrupt - exiting immediately (last checkpoint is kept).\n";
    std::_Exit(130);
  }
  std::cerr << "\nInterrupt received - finishing the current step and saving a checkpoint "
               "(press Ctrl+C again to abort)...\n";
}

void usage() {
  std::cout << R"(go9 - 9x9 Go with self-play reinforcement learning

Usage:
  go9 train    [--run_dir DIR] [--config FILE] [--key value ...]
               Train by self-play. Resumes automatically if DIR already holds a checkpoint.
               Ctrl+C saves a checkpoint (model, optimizer, replay buffer) and exits.
               All keys: see configs/default.ini (e.g. --algo hybrid --blocks 10 --channels 128)
  go9 gui      [--run DIR | --model FILE] [--port 8080] [--host 127.0.0.1] [--visits 400]
               [--device auto|cpu|cuda] [--open]
               Play / analyse in the browser and watch training curves.
  go9 gtp      [--run DIR | --model FILE] [--visits 400] [--device ...]
               GTP engine for Sabaki / Lizzie / GoGui (supports lz-analyze, kata-analyze).
  go9 match    --a FILE|random --b FILE|random [--games 40] [--sims 100] [--device ...]
               Head-to-head evaluation between two models.
  go9 selftest Quick environment sanity check (random games, rules invariants).
  go9 bench    [--model FILE] [--device ...]  Neural-net and MCTS throughput.
)";
}

std::map<std::string, std::string> parse_kv(const std::vector<std::string>& args, std::vector<std::string>* rest,
                                            const std::vector<std::string>& flags = {}) {
  std::map<std::string, std::string> kv;
  for (size_t i = 0; i < args.size(); ++i) {
    std::string a = args[i];
    if (a.rfind("--", 0) == 0) {
      std::string key = a.substr(2), val;
      auto eq = key.find('=');
      bool is_flag = std::find(flags.begin(), flags.end(), key) != flags.end();
      if (eq != std::string::npos) {
        val = key.substr(eq + 1);
        key = key.substr(0, eq);
      } else if (is_flag) {
        val = "1";
      } else if (i + 1 < args.size()) {
        val = args[++i];
      }
      kv[key] = val;
    } else if (rest) {
      rest->push_back(a);
    }
  }
  return kv;
}

std::string get(const std::map<std::string, std::string>& kv, const std::string& k, const std::string& def = "") {
  auto it = kv.find(k);
  return it == kv.end() ? def : it->second;
}

std::string resolve_model(const std::map<std::string, std::string>& kv) {
  std::string model = get(kv, "model");
  std::string run = get(kv, "run", get(kv, "run_dir"));
  if (model.empty() && !run.empty()) {
    fs::path p = fs::path(run) / "model_latest.pt";
    if (fs::exists(p)) model = p.string();
    else std::cerr << "[warn] " << p.string() << " does not exist yet - using an untrained network\n";
  }
  if (!model.empty() && !fs::exists(model)) {
    std::cerr << "Model file not found: " << model << "\n";
    std::exit(1);
  }
  return model;
}

int cmd_train(const std::vector<std::string>& args) {
  train::TrainConfig cfg;
  std::string err;
  // Determine run_dir first so a resumed run picks up its saved config.
  auto kv = parse_kv(args, nullptr);
  std::string run_dir = get(kv, "run_dir", get(kv, "run", cfg.run_dir));
  fs::path saved = fs::path(run_dir) / "config.ini";
  if (fs::exists(saved)) {
    if (!cfg.load_file(saved.string(), &err)) {
      std::cerr << "Error in " << saved << ": " << err << "\n";
      return 1;
    }
  }
  std::vector<std::string> overrides;
  for (size_t i = 0; i < args.size(); ++i) {
    if (args[i] == "--config" && i + 1 < args.size()) {
      if (!cfg.load_file(args[i + 1], &err)) {
        std::cerr << "Config error: " << err << "\n";
        return 1;
      }
      ++i;
    } else if (args[i].rfind("--config=", 0) == 0) {
      if (!cfg.load_file(args[i].substr(9), &err)) {
        std::cerr << "Config error: " << err << "\n";
        return 1;
      }
    } else if (args[i] == "--run" && i + 1 < args.size()) {
      overrides.push_back("--run_dir");
      overrides.push_back(args[++i]);
    } else {
      overrides.push_back(args[i]);
    }
  }
  if (!cfg.apply_args(overrides, &err)) {
    std::cerr << "Argument error: " << err << "\n";
    return 1;
  }
  cfg.run_dir = run_dir;
  if (fs::exists(saved)) std::cout << "Resuming from " << run_dir << " (saved config + your overrides)\n";
  std::cout << "Configuration:\n" << cfg.summary() << std::flush;
  train::Trainer trainer(cfg);
  return trainer.run(&g_stop);
}

std::shared_ptr<play::PlayEngine> make_engine(const std::map<std::string, std::string>& kv) {
  auto device = nn::pick_device(get(kv, "device", "auto"));
  std::string model = resolve_model(kv);
  auto eval = nn::Evaluator::create(model, device);
  float komi = std::stof(get(kv, "komi", "7"));
  return std::make_shared<play::PlayEngine>(eval, model, komi);
}

int cmd_gui(const std::vector<std::string>& args) {
  auto kv = parse_kv(args, nullptr, {"open"});
  auto engine = make_engine(kv);
  server::ServerOptions o;
  o.port = std::stoi(get(kv, "port", "8080"));
  o.host = get(kv, "host", "127.0.0.1");
  o.web_dir = get(kv, "web");
  o.run_dir = get(kv, "run", get(kv, "run_dir"));
  o.default_visits = std::stoi(get(kv, "visits", "400"));
  o.open_browser = kv.count("open") > 0;
  std::cout << "Model: " << (engine->model_path().empty() ? "(untrained)" : engine->model_path())
            << " | device: " << engine->evaluator().device().str() << "\n";
  return server::run_server(engine, o);
}

int cmd_gtp(const std::vector<std::string>& args) {
  auto kv = parse_kv(args, nullptr);
  auto engine = make_engine(kv);
  return gtp::run_gtp(engine, std::stoi(get(kv, "visits", "400")));
}

int cmd_match(const std::vector<std::string>& args) {
  auto kv = parse_kv(args, nullptr);
  auto device = nn::pick_device(get(kv, "device", "auto"));
  auto load = [&](const std::string& p) -> std::shared_ptr<nn::Evaluator> {
    if (p == "random") return nullptr;
    return nn::Evaluator::create(p, device);
  };
  std::string a = get(kv, "a"), b = get(kv, "b", "random");
  if (a.empty()) {
    std::cerr << "--a is required\n";
    return 1;
  }
  train::MatchConfig mc;
  mc.games = std::stoi(get(kv, "games", "40"));
  mc.sims = std::stoi(get(kv, "sims", "100"));
  mc.parallel_games = std::min(mc.games, 32);
  mc.komi = std::stof(get(kv, "komi", "7"));
  auto t0 = std::chrono::steady_clock::now();
  auto r = train::play_match(load(a), load(b), mc, &g_stop, a, b);
  double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("%s vs %s: %d wins, %d losses, %d draws (score %.1f%%, Elo diff %+.0f)\n", a.c_str(), b.c_str(),
              r.a_wins, r.b_wins, r.draws, 100 * r.a_score(), train::elo_from_score(r.a_score()));
  std::printf("  as Black: %d wins, as White: %d wins, avg length %.1f moves, %.1fs\n", r.a_black_wins,
              r.a_white_wins, r.avg_length, sec);
  if (kv.count("sgf") && !r.sgfs.empty()) {
    std::ofstream out(get(kv, "sgf"));
    for (auto& s : r.sgfs) out << s;
    std::printf("  games written to %s\n", get(kv, "sgf").c_str());
  }
  return 0;
}

int cmd_selftest() {
  using namespace go;
  std::mt19937_64 rng(2024);
  int games = 0, moves = 0, captures = 0, bwins = 0;
  auto t0 = std::chrono::steady_clock::now();
  for (int g = 0; g < 300; ++g) {
    GoEnv env(7.0f, KoRule::PositionalSuperko, 300);
    env.reset();
    while (!env.done()) {
      auto legal = env.legal_actions();
      // Avoid passing early so games are long and tactical.
      int a = legal[rng() % legal.size()];
      if (a == PASS && legal.size() > 1 && env.board().move_number() < 150) a = legal[rng() % (legal.size() - 1)];
      auto r = env.step(a);
      if (r.illegal) {
        std::cerr << "selftest FAILED: legal action rejected\n";
        return 1;
      }
      ++moves;
      if (env.board().hash() != env.board().compute_hash()) {
        std::cerr << "selftest FAILED: hash mismatch\n";
        return 1;
      }
      // No group on the board may have zero liberties.
      for (int p = 0; p < NN; ++p)
        if (env.board().at(p) != EMPTY && env.board().group_liberties(p) == 0) {
          std::cerr << "selftest FAILED: group without liberties\n" << env.render();
          return 1;
        }
    }
    captures += env.board().captures(BLACK) + env.board().captures(WHITE);
    bwins += env.winner() == BLACK;
    ++games;
  }
  double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("selftest OK: %d random games, %d moves (%.0f moves/s incl. checks), %d stones captured, "
              "Black won %d\n",
              games, moves, moves / sec, captures, bwins);
  return 0;
}

int cmd_bench(const std::vector<std::string>& args) {
  auto kv = parse_kv(args, nullptr);
  auto device = nn::pick_device(get(kv, "device", "auto"));
  auto eval = nn::Evaluator::create(resolve_model(kv), device);
  go::Board b;
  std::vector<go::Snapshot> snaps(256, b.snapshot());
  for (int bs : {1, 16, 64, 256}) {
    std::vector<go::Snapshot> s(snaps.begin(), snaps.begin() + bs);
    eval->evaluate(s);
    auto t0 = std::chrono::steady_clock::now();
    int reps = 0;
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 2.0) {
      eval->evaluate(s);
      ++reps;
    }
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("batch %3d: %8.0f positions/s\n", bs, bs * reps / sec);
  }
  mcts::Config mc;
  mcts::Search search(mc, b, 1);
  auto t0 = std::chrono::steady_clock::now();
  search.run(*eval, 1600, 16);
  double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("MCTS: 1600 visits in %.2fs (%.0f visits/s), best move %s\n", sec, 1600 / sec,
              go::action_to_gtp(search.best_move()).c_str());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::signal(SIGINT, on_signal);
#ifdef SIGTERM
  std::signal(SIGTERM, on_signal);
#endif
  {
    std::error_code ec;
    fs::path exe = fs::absolute(argv[0], ec);
    server::g_argv0_dir = exe.parent_path().string();
  }
  if (argc < 2) {
    usage();
    return 0;
  }
  std::string cmd = argv[1];
  std::vector<std::string> args(argv + 2, argv + argc);
  try {
    if (cmd == "train") return cmd_train(args);
    if (cmd == "gui") return cmd_gui(args);
    if (cmd == "gtp") return cmd_gtp(args);
    if (cmd == "match") return cmd_match(args);
    if (cmd == "selftest") return cmd_selftest();
    if (cmd == "bench") return cmd_bench(args);
    usage();
    return cmd == "help" || cmd == "--help" || cmd == "-h" ? 0 : 1;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
