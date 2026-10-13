// Local HTTP server for the browser GUI (board, analysis overlay, training dashboard).
// httplib.h must come first on Windows (winsock2.h before any windows.h).
#include "httplib.h"
#include "server/server.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

#include "nn/model.h"
#include "util/log.h"

namespace fs = std::filesystem;

namespace server {

using namespace go;

namespace {

std::string jnum(double v) {
  if (std::isnan(v) || std::isinf(v)) return "null";
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.4f", v);
  return buf;
}

std::string jstr(const std::string& s) {
  std::string o = "\"";
  for (char c : s) {
    switch (c) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          o += buf;
        } else {
          o += c;
        }
    }
  }
  return o + "\"";
}

template <typename T, typename F>
std::string jarr(const T& v, F f) {
  std::string o = "[";
  bool first = true;
  for (const auto& x : v) {
    if (!first) o += ",";
    first = false;
    o += f(x);
  }
  return o + "]";
}

std::string state_json(play::PlayEngine& eng) {
  const GoEnv& env = eng.env();
  const Board& b = env.board();
  std::ostringstream os;
  os << "{\"board\":[";
  for (int a = 0; a < NN; ++a) os << (a ? "," : "") << int(b.at(a));
  os << "],\"to_move\":" << int(b.to_move()) << ",\"move_number\":" << b.move_number()
     << ",\"last_move\":" << b.last_move() << ",\"captures\":{\"black\":" << b.captures(BLACK)
     << ",\"white\":" << b.captures(WHITE) << "},\"komi\":" << jnum(b.komi())
     << ",\"over\":" << (env.done() ? "true" : "false") << ",\"result\":" << jstr(env.result_string())
     << ",\"score_area\":" << jnum(b.score()) << ",\"ko\":" << b.simple_ko_point()
     << ",\"passes\":" << b.consecutive_passes() << ",\"moves\":"
     << jarr(env.moves(), [](int a) { return jstr(action_to_gtp(a)); }) << ",\"move_actions\":"
     << jarr(env.moves(), [](int a) { return std::to_string(a); });
  os << ",\"legal\":[";
  for (int a = 0; a < NN; ++a) os << (a ? "," : "") << (b.is_legal(a) ? 1 : 0);
  os << "],\"winrate_history\":" << jarr(eng.winrate_history(), [](float v) { return jnum(v); })
     << ",\"score_history\":" << jarr(eng.score_history(), [](float v) { return jnum(v); });
  os << ",\"model\":" << jstr(eng.model_path().empty() ? "(untrained random network)" : eng.model_path()) << "}";
  return os.str();
}

std::string analysis_json(const play::Analysis& a) {
  std::ostringstream os;
  os << "{\"visits\":" << a.visits << ",\"to_move\":" << int(a.to_move)
     << ",\"winrate_black\":" << jnum(a.winrate_black) << ",\"score_black\":" << jnum(a.score_black)
     << ",\"nn_value_black\":" << jnum(a.nn_value_black) << ",\"moves\":[";
  bool first = true;
  for (auto& m : a.moves) {
    if (m.visits == 0) continue;
    if (!first) os << ",";
    first = false;
    os << "{\"move\":" << jstr(action_to_gtp(m.move)) << ",\"a\":" << m.move << ",\"visits\":" << m.visits
       << ",\"winrate\":" << jnum(m.winrate) << ",\"score_lead\":" << jnum(m.score_lead)
       << ",\"prior\":" << jnum(m.prior) << ",\"lcb\":" << jnum(m.lcb)
       << ",\"pv\":" << jarr(m.pv, [](int x) { return jstr(action_to_gtp(x)); }) << "}";
  }
  os << "],\"policy\":" << jarr(a.policy, [](float v) { return jnum(v); })
     << ",\"ownership\":" << jarr(a.ownership_black, [](float v) { return jnum(v); }) << "}";
  return os.str();
}

std::string find_web_dir(const std::string& hint, const std::string& argv0_dir) {
  std::vector<fs::path> cands;
  if (!hint.empty()) cands.push_back(hint);
  cands.push_back(fs::path(argv0_dir) / "web");
  cands.push_back(fs::current_path() / "web");
  cands.push_back(fs::path(argv0_dir) / ".." / "web");
  cands.push_back(fs::path(argv0_dir) / ".." / ".." / "web");
  for (auto& c : cands) {
    std::error_code ec;
    if (fs::exists(c / "index.html", ec)) return fs::absolute(c).string();
  }
  return "";
}

int param_int(const httplib::Request& req, const std::string& key, int def) {
  if (!req.has_param(key)) return def;
  try {
    return std::stoi(req.get_param_value(key));
  } catch (...) {
    return def;
  }
}

float param_float(const httplib::Request& req, const std::string& key, float def) {
  if (!req.has_param(key)) return def;
  try {
    return std::stof(req.get_param_value(key));
  } catch (...) {
    return def;
  }
}

std::string read_file(const fs::path& p) {
  std::ifstream in(p, std::ios::binary);
  std::ostringstream os;
  os << in.rdbuf();
  return os.str();
}

}  // namespace

std::string g_argv0_dir;  // set by main

int run_server(std::shared_ptr<play::PlayEngine> engine, const ServerOptions& opts) {
  httplib::Server srv;
  const std::string web = find_web_dir(opts.web_dir, g_argv0_dir);
  if (web.empty()) {
    std::cerr << "Could not find the web/ directory (index.html). Pass --web <dir>.\n";
    return 1;
  }
  srv.set_mount_point("/", web);
  auto& eng = *engine;
  const std::string json = "application/json";

  auto reply_state = [&](httplib::Response& res, const std::string& extra = "") {
    std::string body = "{\"state\":" + state_json(eng) + extra + "}";
    res.set_content(body, json);
  };
  auto error = [&](httplib::Response& res, const std::string& msg) {
    res.status = 400;
    res.set_content("{\"error\":" + jstr(msg) + "}", json);
  };

  srv.Get("/api/state", [&](const httplib::Request&, httplib::Response& res) {
    std::lock_guard<std::mutex> lock(eng.mutex());
    reply_state(res);
  });

  srv.Post("/api/new", [&](const httplib::Request& req, httplib::Response& res) {
    std::lock_guard<std::mutex> lock(eng.mutex());
    eng.new_game(param_float(req, "komi", eng.env().komi()));
    reply_state(res);
  });

  srv.Post("/api/play", [&](const httplib::Request& req, httplib::Response& res) {
    std::lock_guard<std::mutex> lock(eng.mutex());
    int a = gtp_to_action(req.get_param_value("move"));
    if (a < 0) return error(res, "bad move");
    if (!eng.play(a)) return error(res, "illegal move " + req.get_param_value("move"));
    reply_state(res);
  });

  srv.Post("/api/undo", [&](const httplib::Request& req, httplib::Response& res) {
    std::lock_guard<std::mutex> lock(eng.mutex());
    int n = std::max(1, param_int(req, "n", 1));
    for (int i = 0; i < n; ++i) eng.undo();
    reply_state(res);
  });

  srv.Post("/api/resign", [&](const httplib::Request& req, httplib::Response& res) {
    std::lock_guard<std::mutex> lock(eng.mutex());
    if (eng.env().done()) return error(res, "game is over");
    eng.resign(eng.env().to_move());
    reply_state(res);
  });

  srv.Post("/api/genmove", [&](const httplib::Request& req, httplib::Response& res) {
    std::lock_guard<std::mutex> lock(eng.mutex());
    if (eng.env().done()) return error(res, "game is over");
    int visits = std::max(1, param_int(req, "visits", opts.default_visits));
    bool allow_resign = param_int(req, "resign", 1) != 0;
    play::Analysis a;
    int move = eng.choose_move(visits, allow_resign, &a);
    std::string extra = ",\"analysis\":" + analysis_json(a);
    if (move == -2) {
      eng.resign(eng.env().to_move());
      extra += ",\"move\":\"resign\"";
    } else {
      eng.play(move);
      extra += ",\"move\":" + jstr(action_to_gtp(move));
    }
    reply_state(res, extra);
  });

  srv.Post("/api/analyze", [&](const httplib::Request& req, httplib::Response& res) {
    std::lock_guard<std::mutex> lock(eng.mutex());
    int visits = std::max(1, param_int(req, "visits", 100));
    auto a = eng.analyze(visits, std::max(1, param_int(req, "max", 100000)));
    res.set_content("{\"analysis\":" + analysis_json(a) + ",\"state\":" + state_json(eng) + "}", json);
  });

  srv.Get("/api/sgf", [&](const httplib::Request&, httplib::Response& res) {
    std::lock_guard<std::mutex> lock(eng.mutex());
    res.set_content(eng.env().to_sgf(), "application/x-go-sgf");
  });

  srv.Post("/api/sgf", [&](const httplib::Request& req, httplib::Response& res) {
    std::lock_guard<std::mutex> lock(eng.mutex());
    std::string err;
    if (!eng.load_sgf(req.body, &err)) return error(res, "could not load SGF: " + err);
    reply_state(res);
  });

  srv.Get("/api/info", [&](const httplib::Request&, httplib::Response& res) {
    std::lock_guard<std::mutex> lock(eng.mutex());
    auto& net = eng.evaluator().net();
    std::ostringstream os;
    os << "{\"model\":" << jstr(eng.model_path()) << ",\"blocks\":" << net->cfg.blocks
       << ",\"channels\":" << net->cfg.channels << ",\"params\":" << nn::count_parameters(net)
       << ",\"device\":" << jstr(eng.evaluator().device().str()) << ",\"run_dir\":" << jstr(opts.run_dir)
       << ",\"default_visits\":" << opts.default_visits << "}";
    res.set_content(os.str(), json);
  });

  srv.Get("/api/metrics", [&](const httplib::Request& req, httplib::Response& res) {
    std::string which = req.has_param("file") ? req.get_param_value("file") : "metrics";
    if (which != "metrics" && which != "eval") return error(res, "file must be metrics or eval");
    if (opts.run_dir.empty()) return error(res, "no --run directory given to the GUI");
    fs::path p = fs::path(opts.run_dir) / (which + ".csv");
    if (!fs::exists(p)) return error(res, "no " + p.string() + " yet - start training first");
    res.set_content(read_file(p), "text/csv");
  });

  srv.Get("/api/log", [&](const httplib::Request&, httplib::Response& res) {
    if (opts.run_dir.empty()) return error(res, "no --run directory given to the GUI");
    std::string s = read_file(fs::path(opts.run_dir) / "train.log");
    if (s.size() > 20000) s = s.substr(s.size() - 20000);
    res.set_content(s, "text/plain");
  });

  srv.Post("/api/reload", [&](const httplib::Request& req, httplib::Response& res) {
    std::lock_guard<std::mutex> lock(eng.mutex());
    std::string p = req.has_param("path") ? req.get_param_value("path") : "";
    if (p.empty() && !opts.run_dir.empty()) p = (fs::path(opts.run_dir) / "model_latest.pt").string();
    if (p.empty()) p = eng.model_path();
    if (p.empty()) return error(res, "no model path to reload");
    std::string err;
    if (!eng.reload(p, &err)) return error(res, "reload failed: " + err);
    reply_state(res);
  });

  std::string url = "http://" + (opts.host == "0.0.0.0" ? std::string("localhost") : opts.host) + ":" +
                    std::to_string(opts.port) + "/";
  util::log_info("GUI ready at " + url + "  (web files: " + web + ")");
  if (opts.open_browser) {
#ifdef _WIN32
    std::string cmd = "start \"\" \"" + url + "\"";
#elif __APPLE__
    std::string cmd = "open " + url;
#else
    std::string cmd = "xdg-open " + url + " >/dev/null 2>&1 &";
#endif
    int rc = std::system(cmd.c_str());
    (void)rc;
  }
  if (!srv.listen(opts.host, opts.port)) {
    std::cerr << "Failed to listen on " << opts.host << ":" << opts.port << " (port in use?)\n";
    return 1;
  }
  return 0;
}

}  // namespace server
