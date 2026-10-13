#include "gtp/gtp.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>

namespace gtp {

using namespace go;

namespace {

std::string lower(std::string s) {
  for (auto& c : s) c = char(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

const char* kCommands[] = {"protocol_version", "name", "version", "known_command", "list_commands", "quit",
                           "boardsize", "clear_board", "komi", "play", "genmove", "undo", "showboard",
                           "final_score", "set_free_handicap", "lz-analyze", "kata-analyze", "lz-genmove_analyze",
                           "kata-genmove_analyze", "go9-visits", "printsgf"};

// Formats one analysis snapshot as an "info ..." line (Leela Zero or KataGo flavour).
std::string info_line(const play::Analysis& a, bool kata, bool ownership) {
  std::ostringstream os;
  int order = 0;
  for (auto& m : a.moves) {
    if (m.visits == 0) continue;
    if (order) os << " ";
    os << "info move " << action_to_gtp(m.move) << " visits " << m.visits;
    if (kata) {
      os << " utility " << (2 * m.winrate - 1) << " winrate " << m.winrate << " scoreMean " << m.score_lead
         << " scoreStdev 0 scoreLead " << m.score_lead << " scoreSelfplay " << m.score_lead << " prior " << m.prior
         << " lcb " << m.lcb << " utilityLcb " << (2 * m.lcb - 1);
    } else {
      os << " winrate " << int(std::lround(m.winrate * 10000)) << " prior " << int(std::lround(m.prior * 10000))
         << " lcb " << int(std::lround(m.lcb * 10000));
    }
    os << " order " << order << " pv";
    for (int x : m.pv) os << " " << action_to_gtp(x);
    ++order;
  }
  if (kata && ownership) {
    // KataGo ownership is from the perspective of the player to move, rows top to bottom.
    const float sign = a.to_move == BLACK ? 1.f : -1.f;
    os << " ownership";
    for (int i = 0; i < NN; ++i) os << " " << sign * a.ownership_black[i];
  }
  return os.str();
}

}  // namespace

int run_gtp(std::shared_ptr<play::PlayEngine> engine, int visits) {
  auto& eng = *engine;
  std::string line;
  std::atomic<bool> analyzing{false};
  std::thread analysis_thread;

  auto stop_analysis = [&]() {
    if (analysis_thread.joinable()) {
      analyzing = false;
      analysis_thread.join();
      std::cout << "\n" << std::flush;  // terminates the analysis response
    }
  };

  while (std::getline(std::cin, line)) {
    stop_analysis();
    // Strip comments and control characters.
    auto hash = line.find('#');
    if (hash != std::string::npos) line = line.substr(0, hash);
    for (auto& c : line)
      if (c == '\t') c = ' ';
    std::istringstream is(line);
    std::vector<std::string> tok;
    std::string t;
    while (is >> t) tok.push_back(t);
    if (tok.empty()) continue;
    std::string id;
    if (std::isdigit(static_cast<unsigned char>(tok[0][0]))) {
      id = tok[0];
      tok.erase(tok.begin());
      if (tok.empty()) continue;
    }
    const std::string cmd = lower(tok[0]);
    auto ok = [&](const std::string& msg = "") { std::cout << "=" << id << " " << msg << "\n\n" << std::flush; };
    auto fail = [&](const std::string& msg) { std::cout << "?" << id << " " << msg << "\n\n" << std::flush; };

    std::lock_guard<std::mutex> lock(eng.mutex());
    if (cmd == "protocol_version") ok("2");
    else if (cmd == "name") ok("go9");
    else if (cmd == "version") ok("1.0");
    else if (cmd == "known_command") {
      bool known = false;
      if (tok.size() > 1)
        for (auto c : kCommands) known |= tok[1] == c;
      ok(known ? "true" : "false");
    } else if (cmd == "list_commands") {
      std::string s;
      for (auto c : kCommands) s += std::string(s.empty() ? "" : "\n") + c;
      ok(s);
    } else if (cmd == "quit") {
      ok();
      break;
    } else if (cmd == "boardsize") {
      if (tok.size() > 1 && tok[1] == "9") ok();
      else fail("unacceptable size");
    } else if (cmd == "clear_board") {
      eng.new_game(eng.env().komi());
      ok();
    } else if (cmd == "komi") {
      try {
        float k = std::stof(tok.at(1));
        // Keep the game, just change komi (replay moves).
        std::vector<int> moves = eng.env().moves();
        eng.new_game(k);
        for (int a : moves) eng.play(a);
        ok();
      } catch (...) {
        fail("syntax error");
      }
    } else if (cmd == "play") {
      if (tok.size() < 3) {
        fail("syntax error");
        continue;
      }
      Color c = lower(tok[1])[0] == 'b' ? BLACK : WHITE;
      int a = gtp_to_action(tok[2]);
      if (a < 0) {
        fail("invalid vertex");
        continue;
      }
      if (eng.env().done()) {
        fail("game is over");
        continue;
      }
      if (c != eng.env().to_move()) eng.play(PASS);  // GTP allows consecutive moves by one colour
      if (!eng.play(a)) fail("illegal move");
      else ok();
    } else if (cmd == "genmove" || cmd == "lz-genmove_analyze" || cmd == "kata-genmove_analyze") {
      if (tok.size() < 2) {
        fail("syntax error");
        continue;
      }
      Color c = lower(tok[1])[0] == 'b' ? BLACK : WHITE;
      if (eng.env().done()) {
        ok("pass");
        continue;
      }
      if (c != eng.env().to_move()) eng.play(PASS);
      play::Analysis a;
      int move = eng.choose_move(visits, true, &a);
      if (cmd != "genmove") {
        std::cout << "=" << id << "\n" << info_line(a, cmd[0] == 'k', false) << "\n";
        std::cout << "play " << (move == -2 ? "resign" : action_to_gtp(move)) << "\n\n" << std::flush;
      } else {
        ok(move == -2 ? "resign" : action_to_gtp(move));
      }
      if (move == -2) eng.resign(c);
      else eng.play(move);
    } else if (cmd == "undo") {
      if (eng.undo()) ok();
      else fail("cannot undo");
    } else if (cmd == "showboard") {
      ok("\n" + eng.env().render());
    } else if (cmd == "final_score") {
      ok(score_to_result(eng.env().board().score()));
    } else if (cmd == "printsgf") {
      ok(eng.env().to_sgf());
    } else if (cmd == "go9-visits") {
      try {
        visits = std::max(1, std::stoi(tok.at(1)));
        ok();
      } catch (...) {
        fail("syntax error");
      }
    } else if (cmd == "lz-analyze" || cmd == "kata-analyze") {
      // Arguments: [color] [interval] [ownership true]
      int interval_cs = 50;
      bool ownership = false;
      for (size_t i = 1; i < tok.size(); ++i) {
        std::string arg = lower(tok[i]);
        if (arg == "interval" && i + 1 < tok.size()) interval_cs = std::max(1, std::atoi(tok[++i].c_str()));
        else if (arg == "ownership" && i + 1 < tok.size()) ownership = lower(tok[++i]) == "true";
        else if (std::isdigit(static_cast<unsigned char>(arg[0]))) interval_cs = std::max(1, std::atoi(arg.c_str()));
      }
      std::cout << "=" << id << "\n" << std::flush;
      const bool kata = cmd == "kata-analyze";
      analyzing = true;
      analysis_thread = std::thread([&eng, &analyzing, interval_cs, kata, ownership]() {
        auto last = std::chrono::steady_clock::now();
        while (analyzing) {
          play::Analysis a;
          {
            std::lock_guard<std::mutex> lk(eng.mutex());
            if (eng.env().done()) break;
            a = eng.analyze(32);
          }
          auto now = std::chrono::steady_clock::now();
          if (std::chrono::duration<double>(now - last).count() * 100 >= interval_cs) {
            std::cout << info_line(a, kata, ownership) << "\n" << std::flush;
            last = now;
          }
          if (a.visits >= 200000) std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
      });
    } else {
      fail("unknown command");
    }
  }
  stop_analysis();
  return 0;
}

}  // namespace gtp
