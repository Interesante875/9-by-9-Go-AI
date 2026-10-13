#include "go/env.h"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <sstream>

#include "nn/features.h"

namespace go {

GoEnv::GoEnv(float komi, KoRule ko, int max_moves)
    : komi_(komi), ko_(ko), max_moves_(max_moves), board_(komi, ko, max_moves) {}

std::vector<float> GoEnv::reset() {
  board_ = Board(komi_, ko_, max_moves_);
  moves_.clear();
  resigned_ = EMPTY;
  return observation();
}

std::vector<float> GoEnv::observation() const {
  std::vector<float> obs(nn::kInputSize);
  nn::encode(board_.snapshot(), 0, obs.data());
  return obs;
}

StepResult GoEnv::step(int action) {
  StepResult r;
  if (done() || !board_.play(action)) {
    r.illegal = true;
    r.done = done();
    r.observation = observation();
    return r;
  }
  moves_.push_back(action);
  const Color mover = opponent(board_.to_move());
  r.done = board_.is_over();
  if (r.done) {
    Color w = board_.winner();
    r.reward = w == EMPTY ? 0.f : (w == mover ? 1.f : -1.f);
  }
  r.observation = observation();
  return r;
}

bool GoEnv::undo() {
  if (moves_.empty()) return false;
  std::vector<int> m(moves_.begin(), moves_.end() - 1);
  reset();
  for (int a : m) step(a);
  return true;
}

void GoEnv::set_komi(float k) {
  komi_ = k;
  board_.set_komi(k);
}

void GoEnv::resign(Color who) { resigned_ = who; }

float GoEnv::final_score() const { return board_.score(); }

Color GoEnv::winner() const {
  if (resigned_ != EMPTY) return opponent(resigned_);
  return board_.winner();
}

std::string GoEnv::result_string() const {
  if (resigned_ != EMPTY) return resigned_ == BLACK ? "W+R" : "B+R";
  if (!board_.is_over()) return "?";
  return score_to_result(board_.score());
}

std::string score_to_result(float s) {
  if (s == 0) return "0";
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%s+%.1f", s > 0 ? "B" : "W", std::fabs(s));
  return buf;
}

std::string make_sgf(const std::vector<int>& moves, float komi, const std::string& result,
                     const std::string& black, const std::string& white, const std::string& comment) {
  std::ostringstream os;
  os << "(;GM[1]FF[4]CA[UTF-8]AP[go9]SZ[" << N << "]KM[" << komi << "]RU[Chinese]";
  os << "PB[" << black << "]PW[" << white << "]";
  if (!result.empty() && result != "?") os << "RE[" << result << "]";
  if (!comment.empty()) os << "C[" << comment << "]";
  Color c = BLACK;
  for (int a : moves) {
    os << ";" << (c == BLACK ? "B" : "W") << "[" << action_to_sgf(a) << "]";
    c = opponent(c);
  }
  os << ")\n";
  return os.str();
}

std::string GoEnv::to_sgf(const std::string& black, const std::string& white) const {
  return make_sgf(moves_, komi_, result_string(), black, white);
}

bool GoEnv::load_sgf(const std::string& sgf, std::string* error) {
  // Very small SGF reader: follows the first variation, reads SZ, KM, B, W, AB/AW unsupported.
  auto fail = [&](const std::string& msg) {
    if (error) *error = msg;
    return false;
  };
  std::vector<std::pair<std::string, std::string>> props;
  size_t i = 0;
  int depth = 0;
  bool done_main = false;
  while (i < sgf.size() && !done_main) {
    char ch = sgf[i];
    if (ch == '(') {
      if (++depth > 1) {
        // Enter first variation only: we simply continue reading.
      }
      ++i;
    } else if (ch == ')') {
      // End of the first (main) variation.
      done_main = true;
    } else if (std::isupper(static_cast<unsigned char>(ch))) {
      std::string key;
      while (i < sgf.size() && std::isupper(static_cast<unsigned char>(sgf[i]))) key += sgf[i++];
      while (i < sgf.size() && std::isspace(static_cast<unsigned char>(sgf[i]))) ++i;
      while (i < sgf.size() && sgf[i] == '[') {
        std::string val;
        ++i;
        while (i < sgf.size() && sgf[i] != ']') {
          if (sgf[i] == '\\' && i + 1 < sgf.size()) ++i;
          val += sgf[i++];
        }
        ++i;
        props.emplace_back(key, val);
        while (i < sgf.size() && std::isspace(static_cast<unsigned char>(sgf[i]))) ++i;
      }
    } else {
      ++i;
    }
  }
  float komi = komi_;
  std::vector<std::pair<Color, int>> seq;
  for (auto& [k, v] : props) {
    if (k == "SZ" && v != "9") return fail("only 9x9 games are supported (SZ[" + v + "])");
    if (k == "KM") {
      try {
        komi = std::stof(v);
      } catch (...) {
      }
    }
    if (k == "AB" || k == "AW") return fail("setup stones (AB/AW) are not supported");
    if (k == "B" || k == "W") {
      int a = sgf_to_action(v);
      if (a < 0) return fail("bad move " + v);
      seq.emplace_back(k == "B" ? BLACK : WHITE, a);
    }
  }
  komi_ = komi;
  reset();
  for (auto& [c, a] : seq) {
    if (c != board_.to_move()) {
      // Two moves in a row by the same colour: insert a pass for the other side.
      if (!step(PASS).illegal && done()) return fail("game ended early");
    }
    if (step(a).illegal) return fail("illegal move " + action_to_gtp(a) + " at move " + std::to_string(moves_.size() + 1));
  }
  return true;
}

}  // namespace go
