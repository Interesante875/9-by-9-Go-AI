// Replays games recorded from OpenSpiel (tools/gen_openspiel_crossval.py) through go9's rules
// engine and verifies legal-move sets, game termination and winners agree position by position.
#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>

#include "go/board.h"

using namespace go;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: go9_crossval <openspiel_games.txt>\n";
    return 2;
  }
  std::ifstream in(argv[1]);
  if (!in) {
    std::cerr << "cannot open " << argv[1] << "\n";
    return 2;
  }
  std::string line;
  Board board, superko;
  std::set<uint64_t> seen;
  bool repetition = false;  // last move recreated an earlier whole-board position
  int games = 0, positions = 0, errors = 0, line_no = 0, repetition_draws = 0;
  auto report = [&](const std::string& msg) {
    if (++errors <= 20) std::cerr << "line " << line_no << ": " << msg << "\n" << board.to_string() << "\n";
  };
  while (std::getline(in, line)) {
    ++line_no;
    if (line.empty() || line[0] == '#') continue;
    std::istringstream is(line);
    std::string tag;
    is >> tag;
    if (tag == "game") {
      float komi;
      int max_len;
      is >> komi >> max_len;
      // OpenSpiel uses simple ko; go9 supports both rules.
      board = Board(komi, KoRule::Simple, max_len);
      superko = Board(komi, KoRule::PositionalSuperko, max_len);
      seen = {board.hash()};
      repetition = false;
      ++games;
    } else if (tag == "m") {
      std::string mv;
      int n;
      is >> mv >> n;
      std::set<int> theirs;
      for (int i = 0; i < n; ++i) {
        std::string v;
        is >> v;
        theirs.insert(gtp_to_action(v));
      }
      auto ours_v = board.legal_actions();
      std::set<int> ours(ours_v.begin(), ours_v.end());
      if (ours != theirs) {
        std::string only_ours, only_theirs;
        for (int a : ours)
          if (!theirs.count(a)) only_ours += action_to_gtp(a) + " ";
        for (int a : theirs)
          if (!ours.count(a)) only_theirs += action_to_gtp(a) + " ";
        report("legal move mismatch. only go9: [" + only_ours + "] only OpenSpiel: [" + only_theirs + "]");
      }
      ++positions;
      const int a = gtp_to_action(mv);
      if (!board.play(a)) report("go9 rejected OpenSpiel move " + mv);
      repetition = a != PASS && !seen.insert(board.hash()).second;
      if (repetition) {
        // go9's default positional-superko rule must forbid exactly this move.
        if (superko.is_legal(a)) report("superko engine allows a move that repeats a position: " + mv);
      } else if (!superko.play(a)) {
        report("superko engine rejected non-repeating move " + mv);
      }
    } else if (tag == "end") {
      int ret;
      is >> ret;
      if (ret == 0 && repetition && !board.is_over()) {
        // OpenSpiel ends a game as a draw when a whole-board position repeats (simple-ko rules
        // let that happen). go9's positional superko makes such a move illegal (checked above).
        ++repetition_draws;
        continue;
      }
      if (!board.is_over()) report("OpenSpiel game ended but go9 thinks it continues");
      int ours = board.winner() == BLACK ? 1 : board.winner() == WHITE ? -1 : 0;
      if (ours != ret) report("winner mismatch: go9 " + std::to_string(ours) + " OpenSpiel " + std::to_string(ret) +
                              " (go9 score " + std::to_string(board.score()) + ")");
    }
  }
  std::cout << "cross-validated " << games << " games / " << positions << " positions against OpenSpiel: "
            << (errors ? std::to_string(errors) + " MISMATCHES" : "all legal-move sets, endings and winners agree")
            << " (" << repetition_draws << " games ended by OpenSpiel's position-repetition rule, each confirmed "
            << "illegal under go9's positional superko)\n";
  return errors ? 1 : 0;
}
