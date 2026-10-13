"""Go Text Protocol engine (Sabaki, Lizzie, GoGui, ...), including lz-analyze / kata-analyze."""
from __future__ import annotations

import sys
import threading
import time

from .board import BLACK, NN, PASS, action_to_gtp, gtp_to_action, score_to_result

COMMANDS = ["protocol_version", "name", "version", "known_command", "list_commands", "quit", "boardsize",
            "clear_board", "komi", "play", "genmove", "undo", "showboard", "final_score", "lz-analyze",
            "kata-analyze", "lz-genmove_analyze", "kata-genmove_analyze", "go9-visits", "printsgf"]


def info_line(a, kata: bool, ownership: bool) -> str:
    parts = []
    order = 0
    for m in a.moves:
        if m.visits == 0:
            continue
        s = f"info move {action_to_gtp(m.move)} visits {m.visits}"
        if kata:
            s += (f" utility {2 * m.winrate - 1:.6f} winrate {m.winrate:.6f} scoreMean {m.score_lead:.3f} "
                  f"scoreStdev 0 scoreLead {m.score_lead:.3f} scoreSelfplay {m.score_lead:.3f} prior {m.prior:.6f} "
                  f"lcb {m.lcb:.6f} utilityLcb {2 * m.lcb - 1:.6f}")
        else:
            s += f" winrate {round(m.winrate * 10000)} prior {round(m.prior * 10000)} lcb {round(m.lcb * 10000)}"
        s += f" order {order} pv " + " ".join(action_to_gtp(x) for x in m.pv)
        parts.append(s)
        order += 1
    out = " ".join(parts)
    if kata and ownership:
        sign = 1 if a.to_move == BLACK else -1
        out += " ownership " + " ".join(f"{sign * float(v):.3f}" for v in a.ownership_black[:NN])
    return out


def run_gtp(eng, visits: int = 400, inp=sys.stdin, out=sys.stdout) -> int:
    analyzing = threading.Event()
    worker = None

    def write(s):
        out.write(s)
        out.flush()

    def stop_analysis():
        nonlocal worker
        if worker is not None:
            analyzing.clear()
            worker.join()
            worker = None
            write("\n")

    for line in inp:
        stop_analysis()
        line = line.split("#", 1)[0].replace("\t", " ").strip()
        tok = line.split()
        if not tok:
            continue
        gid = ""
        if tok[0].isdigit():
            gid = tok.pop(0)
            if not tok:
                continue
        cmd = tok[0].lower()

        def ok(msg=""):
            write(f"={gid} {msg}\n\n")

        def fail(msg):
            write(f"?{gid} {msg}\n\n")

        with eng.lock:
            if cmd == "protocol_version":
                ok("2")
            elif cmd == "name":
                ok("go9py")
            elif cmd == "version":
                ok("1.0")
            elif cmd == "known_command":
                ok("true" if len(tok) > 1 and tok[1] in COMMANDS else "false")
            elif cmd == "list_commands":
                ok("\n".join(COMMANDS))
            elif cmd == "quit":
                ok()
                break
            elif cmd == "boardsize":
                ok() if len(tok) > 1 and tok[1] == "9" else fail("unacceptable size")
            elif cmd == "clear_board":
                eng.new_game(eng.env.komi)
                ok()
            elif cmd == "komi":
                try:
                    k = float(tok[1])
                except (IndexError, ValueError):
                    fail("syntax error")
                    continue
                moves = list(eng.env.moves)
                eng.new_game(k)
                for a in moves:
                    eng.play(a)
                ok()
            elif cmd == "play":
                if len(tok) < 3:
                    fail("syntax error")
                    continue
                color = BLACK if tok[1].lower().startswith("b") else 2
                a = gtp_to_action(tok[2])
                if a < 0:
                    fail("invalid vertex")
                elif eng.env.done:
                    fail("game is over")
                else:
                    if color != eng.env.to_move:
                        eng.play(PASS)
                    ok() if eng.play(a) else fail("illegal move")
            elif cmd in ("genmove", "lz-genmove_analyze", "kata-genmove_analyze"):
                if len(tok) < 2:
                    fail("syntax error")
                    continue
                color = BLACK if tok[1].lower().startswith("b") else 2
                if eng.env.done:
                    ok("pass")
                    continue
                if color != eng.env.to_move:
                    eng.play(PASS)
                move, a = eng.choose_move(visits, True)
                txt = "resign" if move == -2 else action_to_gtp(move)
                if cmd == "genmove":
                    ok(txt)
                else:
                    write(f"={gid}\n{info_line(a, cmd.startswith('kata'), False)}\nplay {txt}\n\n")
                if move == -2:
                    eng.resign(color)
                else:
                    eng.play(move)
            elif cmd == "undo":
                ok() if eng.undo() else fail("cannot undo")
            elif cmd == "showboard":
                ok("\n" + eng.env.render())
            elif cmd == "final_score":
                ok(score_to_result(eng.env.board.score()))
            elif cmd == "printsgf":
                ok(eng.env.to_sgf())
            elif cmd == "go9-visits":
                try:
                    visits = max(1, int(tok[1]))
                    ok()
                except (IndexError, ValueError):
                    fail("syntax error")
            elif cmd in ("lz-analyze", "kata-analyze"):
                interval, ownership = 50, False
                i = 1
                while i < len(tok):
                    t = tok[i].lower()
                    if t == "interval" and i + 1 < len(tok):
                        interval = max(1, int(tok[i + 1]))
                        i += 1
                    elif t == "ownership" and i + 1 < len(tok):
                        ownership = tok[i + 1].lower() == "true"
                        i += 1
                    elif t.isdigit():
                        interval = max(1, int(t))
                    i += 1
                write(f"={gid}\n")
                kata = cmd == "kata-analyze"
                analyzing.set()

                def loop(interval=interval, kata=kata, ownership=ownership):
                    last = time.time()
                    while analyzing.is_set():
                        with eng.lock:
                            if eng.env.done:
                                break
                            a = eng.analyze(32)
                        if time.time() - last >= interval / 100:
                            write(info_line(a, kata, ownership) + "\n")
                            last = time.time()

                worker = threading.Thread(target=loop, daemon=True)
                worker.start()
            else:
                fail("unknown command")
    stop_analysis()
    return 0
