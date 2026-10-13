"""Local HTTP server for the browser GUI (repository's web/ folder). Same JSON API as the C++ server."""
from __future__ import annotations

import json
import math
import os
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .board import WHITE, action_to_gtp, gtp_to_action
from .model import count_parameters

DEFAULT_WEB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "web"))


def _num(v):
    v = float(v)
    return None if math.isnan(v) or math.isinf(v) else round(v, 4)


def state_json(eng) -> dict:
    env = eng.env
    b = env.board
    legal = b.legal_mask() if not env.done else [0] * 82
    return {
        "board": b.board_list(), "to_move": b.to_move, "move_number": b.move_number, "last_move": b.last_move,
        "captures": {"black": b.captures[1], "white": b.captures[2]}, "komi": b.komi, "over": env.done,
        "result": env.result_string(), "score_area": b.score(), "ko": b.simple_ko_action(), "passes": b.passes,
        "moves": [action_to_gtp(a) for a in env.moves], "move_actions": list(env.moves), "legal": legal[:81],
        "winrate_history": [_num(v) for v in eng.winrate_hist], "score_history": [_num(v) for v in eng.score_hist],
        "model": eng.model_path or "(untrained random network)",
    }


def analysis_json(a) -> dict:
    return {
        "visits": a.visits, "to_move": a.to_move, "winrate_black": _num(a.winrate_black),
        "score_black": _num(a.score_black), "nn_value_black": _num(a.nn_value_black),
        "moves": [{"move": action_to_gtp(m.move), "a": m.move, "visits": m.visits, "winrate": _num(m.winrate),
                   "score_lead": _num(m.score_lead), "prior": _num(m.prior), "lcb": _num(m.lcb),
                   "pv": [action_to_gtp(x) for x in m.pv]} for m in a.moves if m.visits > 0],
        "policy": [_num(x) for x in a.policy], "ownership": [_num(x) for x in a.ownership_black],
    }


def make_handler(eng, web_dir: str, run_dir: str, default_visits: int):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=web_dir, **kwargs)

        def log_message(self, fmt, *args):  # keep the console quiet
            pass

        def _send(self, code: int, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else (
                json.dumps(body).encode() if ctype == "application/json" else body.encode())
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _error(self, msg: str):
            self._send(400, {"error": msg})

        def _params(self):
            q = parse_qs(urlparse(self.path).query)
            return {k: v[0] for k, v in q.items()}

        def _int(self, p, k, d):
            try:
                return int(p.get(k, d))
            except ValueError:
                return d

        def do_GET(self):
            path = urlparse(self.path).path
            p = self._params()
            if not path.startswith("/api/"):
                return super().do_GET()
            with eng.lock:
                if path == "/api/state":
                    return self._send(200, {"state": state_json(eng)})
                if path == "/api/sgf":
                    return self._send(200, eng.env.to_sgf(), "application/x-go-sgf")
                if path == "/api/info":
                    return self._send(200, {"model": eng.model_path, "blocks": eng.ev.net.config["blocks"],
                                            "channels": eng.ev.net.config["channels"],
                                            "params": count_parameters(eng.ev.net), "device": str(eng.ev.device),
                                            "run_dir": run_dir, "default_visits": default_visits})
            if path == "/api/metrics":
                which = p.get("file", "metrics")
                if which not in ("metrics", "eval"):
                    return self._error("file must be metrics or eval")
                if not run_dir:
                    return self._error("no --run directory given to the GUI")
                f = os.path.join(run_dir, which + ".csv")
                if not os.path.exists(f):
                    return self._error(f"no {f} yet - start training first")
                with open(f) as fh:
                    return self._send(200, fh.read(), "text/csv")
            if path == "/api/log":
                if not run_dir:
                    return self._error("no --run directory given to the GUI")
                f = os.path.join(run_dir, "train.log")
                text = open(f).read()[-20000:] if os.path.exists(f) else ""
                return self._send(200, text, "text/plain")
            self._error("unknown endpoint")

        def do_POST(self):
            path = urlparse(self.path).path
            p = self._params()
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length).decode("utf-8", "replace") if length else ""
            with eng.lock:
                try:
                    if path == "/api/new":
                        eng.new_game(float(p.get("komi", eng.env.komi)))
                    elif path == "/api/play":
                        a = gtp_to_action(p.get("move", ""))
                        if a < 0:
                            return self._error("bad move")
                        if not eng.play(a):
                            return self._error(f"illegal move {p.get('move')}")
                    elif path == "/api/undo":
                        for _ in range(max(1, self._int(p, "n", 1))):
                            eng.undo()
                    elif path == "/api/resign":
                        if eng.env.done:
                            return self._error("game is over")
                        eng.resign(eng.env.to_move)
                    elif path == "/api/genmove":
                        if eng.env.done:
                            return self._error("game is over")
                        move, a = eng.choose_move(max(1, self._int(p, "visits", default_visits)),
                                                  self._int(p, "resign", 1) != 0)
                        out = {"analysis": analysis_json(a)}
                        if move == -2:
                            eng.resign(eng.env.to_move)
                            out["move"] = "resign"
                        else:
                            eng.play(move)
                            out["move"] = action_to_gtp(move)
                        out["state"] = state_json(eng)
                        return self._send(200, out)
                    elif path == "/api/analyze":
                        a = eng.analyze(max(1, self._int(p, "visits", 100)), max(1, self._int(p, "max", 100000)))
                        return self._send(200, {"analysis": analysis_json(a), "state": state_json(eng)})
                    elif path == "/api/sgf":
                        eng.load_sgf(body)
                    elif path == "/api/reload":
                        target = p.get("path") or (os.path.join(run_dir, "model_latest.pt") if run_dir else "") \
                            or eng.model_path
                        if not target:
                            return self._error("no model path to reload")
                        eng.reload(target)
                    else:
                        return self._error("unknown endpoint")
                except Exception as e:  # noqa: BLE001 - report to the GUI
                    return self._error(str(e))
                return self._send(200, {"state": state_json(eng)})

    return Handler


def run_server(eng, host="127.0.0.1", port=8080, web_dir=None, run_dir="", default_visits=400,
               open_browser=False, ready: threading.Event | None = None):
    web_dir = web_dir or DEFAULT_WEB_DIR
    if not os.path.exists(os.path.join(web_dir, "index.html")):
        raise SystemExit(f"Could not find the web GUI at {web_dir} (pass --web <dir>)")
    srv = ThreadingHTTPServer((host, port), make_handler(eng, web_dir, run_dir, default_visits))
    url = f"http://{'localhost' if host == '0.0.0.0' else host}:{port}/"
    print(f"GUI ready at {url}  (web files: {web_dir})", flush=True)
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    if ready:
        ready.set()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return srv


__all__ = ["run_server", "state_json", "analysis_json", "WHITE"]
