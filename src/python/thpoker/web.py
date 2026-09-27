"""A local web table (DESIGN.md Section 9.2) on the standard-library HTTP server:
`bin/run_thpoker.py web` serves play, the coach hint, and hand review in a browser at
http://127.0.0.1:8000.

It listens on this machine only, keeps its sessions in memory, and logs hands like the command
line. During play it only ever returns the user's observation, never the full state.
"""

import argparse
import http.server
import json
import math
from   pathlib                  import Path
import secrets
import threading
from   typing                   import Any

from   thpoker.analysis.ev      import PROFILES, Profile, pick_profile
from   thpoker.cli              import (PlaySession, add_device_argument,
                                        build_config, parse_args)
from   thpoker.game.cards       import card_str
from   thpoker.game.engine      import is_terminal
from   thpoker.game.rules       import IllegalActionError
from   thpoker.game.state       import Action, ActionType
from   thpoker.storage          import SessionLog, default_log_dir
from   thpoker.table            import TableConfig, TableRunner

STATIC = Path(__file__).resolve().parent / "web_static"
ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}
MAX_BODY = 16_384
MAX_SESSIONS = 8  # the oldest session is dropped beyond this
_KINDS = {kind.value.lower(): kind for kind in ActionType}


class WebError(ValueError):
    """A request the table cannot serve: bad input, or not the moment for it."""


class WebTable(PlaySession):
    """One session: the runner, the narration so far, and the session log."""

    def __init__(self, config: TableConfig, scale: int, log: SessionLog | None, profile: Profile):
        self.lines: list[str] = []
        super().__init__(TableRunner(config), scale, log, profile)

    def say(self, line: str):
        self.lines.append(line)

    def next_hand(self):
        """Deal the next hand, or for a user out of a tournament play it out to the end."""
        runner = self.runner
        if runner.session.finished:
            raise WebError("the session is over")
        if runner.hand is not None and not is_terminal(runner.hand):
            raise WebError("the hand is still being played")
        if runner.session.awaiting_rebuy:
            raise WebError("you are out of chips: rebuy or start a new game")
        if self.knocked_out():
            self.fast_forward()
            return
        self.lines.append(f"=== Hand {runner.session.hand_number + 1}")
        self.record(runner.start_hand(self.minutes()))
        self.hand_over()

    def rebuy(self):
        if not self.runner.session.awaiting_rebuy:
            raise WebError("a rebuy is only for a cash player out of chips")
        self.record(self.runner.user_rebuy())

    def act(self, payload: dict[str, Any]):
        if not self.runner.user_to_act():
            raise WebError("it is not your turn")
        kind = payload.get("kind")
        if not isinstance(kind, str):
            raise WebError("the action needs a kind")
        legal = self.runner.user_legal_actions()
        if kind == "allin":
            action = (
                Action(ActionType.RAISE if legal.can_raise else ActionType.BET, legal.max_raise_to)
                if legal.max_raise_to
                else Action(ActionType.CALL)
            )
        elif kind in ("bet", "raise"):
            amount = payload.get("amount")
            if not isinstance(amount, (int, float)) or isinstance(amount, bool) or not math.isfinite(amount):
                raise WebError("a bet or raise needs an amount")
            action = Action(_KINDS[kind], round(float(amount) * self.scale))
        elif kind in _KINDS:
            action = Action(_KINDS[kind])
        else:
            raise WebError(f"unknown action {kind!r}")
        try:
            self.record(self.runner.act(action))
        except IllegalActionError as error:
            raise WebError(str(error)) from error
        self.hand_over()

    def state(self) -> dict[str, Any]:
        runner = self.runner
        seats = []
        view = runner.user_view() if runner.hand is not None else None
        hand_over = runner.hand is None or is_terminal(runner.hand)
        for seat in range(runner.config.session.num_seats):
            entry: dict[str, Any] = {
                "seat": seat,
                "name": runner.seat_label(seat),
                "stack": runner.session.stacks[seat] / self.scale,
            }
            if view is not None:
                cards = view.hole_cards[seat]
                entry.update(
                    committed=view.committed_this_street[seat] / self.scale,
                    folded=view.folded[seat],
                    all_in=view.all_in[seat],
                    dealt_in=view.dealt_in[seat],
                    button=seat == view.button,
                    # Between hands the session's stacks, which count a rebuy.
                    stack=(runner.session.stacks if hand_over else view.stacks)[seat] / self.scale,
                    cards=[card_str(c) for c in cards] if cards else None,
                )
            if seat in runner.bots:
                entry["hud"] = runner.hud.seats[seat].summary()
            seats.append(entry)
        result: dict[str, Any] = {
            "seats": seats,
            "log": self.lines[-60:],
            "hand_over": hand_over,
            "session_over": runner.session.finished,
            "awaiting_rebuy": runner.session.awaiting_rebuy,
            "knocked_out": self.knocked_out(),
            "your_turn": runner.user_to_act(),
            "step": 1 / self.scale,  # the smallest chip amount, in the units shown
        }
        if view is not None:
            result.update(
                board=[card_str(c) for c in view.board],
                pot=view.pot / self.scale,
                big_blind=view.config.big_blind / self.scale,
            )
        if runner.user_to_act():
            legal = runner.user_legal_actions()
            result["legal"] = {
                "fold": legal.can_fold,
                "check": legal.can_check,
                "call": legal.call_amount / self.scale if legal.can_call else None,
                "raise": {
                    "kind": "raise" if legal.can_raise else "bet",
                    "min": legal.min_raise_to / self.scale,
                    "max": legal.max_raise_to / self.scale,
                }
                if legal.can_bet or legal.can_raise
                else None,
            }
        return result

    def hint(self) -> list[str]:
        if not self.runner.user_to_act():
            raise WebError("a hint needs your turn to act")
        return self.hint_lines()

    def review(self) -> list[str]:
        """The user's last finished hand, which after a fast-forward is not the hand shown."""
        hand, user, current = self.finished, self.runner.user_seat, self.runner.hand
        playing = current is not None and not is_terminal(current)
        if hand is None or user is None or playing:
            raise WebError("a review needs a finished hand")
        if not any(e.seat == user for e in hand.history):
            return ["You made no decision in that hand."]
        return self.review_lines(hand)


def _session_argv(options: dict[str, Any]) -> list[str]:
    """Command-line arguments for a new session, from checked JSON fields."""
    argv = ["--no-log"]
    fields: dict[str, type | tuple[type, ...]] = {
        "mode": str,
        "seats": int,
        "tier": int,
        "blinds": str,
        "stack": (int, float),
        "seed": int,
    }
    for name, kind in fields.items():
        value = options.get(name)
        if value is None:
            continue
        if (
            not isinstance(value, kind)
            or isinstance(value, bool)
            or (isinstance(value, float) and not math.isfinite(value))
        ):
            raise WebError(f"{name} has the wrong type")
        argv += [f"--{name}", str(value)]
    if options.get("hide_styles") is True:
        argv.append("--hide-styles")
    return argv


class App:
    """All sessions, served one request at a time."""

    def __init__(self, log_dir: Path | None, profile: Profile = PROFILES["pc"]):
        self.log_dir = log_dir
        self.profile = profile
        self.tables: dict[str, WebTable] = {}
        self.lock = threading.Lock()

    def handle(self, method: str, parts: list[str], payload: dict[str, Any]) -> dict[str, Any]:
        """Route an API request (the path after /api/, split). Raises `WebError` or `KeyError`."""
        with self.lock:
            if method == "POST" and parts == ["sessions"]:
                try:
                    config, scale = build_config(parse_args(_session_argv(payload)))
                    log = SessionLog(self.log_dir / f"session-{config.session.seed}.jsonl") if self.log_dir else None
                    table = WebTable(config, scale, log, self.profile)
                    table.next_hand()
                except (ValueError, ArithmeticError, SystemExit) as error:
                    raise WebError(f"invalid setup: {error}") from error
                key = secrets.token_hex(8)
                self.tables[key] = table
                while len(self.tables) > MAX_SESSIONS:
                    del self.tables[next(iter(self.tables))]
                return {"id": key, "state": table.state()}
            if len(parts) < 2 or parts[0] != "sessions":
                raise KeyError(parts)
            table = self.tables[parts[1]]
            command = parts[2] if len(parts) > 2 else None
            if method == "GET" and command is None:
                return table.state()
            if method == "POST" and command == "action":
                table.act(payload)
                return table.state()
            if method == "POST" and command == "next":
                table.next_hand()
                return table.state()
            if method == "POST" and command == "rebuy":
                table.rebuy()
                return table.state()
            if method == "POST" and command == "hint":
                return {"lines": table.hint()}
            if method == "POST" and command == "review":
                return {"lines": table.review()}
            raise KeyError(parts)


class _Handler(http.server.BaseHTTPRequestHandler):
    app: App

    def do_GET(self):
        if not self._local_host():
            return
        if self.path in ASSETS:
            name, kind = ASSETS[self.path]
            self._send(200, (STATIC / name).read_bytes(), kind)
        elif self.path.startswith("/api/"):
            self._api("GET")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if not self._local_host():
            return
        if not self.path.startswith("/api/"):
            self._send(404, b"not found", "text/plain")
        elif self.headers.get_content_type() != "application/json":
            # Browsers send JSON across sites only after a preflight this server never grants.
            self._json(415, {"error": "the request body must be JSON"})
        else:
            self._api("POST")

    def _local_host(self) -> bool:
        """Serve only requests addressed to this machine by name, so a web page on another host
        cannot reach the table through a name that resolves here (DNS rebinding)."""
        assert isinstance(self.server, http.server.HTTPServer)
        port = self.server.server_port
        if self.headers.get("Host") in (f"127.0.0.1:{port}", f"localhost:{port}"):
            return True
        self._json(403, {"error": "the table only serves 127.0.0.1"})
        return False

    def _api(self, method: str):
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if not 0 <= length <= MAX_BODY:
                self._json(413 if length > MAX_BODY else 400, {"error": "bad request length"})
                return
            payload = json.loads(self.rfile.read(length) or b"{}") if method == "POST" else {}
            if not isinstance(payload, dict):
                raise WebError("the request body must be a JSON object")
            self._json(200, self.app.handle(method, [p for p in self.path[5:].split("/") if p], payload))
        except KeyError:
            self._json(404, {"error": "no such session or command"})
        except (ValueError, ArithmeticError, TypeError) as error:
            self._json(400, {"error": str(error)})
        except Exception as error:  # noqa: BLE001 - answer rather than drop the connection
            self._json(500, {"error": f"internal error: {type(error).__name__}"})

    def _json(self, status: int, payload: dict[str, Any]):
        self._send(status, json.dumps(payload).encode(), "application/json")

    def _send(self, status: int, body: bytes, kind: str):
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any):  # noqa: A002 - the base class's name
        """Quiet: the table is local and every request would otherwise be printed."""


def serve(
    port: int = 8000, log_dir: Path | None = None, profile: Profile = PROFILES["pc"]
) -> http.server.ThreadingHTTPServer:
    """A server for the web table on this machine only; call `serve_forever` to run it."""
    handler = type("Handler", (_Handler,), {"app": App(log_dir, profile)})
    return http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="bin/run_thpoker.py web", description="Play in a browser on this machine.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-dir", type=Path, default=default_log_dir())
    parser.add_argument("--no-log", action="store_true")
    add_device_argument(parser)
    args = parser.parse_args(argv)
    server = serve(args.port, None if args.no_log else args.log_dir, pick_profile(args.device))
    print(f"Open http://127.0.0.1:{args.port} in a browser (Ctrl-C stops the table).")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()
    return 0
