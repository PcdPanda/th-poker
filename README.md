# Hold'em trainer

An offline No-Limit Hold'em trainer: play against bots of three difficulty tiers, heads-up or at tables of up to 8, in cash games or single-table tournaments, then review your decisions against a strong reference. DESIGN.md is the full specification.

Needs Python 3.10 or newer and numpy. Install it with `pip install .` (or `make install`), which adds a `thpoker` command; from a checkout, `pip install -r requirements.txt` and then `python bin/run_thpoker.py` do the same without installing the package.

## Play

thpoker                                    # 6-max cash game, Tier 2 bots
thpoker --seats 2 --tier 3                 # heads-up against the strongest bots
thpoker --mode tournament --preset turbo   # a single-table tournament
thpoker --hide-styles                      # guess the bots' styles as you play
thpoker web                                # the same table in a browser
                                           # (on a phone: run it in Termux or Pydroid)

At the table: `f` fold, `x` check, `c` call, `b 50%` / `b 300` / `b 3bb` / `b 2.5x` bet or raise, `a` all-in, `h` coach hint, `?` legal actions, `q` quit. After each hand, `r` reviews it and `p` asks for your estimates first.

## Learn

thpoker review ~/.poker_trainer/sessions/session-SEED.jsonl # top mistakes
thpoker review LOG --hand 12 --grids                        # one hand in full
thpoker review LOG --quiz                                   # predict, then reveal
thpoker drill preflop                                       # also pushfold, thresholds, icm
thpoker drill mistakes                                      # your reviewed mistakes again
thpoker leaks                                               # patterns across sessions
thpoker progress                                            # EV lost per 100 hands
thpoker calibration                                         # how good your estimates are

Reviews, hints, and the mistake drill time this machine to pick how much work to do; on a slow phone pass `--device phone` (or `--device pc` to force the full budget).

## Development

make test       # pytest, from the repository root
make check      # ruff, mypy, and the tests
make data       # rebuild the three shipped tables in src/python/thpoker/data
make develop    # pip install -e .