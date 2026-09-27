#!/usr/bin/env python3
"""Rebuild a shipped table in thpoker/data: `bin/generate_data.py pushfold` (also preflop_ranges
and preflop_equity). The same inputs give the same bytes, so an unchanged model leaves no diff."""

import argparse
import gzip
import json
import os
from   pathlib                  import Path
from   thpoker.charts           import PREFLOP_RANGES, PUSHFOLD_CHARTS
from   thpoker.generators       import preflop_equity, preflop_ranges, pushfold
from   thpoker.odds             import PREFLOP_TABLE # noqa: E402
import time
from   typing                   import Any

# Per solved table: the generator, where it is written, and where its gain is measured.
SOLVED = {
    "pushfold": (pushfold, PUSHFOLD_CHARTS, ""),
    "preflop_ranges": (preflop_ranges, PREFLOP_RANGES, "at a first-in decision "),
}


def write(path: Path, table: str, data: dict[str, Any]):
    text = json.dumps({"generated_by": f"bin/generate_data.py {table}", **data}, separators=(",", ":"))
    # No timestamp in the file: the same data gives the same bytes.
    path.write_bytes(gzip.compress((text + "\n").encode(), mtime=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    tables = parser.add_subparsers(dest="table", required=True)
    for table, (module, _, _) in SOLVED.items():
        solved = tables.add_parser(table, help=module.__doc__.splitlines()[0])
        solved.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    equity = tables.add_parser("preflop_equity", help=preflop_equity.__doc__.splitlines()[0])
    equity.add_argument("--boards", type=int, default=10_000)
    equity.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    started = time.monotonic()
    if args.table == "preflop_equity":
        data = preflop_equity.generate(args.boards, args.seed)
        write(PREFLOP_TABLE, args.table, data)
        print(
            f"wrote {PREFLOP_TABLE} from {args.boards} boards in "
            f"{time.monotonic() - started:.0f}s; largest standard error {data['stderr']}"
        )
        return
    module, path, gain = SOLVED[args.table]
    data = module.generate(args.workers)
    write(path, args.table, data)
    print(
        f"wrote {path} in {time.monotonic() - started:.0f}s; largest best-response gain "
        f"'{gain}{data['best_response_gain_bb']}' bb per hand"
    )


if __name__ == "__main__":
    main()
