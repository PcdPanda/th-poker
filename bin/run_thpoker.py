#!/usr/bin/env python3
"""Play, review, and drill: `bin/run_thpoker.py` starts a quick cash game, `bin/run_thpoker.py web`
serves the table in a browser, and review, drill, leaks, progress, and calibration are
subcommands (`bin/run_thpoker.py review --help`)."""

import sys
from   thpoker.cli              import main # noqa: E402
from   thpoker.web              import main as web_main # noqa: E402

if __name__ == "__main__":
    sys.exit(web_main(sys.argv[2:]) if sys.argv[1:2] == ["web"] else main())
