#!/usr/bin/env python3
"""Measure each bot tier's reference loss per 100 hands when its own play in the user's seat is
reviewed (DESIGN.md Section 7.7): `bin/rating_calibration.py [hands] [workers]`."""

import argparse


from   thpoker.analysis.rating_calibration \
                                import calibrate # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hands", type=int, nargs="?", default=200, help="hands per tier (default 200)")
    parser.add_argument("workers", type=int, nargs="?", default=4, help="processes (default 4)")
    args = parser.parse_args()
    for tier, (per_100, stderr, count) in calibrate(args.hands, args.workers).items():
        print(
            f"tier {tier}: {per_100:.1f} bb lost per 100 hands against the reference"
            f" (standard error {stderr:.1f}, {count} hands)"
        )


if __name__ == "__main__":
    main()
