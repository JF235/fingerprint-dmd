import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="dmd",
        description="Dense Minutia Descriptor CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("extract", add_help=False, help="Extract DMD templates from fingerprint images")
    sub.add_parser("match", add_help=False, help="Run identification/verification experiment")
    sub.add_parser("plot", add_help=False, help="Visualize extraction and match results")

    args, remaining = parser.parse_known_args()

    if args.command == "extract":
        from dmd.commands.extract import run
        run(remaining)
    elif args.command == "match":
        from dmd.commands.match import run
        run(remaining)
    elif args.command == "plot":
        from dmd.commands.plot import run
        run(remaining)
