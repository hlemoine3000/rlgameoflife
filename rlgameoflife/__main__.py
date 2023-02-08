import argparse

from rlgameoflife import game


def argument_parser():
    """
    Create a parser for main rlgameoflife.
    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(description="Train agents in simulation.")
    parser.add_argument(
        "-t", "--train", help="Launch simulation and train agents.", action="store_true"
    )
    return parser


def main():
    args = argument_parser().parse_args()

    if args.train:
        game.game()


main()
