import argparse
import logging

import tqdm

from rlgameoflife import game
from rlgameoflife import visualisation


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def argument_parser():
    """
    Create a parser for main rlgameoflife.
    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(description="Train agents in simulation.")
    parser.add_argument(
        "-i", "--iterations", help="Number of iterations to run.", default=400, type=int
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output Directory to store the simulation artifacts.",
        default="outputs",
    )
    parser.add_argument(
        "-s",
        "--simulate",
        help="Launch simulation and train agents.",
        action="store_true",
    )
    parser.add_argument("-d", "--debug", help="Enable debug logs.", action="store_true")
    parser.add_argument(
        "-v",
        "--visualize",
        help="Create a video from a simulation history.",
        default=None,
    )

    return parser


def main():
    args = argument_parser().parse_args()

    main_logger = logging.getLogger()
    logging_level = logging.DEBUG if args.debug else logging.INFO
    main_logger.setLevel(logging_level)
    tqdm_handler = TqdmLoggingHandler()
    tqdm_handler.setFormatter(
        logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    )
    main_logger.addHandler(tqdm_handler)
    if args.simulate:
        my_world = game.World(args.iterations, args.output)
        my_world.simulate()

    if args.visualize:
        my_vis = visualisation.Visualizer(args.visualize)
        my_vis.make_gif()


main()
