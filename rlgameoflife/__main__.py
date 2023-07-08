import argparse
import logging
import os

import tqdm

from rlgameoflife import worlds
from rlgameoflife import visualisation
from rlgameoflife import agent
from rlgameoflife import optuna_trainer


DEFAULT_OUTPUT_DIRECTORY = "outputs"


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
        default=DEFAULT_OUTPUT_DIRECTORY,
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
    parser.add_argument(
        "-l", "--last", help="Visualize last simulation.", action="store_true"
    )
    parser.add_argument("-t", "--train", help="Train agents.", action="store_true")
    parser.add_argument("-p", "--optuna", help="Train agents with optuna optimization.", action="store_true")

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

    if args.train:
        agent_trainer = agent.AgentTrainer(agent.AgentTrainerParameters())
        agent_trainer.train()
        agent_trainer.evaluate()
        return
    
    if args.optuna:
        agent_trainer = optuna_trainer.OptunaAgentTrainer()
        agent_trainer.optimize()
        return

    if args.simulate:
        my_world = worlds.BasicWorld(args.iterations, args.output)
        my_world.simulate()

    sim_dir = None
    if args.last:
        dir_list = os.listdir(DEFAULT_OUTPUT_DIRECTORY)
        dir_list.sort()

        for dir in dir_list:
            if os.path.isdir(os.path.join(DEFAULT_OUTPUT_DIRECTORY, dir)):
                sim_dir = os.path.join(DEFAULT_OUTPUT_DIRECTORY, dir)
                continue
        if not sim_dir:
            main_logger.warning("latest simulation not found.")
    elif args.visualize:
        sim_dir = args.visualize
    if sim_dir:
        my_vis = visualisation.Visualizer(sim_dir)
        my_vis.make_video()


main()
