import datetime
import json
import logging
import os
import random
import tqdm

from rlgameoflife import entities
from rlgameoflife import events
from rlgameoflife import math_utils
from rlgameoflife import mover


class World:
    def __init__(self, total_ticks: int, output_dir: str) -> None:
        self._logger = logging.getLogger(__class__.__name__)

        self.total_ticks = total_ticks
        now = datetime.datetime.now()
        self._output_dir = os.path.join(output_dir, now.strftime("%m%d%Y%H%M%S"))
        self._history = entities.EntitiesHistoryLoader(self._output_dir)
        self.boundaries = math_utils.Vector2D(1000.0, 1000.0)

        # Create Initial entities.
        self.creature_group = entities.EntityGroup(
            [
                entities.Creature(
                    math_utils.Vector2D(100, 100),
                    math_utils.Vector2D(1.0, 0),
                    0,
                    self._history,
                )
            ],
            "creature_group",
        )
        self.food_group = entities.EntityGroup(
            [
                entities.Food(math_utils.Vector2D(500, 500), 0, self._history),
                entities.Food(math_utils.Vector2D(500, 400), 0, self._history),
                entities.Food(math_utils.Vector2D(500, 300), 0, self._history),
            ],
            "food_group",
        )
        self.entities_group = entities.EntityGroup(
            [self.creature_group, self.food_group], "all_entities_group"
        )

        # Set up events
        self._tick = 0
        self.tick_events = events.TickEvents()
        self.tick_events.set_tick_event(events.EventType.SPAWN_FOOD_EVENT, 200)

        # Set up movers
        self._movers = [mover.SimpleVisualCreatureMover(self.creature_group)]

    def spawn_food(self) -> None:
        self.food_group.add(
            entities.Food(
                math_utils.Vector2D(
                    random.randint(5, self.boundaries.x - 5),
                    random.randint(5, self.boundaries.y - 5),
                ),
                self._tick,
                self._history,
            )
        )

    def events(self) -> None:
        for event in self.tick_events.get():
            if event == events.EventType.SPAWN_FOOD_EVENT:
                self.spawn_food()
        self.tick_events.update()

    def update_groups(self) -> None:
        self.entities_group.update()

    def move(self) -> None:
        for mov in self._movers:
            mov.move(self.entities_group)

    def save_history(self) -> None:
        self._logger.info(f"Save simulation history at {self._output_dir}")
        self._history.save()

    def save_parameters(self) -> None:
        parameters_filepath = os.path.join(self._output_dir, "parameters.json")
        parameters_dict = {
            "total_ticks": self.total_ticks,
            "boundaries": {"x": self.boundaries.x, "y": self.boundaries.y},
        }
        self._logger.info(f"Save simulation parameters at {parameters_filepath}")
        os.makedirs(self._output_dir, exist_ok=True)
        with open(parameters_filepath, "w") as parameters_file:
            json.dump(parameters_dict, parameters_file, indent=4)

    def save_simulation(self) -> None:
        self.save_parameters()
        self.save_history()

    def simulate(self):
        self._tick = 0
        pbar = tqdm.tqdm(range(self.total_ticks))
        for tick in pbar:
            self._tick = tick
            self.events()
            self.move()
            self.update_groups()

        self.save_simulation()
        self._logger.info("Simulation complete.")
