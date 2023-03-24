import datetime
import json
import logging
import os
import random
import tqdm
from typing import Tuple

from rlgameoflife import entities
from rlgameoflife import events
from rlgameoflife import math_utils
from rlgameoflife import mover


class BaseWorld:
    def __init__(self, total_ticks: int, output_dir: str, boundaries: Tuple[float, float] = (1000, 1000)) -> None:
        self._logger = logging.getLogger(__class__.__name__)

        self._total_ticks = total_ticks
        now = datetime.datetime.now()
        self._output_dir = os.path.join(output_dir, now.strftime("%m%d%Y%H%M%S"))
        self._history = entities.EntitiesHistoryLoader(self._output_dir)
        self._boundaries = math_utils.Vector2D(boundaries[0], boundaries[1])

        self._entities_group = entities.EntityGroup([], "all_entities_group")
        self._movers = []

        # Set up events
        self._tick = 0
        self._tick_events = events.TickEvents()
        
    
    def add_entities_group(self, entities_group: entities.EntityGroup) -> None:
        self._entities_group.add(entities_group)
    
    def add_mover(self, mv: mover.Mover):
        self._movers.append(mv)

    def add_tick_event(self, event_type: events.EventType, trigger_tick: int) -> None:
        self._tick_events.add_tick_event(event_type, trigger_tick)

    def tick_events_actions(self, event: events.EventType) -> None:
        if self._tick == 0:
            self._logger.warning("events action not implemented.")
        pass

    def events(self) -> None:
        for event in self._tick_events.get():
            self.tick_events_actions(event)
        self._tick_events.update()

    def update_groups(self) -> None:
        self._entities_group.update()

    def move(self) -> None:
        for mov in self._movers:
            mov.move(self._entities_group)

    def save_history(self) -> None:
        self._logger.info(f"Save simulation history at {self._output_dir}")
        self._history.save()

    def save_parameters(self) -> None:
        parameters_filepath = os.path.join(self._output_dir, "parameters.json")
        parameters_dict = {
            "total_ticks": self._total_ticks,
            "boundaries": {"x": self._boundaries.x, "y": self._boundaries.y},
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
        pbar = tqdm.tqdm(range(self._total_ticks))
        for tick in pbar:
            self._tick = tick
            self.events()
            self.move()
            self.update_groups()

        self.save_simulation()
        self._logger.info("Simulation complete.")


class BasicWorld(BaseWorld):
    def __init__(self, total_ticks: int, output_dir: str, boundaries: Tuple[int, int] = (1000, 1000)) -> None:
        super().__init__(total_ticks, output_dir, boundaries)
        
        # Create initial entities
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
        self.add_entities_group(self.creature_group)
        self.add_entities_group(self.food_group)
        
        # Set up movers
        self.add_mover(mover.SimpleVisualCreatureMover(self.creature_group))

        # Set up events
        self.add_tick_event(events.EventType.SPAWN_FOOD_EVENT, 200)

    def spawn_food(self) -> None:
        self.food_group.add(
            entities.Food(
                math_utils.Vector2D(
                    random.randint(5, self._boundaries.x - 5),
                    random.randint(5, self._boundaries.y - 5),
                ),
                self._tick,
                self._history,
            )
        )
    
    def tick_events_actions(self, event: events.EventType) -> None:
        if event == events.EventType.SPAWN_FOOD_EVENT:
            self.spawn_food()
        
