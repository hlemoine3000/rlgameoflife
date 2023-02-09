import datetime
import logging
import os
import random
import tqdm

from rlgameoflife import entities
from rlgameoflife import events
from rlgameoflife import math_utils


class World:
    def __init__(self, output_dir: str) -> None:
        self._logger = logging.getLogger(__class__.__name__)

        now = datetime.datetime.now()
        self._output_dir = os.path.join(output_dir, now.strftime("%m%d%Y%H%M%S"))

        self.boundaries = math_utils.Vector2D(1000.0, 1000.0)
        self.creature_group = entities.EntityGroup(
            [
                entities.Creature(
                    math_utils.Vector2D(100, 100), math_utils.Vector2D(1.0, 0), 0
                )
            ],
            "creature_group"
        )
        self.food_group = entities.EntityGroup(
            [entities.Food(math_utils.Vector2D(500, 500), 0)],
            "food_group"
        )
        self.entities_group = entities.EntityGroup(
            [self.creature_group, self.food_group],
            "all_entities_group"
        )

        self._tick = 0
        self.tick_events = events.TickEvents()
        self.tick_events.set_tick_event(events.EventType.SPAWN_FOOD_EVENT, 60 * 10)

    def spawn_food(self) -> None:
        self.food_group.add(
            entities.Food(
                math_utils.Vector2D(
                    random.randint(5, self.boundaries.x - 5),
                    random.randint(5, self.boundaries.y - 5),
                ), 0
            )
        )

    def events(self) -> None:
        for event in self.tick_events.get():
            if event == events.EventType.SPAWN_FOOD_EVENT:
                self._logger.info("Spawning food.")
        self.tick_events.update()

    def update_groups(self) -> None:
        self.entities_group.update()

    def creature_move(self) -> None:
        if len(self.food_group) == 0:
            return
        for creature in self.creature_group:
            nearest_food_distance = 10000
            for food in self.food_group:
                food_vector = creature.position.subtract(food.position)
                food_distance = food_vector.magnitude()
                if food_distance < nearest_food_distance:
                    nearest_food_distance = food_distance
                    nearest_food_vector = food_vector
            creature.move(nearest_food_vector)

    def move(self) -> None:
        self.creature_move()
    
    def save_history(self) -> None:
        history_output_dir = os.path.join(self._output_dir, 'history')
        self._logger.info(f'Save simulation history at {history_output_dir}')
        self.entities_group.save_history(history_output_dir)

    def simulate(self, total_ticks: int):
        self._tick = 0
        pbar = tqdm.tqdm(range(total_ticks))
        for tick in pbar:
            self._tick = tick
            self.events()
            self.move()
            self.update_groups()

        self.save_history()
        self._logger.info('Simulation complete.')