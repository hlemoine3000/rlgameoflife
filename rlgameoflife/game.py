import logging
import random
import tqdm

from rlgameoflife import entities
from rlgameoflife import events
from rlgameoflife import math_utils


class World:
    def __init__(self) -> None:
        self.logger = logging.getLogger(__class__.__name__)

        self.boundaries = math_utils.Vector2D(1000.0, 1000.0)
        self.creature_group = entities.EntityGroup(
            [
                entities.Creature(
                    math_utils.Vector2D(100, 100), math_utils.Vector2D(1.0, 0)
                )
            ]
        )
        self.food_group = entities.EntityGroup(
            [entities.Food(math_utils.Vector2D(500, 500), math_utils.Vector2D(1.0, 0))]
        )
        self.entities_group = entities.EntityGroup(
            [self.entities_group, self.food_group]
        )

        self._tick_counter = events.TickCounter()
        self.tick_events = events.TickEvents()
        self.tick_events.set_tick_event(events.EventType.SPAWN_FOOD_EVENT, 60 * 10)

    def spawn_food(self) -> None:
        self.food_group.add(
            entities.Food(
                math_utils.Vector2D(
                    random.randint(5, self.boundaries.x - 5),
                    random.randint(5, self.boundaries.y - 5),
                )
            )
        )

    def events(self) -> None:
        for event in self.tick_events.get():
            if event == events.EventType.SPAWN_FOOD_EVENT:
                self.logger.info("Spawning food.")
        self.tick_events.update()

    def update_groups(self) -> None:
        self.entities_group.update()
    
    def simulate(self, total_ticks: int):
        self._tick_counter.reset()
        pbar = tqdm.tqdm(range(total_ticks))
        for tick in pbar:
            self.events()
            # move
            self.update_groups()
