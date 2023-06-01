import logging

from rlgameoflife import collider
from rlgameoflife import entities
from rlgameoflife import visual_pattern

import numpy as np


INFINITE_DISTANCE = 1000000


class Mover:
    def __init__(self, target_group: entities.EntityGroup) -> None:
        self._logger = logging.getLogger(__class__.__name__)
        self._target_group = target_group
        self._before_move_collider_group = collider.ColliderGroup()
        self._after_move_collider_group = collider.ColliderGroup()

    def _move(self, all_group: entities.EntityGroup) -> None:
        self._logger.warning("not implemented.")
        pass

    def move(self, all_group: entities.EntityGroup) -> None:
        self._before_move_collider_group.collide(all_group)
        self._move(all_group)
        self._after_move_collider_group.collide(all_group)


class SimpleCreatureMover(Mover):
    def __init__(self, target_group: entities.EntityGroup) -> None:
        super().__init__(target_group)
        self._logger = logging.getLogger(__class__.__name__)
        self._before_move_collider_group.add(
            collider.CreatureFoodCollider(target_group)
        )

    def _move(self, all_group: entities.EntityGroup) -> None:
        for creature in self._target_group:
            # Get nearest food.
            nearest_food_distance = INFINITE_DISTANCE
            for food_idx, food in enumerate(all_group):
                if type(food) is entities.EntityGroup:
                    # Recursively get the food.
                    self._move(food)
                    continue
                if food.entity_type != entities.EntityType.FOOD:
                    # This is no food.
                    continue
                food_vector = creature.position.subtract(food.position)
                food_distance = food_vector.magnitude()
                if food_distance < nearest_food_distance:
                    nearest_food_distance = food_distance
                    nearest_food_vector = food_vector
                    nearest_food_idx = food_idx
            if nearest_food_distance == INFINITE_DISTANCE:
                # No food in all_group
                continue
            if nearest_food_distance < 5:
                # Creature eat the food when near.
                all_group.kill(nearest_food_idx)
                return
            # Creature move to nearest food.
            creature.move(nearest_food_vector)


class SimpleVisualCreatureMover(Mover):
    def __init__(self, target_group: entities.EntityGroup) -> None:
        super().__init__(target_group)
        self._logger = logging.getLogger(__class__.__name__)
        self._before_move_collider_group.add(
            collider.CreatureFoodCollider(target_group)
        )

    def _move(self, all_group: entities.EntityGroup):
        for creature in self._target_group:
            # Search for food.
            creature_vision = visual_pattern.VisualConePattern(np.pi/2, 1000.0, 9)
            creature_vision.update(creature, all_group)
            nearest_entity_distance = creature_vision.nearest_entity_distance(
                entities.EntityType.FOOD
            )
            if nearest_entity_distance == creature_vision.view_range:
                # There is no food in view field.
                # Search for food around.
                creature.rotate(0.02)
                continue
            nearest_food_quandrant_angle = -creature_vision.nearest_entity_angle(
                entity_type=entities.EntityType.FOOD
            )
            mov = creature.direction.rotate(nearest_food_quandrant_angle).scale(2.)
            creature.move(mov)
