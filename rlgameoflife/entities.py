import enum
import logging
import numpy as np

from rlgameoflife import math_utils


class EntityType(enum.Enum):
    NOTHING = 1
    ITEM = 2
    CREATURE = 3


class EntityState(enum.Enum):
    DEAD = 1
    ALIVE = 2


class BaseEntity:
    def __init__(
        self,
        pos: math_utils.Vector2D,
        dir: math_utils.Vector2D,
        entity_type: EntityType = EntityType.NOTHING,
        entiy_state: EntityState = EntityState.ALIVE,
        max_speed: float = 1.0,
        max_angle: float = 0.02,  # radians
    ) -> None:
        self.logger = logging.getLogger(__class__.__name__)

        self.entity_type = entity_type
        self.entiy_state = entiy_state
        self.direction = dir.normalize()
        self.position = pos
        self.next_position = pos
        self.max_speed = max_speed
        self.max_angle = max_angle

        self.history = np.concatenate(
            (
                self.position.vector,
                self.direction.vector,
                np.array([self.entity_type, self.entiy_state]),
            )
        )
        print(self.history)

    def move(self, mov: math_utils.Vector2D) -> None:
        movement = mov.copy()
        # Check max speed.
        movement_distance = mov.magnitude()
        if movement_distance > self.max_speed:
            movement_distance = self.max_speed

        # Check max angle
        movement_angle = movement.angle_between(self.direction)
        if abs(movement_angle) > self.max_angle:
            movement_angle = self.max_angle if movement_angle > 0 else -self.max_angle

        # Calculate movement
        self.direction = self.direction.rotate(movement_angle).normalize()
        self.next_position += self.direction.scale(movement_distance)

    def flush_move(self) -> None:
        self.next_position = self.position

    def update(self) -> None:
        self.position = self.next_position


class Creature(BaseEntity):
    def __init__(self, pos: math_utils.Vector2D, dir: math_utils.Vector2D) -> None:
        super().__init__(
            pos,
            dir,
            entity_type=EntityType.CREATURE,
            entiy_state=EntityState.ALIVE,
            max_speed=2.0,
            max_angle=0.02,
        )


class Food(BaseEntity):
    def __init__(self, pos: math_utils.Vector2D, dir: math_utils.Vector2D) -> None:
        super().__init__(
            pos,
            dir,
            entity_type=EntityType.ITEM,
            entiy_state=EntityState.ALIVE,
            max_speed=0.0,
            max_angle=0.02,
        )


class EntityGroup:
    def __init__(self, entity_list: list, group_name: str) -> None:
        self.logger = logging.getLogger(__class__.__name__)

        for entity in entity_list:
            if not self._valid_entity(entity):
                return
        self._entity_list = entity_list
        self._group_name = group_name

    def _valid_entity(self, entity) -> bool:
        if type(entity) is not BaseEntity or type(entity) is not EntityGroup:
            self.logger.error(
                "Entity must be of BaseEntity or EntityGroup type. Not {}".format(
                    type(entity)
                )
            )
            return False
        return True

    def add(self, entity) -> None:
        if not self._valid_entity(entity):
            return
        self._entity_list.append(entity)

    def update(self) -> None:
        for entity in self._entity_list:
            entity.update()
