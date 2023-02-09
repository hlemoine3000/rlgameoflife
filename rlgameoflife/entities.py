import copy
import enum
import logging
import numpy as np
import os

from rlgameoflife import math_utils


CREATURE_INDEX = 0
FOOD_INDEX = 0


class EntityType(enum.Enum):
    NOTHING = 1
    ITEM = 2
    CREATURE = 3


class EntityState(enum.Enum):
    DEAD = 1
    ALIVE = 2


class EntityObject:
    def __init__(self, name: str) -> None:
        self._logger = logging.getLogger(__class__.__name__)
        self._name = name
    
    @property
    def name(self):
        return self._name
    
    def update(self) -> None:
        self._logger.warning(f'not implemented')
        pass

    def save_history(self, output_dir: str) -> None:
        self._logger.warning(f'not implemented')
        pass


class BaseEntity(EntityObject):
    def __init__(
        self,
        pos: math_utils.Vector2D,
        dir: math_utils.Vector2D,
        tick: int,
        entity_type: EntityType = EntityType.NOTHING,
        entiy_state: EntityState = EntityState.ALIVE,
        max_speed: float = 1.0,
        max_angle: float = 0.02,  # radians
        name: str = "entity",
    ) -> None:
        super().__init__(name)
        self._logger = logging.getLogger(__class__.__name__)

        self._entity_type = entity_type
        self._entiy_state = entiy_state
        self._direction = dir.normalize()
        self._position = pos
        self._next_position = pos
        self._max_speed = max_speed
        self.max_angle = max_angle

        # Initialise entity history at tick -1
        self._tick = tick - 1
        self._history = self._history_array()
        self._tick = tick

    @property
    def position(self) -> math_utils.Vector2D:
        return self._position

    @property
    def direction(self) -> math_utils.Vector2D:
        return self._direction

    def _history_array(self) -> np.array:
        return np.concatenate(
            (
                self._tick,
                self._position.vector,
                self._direction.vector,
                np.array([self._entity_type, self._entiy_state]),
            )
        )

    def move(self, mov: math_utils.Vector2D) -> None:
        movement = copy.copy(mov)
        # Check max speed.
        movement_distance = mov.magnitude()
        if movement_distance > self._max_speed:
            movement_distance = self._max_speed

        # Check max angle
        movement_angle = movement.angle_between(self._direction)
        if abs(movement_angle) > self.max_angle:
            movement_angle = self.max_angle if movement_angle > 0 else -self.max_angle

        # Calculate movement
        self._direction = self._direction.rotate(movement_angle).normalize()
        self._next_position = self._position.add(self._direction.scale(movement_distance))

    def flush_move(self) -> None:
        self._next_position = self._position

    def update(self) -> None:
        self._position = self._next_position
        self._history = np.vstack((self._history, self._history_array()))
        self._tick += 1
    
    def save_history(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os .path.join(output_dir, f'{self._name}.npy')
        with open(filepath, 'wb') as entity_file:
            np.save(entity_file, self._history)


class Creature(BaseEntity):
    def __init__(self, pos: math_utils.Vector2D, dir: math_utils.Vector2D, tick: int) -> None:
        global CREATURE_INDEX
        name = f'creature_{CREATURE_INDEX}'
        CREATURE_INDEX += 1
        super().__init__(
            pos,
            dir,
            tick,
            entity_type=EntityType.CREATURE,
            entiy_state=EntityState.ALIVE,
            max_speed=2.0,
            max_angle=0.02,
            name=name
        )


class Food(BaseEntity):
    def __init__(self, pos: math_utils.Vector2D, tick: int) -> None:
        global FOOD_INDEX
        name = f'food_{FOOD_INDEX}'
        super().__init__(
            pos,
            math_utils.Vector2D(1., 0.),
            tick,
            entity_type=EntityType.ITEM,
            entiy_state=EntityState.ALIVE,
            max_speed=0.0,
            max_angle=0.02,
            name=name
        )


class EntityGroup(EntityObject):
    def __init__(self, entity_list: list, name: str) -> None:
        super().__init__(name)
        self._logger = logging.getLogger(__class__.__name__)

        for entity in entity_list:
            if not self._valid_entity(entity):
                return
        self._entity_list = entity_list
        self._name = name

    def __iter__(self):
        self._num_entity = len(self._entity_list)
        self._entity_idx = 0
        return self

    def __next__(self) -> BaseEntity:
        if self._entity_idx >= self._num_entity:
            raise StopIteration
        entity = self._entity_list[self._entity_idx]
        self._entity_idx += 1
        return entity

    def __len__(self) -> int:
        return len(self._entity_list)
    
    def __getitem__(self, item) -> EntityObject:
        if type(item) is int:
            return self._entity_list[item]
        if type(item) is str:
            for entity in self._entity_list:
                if item == entity.name:
                    return entity
            raise KeyError("Name {} not found in entities".format(item))
        else:
            raise TypeError("Cannot index with {}".format(type(item)))

    def keys(self):
        return [entity.name for entity in self._entity_list]

    def _valid_entity(self, entity) -> bool:
        if not issubclass(type(entity), EntityObject):
            self._logger.error(
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

    def save_history(self, output_dir: str) -> None:
        for entity in self._entity_list:
            entity.save_history(output_dir)