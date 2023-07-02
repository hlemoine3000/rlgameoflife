import copy
import enum
import logging
import numpy as np
import os
import typing

from rlgameoflife import math_utils


class ZeroDirectionVectorException(Exception):
    """raised when direction vector is zero"""


class EnittyIndexer:
    def __init__(self) -> None:
        self._index = 0

    @property
    def index(self) -> int:
        _index = self._index
        self._index += 1
        return _index


ENTITY_INDEXER = EnittyIndexer()


class EntityType(enum.Enum):
    NOTHING = -1
    FOOD = 0
    CREATURE = 1


class EntitiesHistoryLoader:
    def __init__(self, output_dir: str) -> None:
        self._history_npd = {}  # Dictionary to store history of each entity
        self._output_dir = output_dir

    def add(
        self,
        entity_name: str,
        tick: int,
        pos: math_utils.Vector2D,
        dir: math_utils.Vector2D,
        entity_type: EntityType,
    ) -> None:
        history_np = np.array([tick, *pos.vector, *dir.vector, entity_type.value])
        if entity_name not in self._history_npd:
            self._history_npd[entity_name] = history_np[np.newaxis, :]
        else:
            if tick in self._history_npd[entity_name][:, 0]:
                raise Exception("tick %i has already been added")
            self._history_npd[entity_name] = np.vstack(
                (self._history_npd[entity_name], history_np)
            )

    def save(self) -> None:
        os.makedirs(self._output_dir, exist_ok=True)
        with open(
            os.path.join(self._output_dir, f"entities_history.npz"), "wb"
        ) as entity_file:
            np.savez_compressed(entity_file, **self._history_npd)

    def load(self, filepath: str) -> None:
        with open(filepath, "rb") as f:
            npz_file = np.load(f)
            for entity_name in npz_file.files:
                self._history_npd[entity_name] = npz_file[entity_name]

    def get_history(self, entity_name: str):
        return self._history_npd.get(entity_name, None)

    def get_total_ticks(self) -> int:
        total_ticks = 0
        for entity_np in self._history_npd.values():
            total_ticks = int(np.amax(entity_np[:, 0], initial=total_ticks))
        return total_ticks

    def get_timed_history(self) -> tuple[dict, tuple[int, int, int, int]]:
        total_ticks = self.get_total_ticks()
        timed_history_dict = {}
        max_x, max_y, min_x, min_y = 0, 0, 10000, 10000
        for tick in range(total_ticks + 1):
            timed_history_dict[tick] = {}
            for entity_name, entity_np in self._history_npd.items():
                event_np = entity_np[entity_np[:, 0] == tick]
                if len(event_np) == 0:
                    continue
                if len(event_np) != 1:
                    raise Exception("Multiple position logged for tick %i", tick)
                event_np = np.squeeze(event_np)
                timed_history_dict[tick][entity_name] = {
                    "position": event_np[1:3],
                    "direction": event_np[3:5],
                    "type": event_np[5],
                }
                max_x = max(max_x, event_np[1])
                max_y = max(max_y, event_np[2])
                min_x = min(min_x, event_np[1])
                min_y = min(min_y, event_np[2])
        return timed_history_dict, (min_x, min_y, max_x, max_y)


class EntityObject:
    def __init__(
        self,
        pos: math_utils.Vector2D,
        dir: math_utils.Vector2D,
        name: str,
        entity_type: EntityType,
    ) -> None:
        self._logger = logging.getLogger(__class__.__name__)
        self._name = name
        self._entity_type = entity_type
        if dir.magnitude() == 0.0:
            self._logger.error("direction vector cannot be [0, 0]")
            raise ZeroDirectionVectorException
        self._direction = dir.normalize()
        self._position = pos
    
    def __str__(self):
        return f"EntityObject('{self._name}', {self._entity_type}, {self._position}, {self._direction})"

    def __repr__(self):
        return f"EntityObject('{self._name}', {self._entity_type}, {self._position}, {self._direction})"

    @property
    def name(self):
        return self._name

    @property
    def entity_type(self) -> EntityType:
        return self._entity_type

    @property
    def position(self) -> math_utils.Vector2D:
        return self._position

    @property
    def direction(self) -> math_utils.Vector2D:
        return self._direction

    def update(self) -> None:
        self._logger.warning(f"not implemented")
        pass


class BaseEntity(EntityObject):
    def __init__(
        self,
        pos: math_utils.Vector2D,
        dir: math_utils.Vector2D,
        tick: int,
        history: EntitiesHistoryLoader,
        entity_type: EntityType = EntityType.FOOD,
        max_speed: float = 1.0,
        max_angle: float = 0.02,  # radians
        name: str = "entity",
    ) -> None:
        super().__init__(pos, dir, name, entity_type)
        self._logger = logging.getLogger(__class__.__name__)

        self._next_position = pos
        self._next_direction = dir
        self._max_speed = max_speed
        self.max_angle = max_angle

        # Initialise entity history at tick -1
        self._history = history
        self._history.add(self._name, tick - 1, pos, dir, entity_type)
        self._tick = tick

    def rotate(self, angle: float) -> None:
        """Rotate the entity by the given angle in radians."""
        # Check max angle
        if abs(angle) > self.max_angle:
            angle = self.max_angle if angle > 0 else -self.max_angle
        # Calculate direction
        self._next_direction = self._direction.rotate(angle).normalize()

    def move(self, mov: math_utils.Vector2D) -> None:
        """Move the entity by the given movement vector."""
        movement = copy.copy(mov)
        # Check max speed.
        movement_distance = mov.magnitude()
        if movement_distance > self._max_speed:
            movement_distance = self._max_speed

        # Check max angle
        movement_angle = movement.angle_between(self._direction)
        self.rotate(movement_angle)

        # Calculate movement
        self._next_position = self._position.add(
            self._next_direction.scale(movement_distance)
        )

    def flush_move(self) -> None:
        """Flush the movement to the entity."""
        self._next_position = self._position
        self._next_direction = self._direction

    def update(self) -> None:
        """Update the entity position and direction with the next position and direction."""
        self._position = self._next_position
        self._direction = self._next_direction
        self._history.add(
            self._name,
            self._tick,
            self._position,
            self._direction,
            self._entity_type,
        )
        self._tick += 1


class Creature(BaseEntity):
    def __init__(
        self,
        pos: math_utils.Vector2D,
        dir: math_utils.Vector2D,
        tick: int,
        history: EntitiesHistoryLoader,
    ) -> None:
        global ENTITY_INDEXER
        name = f"creature_{ENTITY_INDEXER.index}"
        super().__init__(
            pos,
            dir,
            tick,
            history,
            entity_type=EntityType.CREATURE,
            max_speed=2.0,
            max_angle=0.02,
            name=name,
        )


class Food(BaseEntity):
    def __init__(
        self, pos: math_utils.Vector2D, tick: int, history: EntitiesHistoryLoader
    ) -> None:
        global ENTITY_INDEXER
        name = f"food_{ENTITY_INDEXER.index}"
        super().__init__(
            pos,
            math_utils.Vector2D(1.0, 0.0),
            tick,
            history,
            entity_type=EntityType.FOOD,
            max_speed=0.0,
            max_angle=0.02,
            name=name,
        )


class EntityGroup(EntityObject):
    def __init__(self, entity_list: typing.List[EntityObject], name: str) -> None:
        super().__init__(
            math_utils.Vector2D(),
            math_utils.Vector2D(1.0, 0.0),
            name,
            EntityType.NOTHING,
        )
        self._logger = logging.getLogger(__class__.__name__)

        for entity in entity_list:
            if not self._valid_entity(entity):
                return
        self._entity_list = entity_list
        self._name = name

    def __str__(self):
        return f"EntityGroup('{self.name}', {self._entity_list})"

    def __repr__(self):
        return f"EntityGroup('{self.name}', {self._entity_list})"

    def __iter__(self):
        self._num_entity = len(self._entity_list)
        self._entity_idx = 0
        return self

    def __next__(self) -> EntityObject:
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

    def kill(self, entity_idx: int) -> None:
        self._entity_list.pop(entity_idx)

    def kills(self, entities_idx: list) -> None:
        for entity_idx in entities_idx:
            self.kill(entity_idx)
