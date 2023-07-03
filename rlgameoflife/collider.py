import logging
import typing

from rlgameoflife import entities


class Collider:
    def __init__(self, target_group: entities.EntityGroup) -> None:
        self._logger = logging.getLogger(__class__.__name__)
        self._target_group = target_group

    def collide(self, all_group: entities.EntityGroup) -> float:
        self._logger.warning("not implemented.")
        pass


class CreatureFoodCollider(Collider):
    def collide(self, all_group: entities.EntityGroup) -> float:
        reward = 0.0
        for target_entity in self._target_group:
            entities_to_kill = []
            for polled_entity_idx, polled_entity in enumerate(all_group):
                if type(polled_entity) is entities.EntityGroup:
                    # Recursively calculate collision with entities.
                    reward += self.collide(polled_entity)
                    continue
                if polled_entity.entity_type != entities.EntityType.FOOD:
                    # This is no food!
                    continue
                entities_distance = target_entity.position.subtract(
                    polled_entity.position
                ).magnitude()
                if entities_distance < 5.0:
                    self._logger.debug("Collision between %s and %s", target_entity, polled_entity)
                    reward += 1.0
                    entities_to_kill.append(polled_entity_idx)
            all_group.kills(entities_to_kill)
        return reward


class ColliderGroup:
    def __init__(self, collider_list: typing.List[Collider] = []) -> None:
        self._collider_list = collider_list

    def add(self, coll: Collider) -> None:
        self._collider_list.append(coll)

    def collide(self, all_group: entities.EntityGroup) -> None:
        for coll in self._collider_list:
            coll.collide(all_group)
