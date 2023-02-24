import math
import numpy as np

from rlgameoflife import entities


class VisualConePattern:
    def __init__(
        self,
        arc_angle: float,
        arc_radius: float,
        num_sensor: int,
        num_entities_type: int,
    ) -> None:
        self._arc_angle = arc_angle
        self._arc_half_angle = self._arc_angle / 2
        self._num_sensor = num_sensor
        self._quadrant_angle_width = self._arc_angle / self._num_sensor
        self._arc_radius = arc_radius
        self._num_entities_type = num_entities_type
        self._visual_pattern = np.zeros((self._num_sensor, self._num_entities_type))

    @property
    def visual_pattern(self) -> np.array:
        return self._visual_pattern

    def reset(self) -> None:
        self._visual_pattern = np.zeros((self._num_sensor, self._num_entities_type))

    def update(
        self, ref_entity: entities.BaseEntity, sample_entity: entities.EntityObject
    ) -> None:
        if type(sample_entity) is entities.EntityGroup:
            for entity in sample_entity:
                self.update(ref_entity, entity)
            return
        referenced_sample_pos = sample_entity.position.subtract(ref_entity.position)
        referenced_sample_distance = referenced_sample_pos.magnitude()
        if referenced_sample_distance > self._arc_radius:
            # sample entity too far.
            return
        referenced_sample_angle = ref_entity.direction.angle_between(
            referenced_sample_pos
        )
        if abs(referenced_sample_angle) > self._arc_half_angle:
            # sample entity not in field of view.
            return
        abs_referenced_sample_angle = referenced_sample_angle + self._arc_half_angle
        visual_quadrant = math.floor(
            abs_referenced_sample_angle / self._quadrant_angle_width
        )
        visual_value = referenced_sample_distance / self._arc_radius
        if visual_value < np.amax(self._visual_pattern[visual_quadrant]):
            # This entity is nearer than the previous one.
            # Clear this quadrant so it is set after.
            self._visual_pattern[visual_quadrant] = np.zeros((self._num_entities_type))
        if np.amax(self._visual_pattern[visual_quadrant]) == 0.:
            self._visual_pattern[visual_quadrant, sample_entity.entity_type.value] = (
                referenced_sample_distance / self._arc_radius
            )
