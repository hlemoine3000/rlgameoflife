import math
import numpy as np

from rlgameoflife import entities


class VisualConePattern:
    def __init__(self, arc_angle: float, arc_radius: float, num_sensor: int) -> None:
        self._arc_angle = arc_angle
        self._arc_half_angle = self._arc_angle / 2
        self._num_sensor = num_sensor
        self._quadrant_angle_width = self._arc_angle / self._num_sensor
        self._arc_radius = arc_radius
        self._num_entities_type = len(entities.EntityType) - 1
        self._visual_pattern = np.ones((self._num_sensor, self._num_entities_type))

    @property
    def visual_pattern(self) -> np.ndarray:
        return self._visual_pattern
    
    @property
    def view_range(self) -> float:
        return self._arc_radius
    
    @property
    def shape(self) -> tuple[int, ...]:
        return self._visual_pattern.shape

    def reset(self) -> None:
        self._visual_pattern = np.ones((self._num_sensor, self._num_entities_type))

    def update(
        self, ref_entity: entities.BaseEntity, sample_entity: entities.EntityObject
    ) -> None:
        if type(sample_entity) is entities.EntityGroup:
            for entity in sample_entity:
                self.update(ref_entity, entity)
            return
        if ref_entity.name == sample_entity.name:
            # Will not look for itself.
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
        # At the very edge of the field of view, will give an index out of range.
        visual_quadrant = (
            visual_quadrant - 1
            if visual_quadrant >= self._num_sensor
            else visual_quadrant
        )
        visual_value = referenced_sample_distance / self._arc_radius
        if visual_value < np.amin(self._visual_pattern[visual_quadrant]):
            # This entity is nearer than the previous one.
            # Clear this quadrant so it is set after.
            self._visual_pattern[visual_quadrant] = np.ones((self._num_entities_type))
        self._visual_pattern[visual_quadrant, sample_entity.entity_type.value] = (
            referenced_sample_distance / self._arc_radius
        )

    def nearest_entity_quadrant(
        self, entity_type: entities.EntityType = None
    ) -> np.array:
        if entity_type:
            return np.argmin(self._visual_pattern[:, entity_type.value])
        return np.argmin(self._visual_pattern, axis=1)

    def nearest_entity_distance(
        self, entity_type: entities.EntityType = None
    ) -> np.array:
        nearest_entity_quadrant = self.nearest_entity_quadrant(entity_type=entity_type)
        if entity_type:
            return (
                self._visual_pattern[nearest_entity_quadrant, entity_type.value]
                * self._arc_radius
            )
        return self._visual_pattern[nearest_entity_quadrant, :] * self._arc_radius

    def nearest_entity_angle(self, entity_type: entities.EntityType = None) -> np.array:
        nearest_entity_quadrant = self.nearest_entity_quadrant(entity_type=entity_type)
        return (
            nearest_entity_quadrant * self._quadrant_angle_width
            + self._quadrant_angle_width / 2
            - self._arc_half_angle
        )
