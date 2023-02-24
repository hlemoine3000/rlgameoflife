from rlgameoflife import entities


INFINITE_DISTANCE = 1000000

class BaseMover:
    def __init__(self, target_group: entities.EntityGroup) -> None:
        self._target_group = target_group


class SimpleCreatureMover(BaseMover):
    def __init__(self, target_group: entities.EntityGroup) -> None:
        super().__init__(target_group)

    def move(self, all_group: entities.EntityGroup):
        for creature in self._target_group:
            # Get nearest food.
            nearest_food_distance = INFINITE_DISTANCE
            for food_idx, food in enumerate(all_group):
                if type(food) is entities.EntityGroup:
                    # Recursively get the food.
                    self.move(food)
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
