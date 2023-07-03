import unittest

from parameterized import parameterized

from rlgameoflife import entities
from rlgameoflife import collider
from rlgameoflife import math_utils


class TestCreatureFoodCollider(unittest.TestCase):
    def setUp(self):
        self.history = entities.EntitiesHistoryLoader("tmp")

        self.food_group = entities.EntityGroup(
            [
                entities.Food(math_utils.Vector2D(50, 50), 0, self.history),
                entities.Food(math_utils.Vector2D(58, 50), 0, self.history),
            ],
            "food_group",
        )

        self.creature_group = entities.EntityGroup(
            [
                entities.Creature(
                    math_utils.Vector2D(30, 30),
                    math_utils.Vector2D(1.0, 0.0),
                    0,
                    self.history,
                )
            ],
            "creature_group",
        )

        self.all_group = entities.EntityGroup(
            [self.food_group, self.creature_group], "test_group"
        )  # initialize as per your definition
        self.creature_food_collider = collider.CreatureFoodCollider(self.creature_group)

    @parameterized.expand(
        [
            (math_utils.Vector2D(54, 50), 2.0),
            (math_utils.Vector2D(50, 50), 1.0),
            (math_utils.Vector2D(40, 50), 0.0),
        ]
    )
    def test_collide(self, target_position, expected):
        target_entity = entities.Creature(target_position, math_utils.Vector2D(1.0, 0.0), 0, self.history)
        target_group = entities.EntityGroup([target_entity], "target_group")
        creature_food_collider = collider.CreatureFoodCollider(target_group)
        got = creature_food_collider.collide(self.all_group)
        self.assertEqual(got, expected)
