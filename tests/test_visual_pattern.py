import numpy as np
import unittest
from parameterized import parameterized

from rlgameoflife import entities
from rlgameoflife import math_utils
from rlgameoflife import visual_pattern


class VisualConePatternTestCase(unittest.TestCase):
    def setUp(self):
        self.arc_angle = np.pi
        self.arc_radius = 100
        self.num_sensor = 5
        self.visual_cone_pattern = visual_pattern.VisualConePattern(
            self.arc_angle, self.arc_radius, self.num_sensor
        )

    @parameterized.expand(
        [
            (
                "foodInFieldOfView",
                entities.Creature(
                    math_utils.Vector2D(0, 0),
                    math_utils.Vector2D(1, 0),
                    0,
                    entities.EntitiesHistoryLoader("/tmp"),
                ),
                entities.Food(
                    math_utils.Vector2D(50, 0),
                    0,
                    entities.EntitiesHistoryLoader("/tmp"),
                ),
                np.array([[1, 1], [1, 1], [0.5, 1], [1, 1], [1, 1]]),
            ),
            (
                "foodInEdgeFieldOfView",
                entities.Creature(
                    math_utils.Vector2D(0, 0),
                    math_utils.Vector2D(1, 0),
                    0,
                    entities.EntitiesHistoryLoader("/tmp"),
                ),
                entities.Food(
                    math_utils.Vector2D(0, 50),
                    0,
                    entities.EntitiesHistoryLoader("/tmp"),
                ),
                np.array([[1, 1], [1, 1], [1, 1], [1, 1], [0.5, 1]]),
            ),
            (
                "creatureInFieldOfView",
                entities.Creature(
                    math_utils.Vector2D(0, 0),
                    math_utils.Vector2D(1, 0),
                    0,
                    entities.EntitiesHistoryLoader("/tmp"),
                ),
                entities.Creature(
                    math_utils.Vector2D(50, 0),
                    math_utils.Vector2D(1, 0),
                    0,
                    entities.EntitiesHistoryLoader("/tmp"),
                ),
                np.array([[1, 1], [1, 1], [1, 0.5], [1, 1], [1, 1]]),
            ),
            (
                "foodNotInFieldOfView",
                entities.Creature(
                    math_utils.Vector2D(0, 0),
                    math_utils.Vector2D(1, 0),
                    0,
                    entities.EntitiesHistoryLoader("/tmp"),
                ),
                entities.Food(
                    math_utils.Vector2D(-50, 0),
                    0,
                    entities.EntitiesHistoryLoader("/tmp"),
                ),
                np.ones((5, 2)),
            ),
            (
                "foodToFar",
                entities.Creature(
                    math_utils.Vector2D(0, 0),
                    math_utils.Vector2D(1, 0),
                    0,
                    entities.EntitiesHistoryLoader("/tmp"),
                ),
                entities.Food(
                    math_utils.Vector2D(200, 0),
                    0,
                    entities.EntitiesHistoryLoader("/tmp"),
                ),
                np.ones((5, 2)),
            ),
            (
                "multipleFoodInFieldOfView",
                entities.Creature(
                    math_utils.Vector2D(0, 0),
                    math_utils.Vector2D(1, 0),
                    0,
                    entities.EntitiesHistoryLoader("/tmp"),
                ),
                entities.EntityGroup(
                    [
                        entities.Food(
                            math_utils.Vector2D(75, 0),
                            0,
                            entities.EntitiesHistoryLoader("/tmp"),
                        ),
                        entities.Food(
                            math_utils.Vector2D(50, 0),
                            0,
                            entities.EntitiesHistoryLoader("/tmp"),
                        ),
                    ],
                    "food_test_group",
                ),
                np.array([[1, 1], [1, 1], [0.5, 1], [1, 1], [1, 1]]),
            ),
            (
                "creatureAndFoodInFieldOfView",
                entities.Creature(
                    math_utils.Vector2D(0, 0),
                    math_utils.Vector2D(1, 0),
                    0,
                    entities.EntitiesHistoryLoader("/tmp"),
                ),
                entities.EntityGroup(
                    [
                        entities.Creature(
                            math_utils.Vector2D(75, 0),
                            math_utils.Vector2D(1, 0),
                            0,
                            entities.EntitiesHistoryLoader("/tmp"),
                        ),
                        entities.Food(
                            math_utils.Vector2D(50, 0),
                            0,
                            entities.EntitiesHistoryLoader("/tmp"),
                        ),
                    ],
                    "food_test_group",
                ),
                np.array([[1, 1], [1, 1], [0.5, 1], [1, 1], [1, 1]]),
            ),
        ]
    )
    def test_update(self, name, entity1, entity2, expected):
        self.visual_cone_pattern.update(entity1, entity2)
        np.testing.assert_array_equal(self.visual_cone_pattern.visual_pattern, expected)
