import os
import tempfile
import unittest
import numpy as np

from rlgameoflife import entities
from rlgameoflife import math_utils


class EntitiesHistoryLoaderTestCase(unittest.TestCase):
    def setUp(self):
        self.output_dir = tempfile.mkdtemp()
        self.loader = entities.EntitiesHistoryLoader(self.output_dir)
        self.entity_name = "entity1"
        self.tick = 1
        self.pos = math_utils.Vector2D(1, 2)
        self.dir = math_utils.Vector2D(3, 4)
        self.entity_type = entities.EntityType.CREATURE

    def test_add(self):
        self.loader.add(self.entity_name, self.tick, self.pos, self.dir, self.entity_type)
        history_np = self.loader.get_history(self.entity_name)
        self.assertIsNotNone(history_np)
        self.assertIsInstance(history_np, np.ndarray)
        self.assertEqual(history_np.shape, (1, 6))
        self.assertEqual(history_np[0, 0], self.tick)
        self.assertEqual(history_np[0, 1], self.pos.x)
        self.assertEqual(history_np[0, 2], self.pos.y)
        self.assertEqual(history_np[0, 3], self.dir.x)
        self.assertEqual(history_np[0, 4], self.dir.y)
        self.assertEqual(history_np[0, 5], self.entity_type.value)

    def test_save_and_load(self):
        self.loader.add(self.entity_name, self.tick, self.pos, self.dir, self.entity_type)
        self.loader.save()
        new_loader = entities.EntitiesHistoryLoader(self.output_dir)
        new_loader.load(os.path.join(self.output_dir, "entities_history.npz"))
        history_np = new_loader.get_history(self.entity_name)
        self.assertIsNotNone(history_np)
        self.assertIsInstance(history_np, np.ndarray)
        self.assertEqual(history_np.shape, (1, 6))
        self.assertEqual(history_np[0, 0], self.tick)
        self.assertEqual(history_np[0, 1], self.pos.x)
        self.assertEqual(history_np[0, 2], self.pos.y)
        self.assertEqual(history_np[0, 3], self.dir.x)
        self.assertEqual(history_np[0, 4], self.dir.y)
        self.assertEqual(history_np[0, 5], self.entity_type.value)

    def test_get_timed_history(self):
        self.loader.add(self.entity_name, 1, math_utils.Vector2D(1, 2), math_utils.Vector2D(3, 4), entities.EntityType.CREATURE)
        self.loader.add(self.entity_name, 2, math_utils.Vector2D(5, 6), math_utils.Vector2D(7, 8), entities.EntityType.ITEM)
        timed_history = self.loader.get_timed_history()
        self.assertEqual(len(timed_history), 3)
        self.assertIn(1, timed_history)
        self.assertIn(self.entity_name, timed_history[1])
        np.testing.assert_array_equal(timed_history[1][self.entity_name]["position"], np.array([1, 2]))
        np.testing.assert_array_equal(timed_history[1][self.entity_name]["direction"], np.array([3, 4]))
        np.testing.assert_array_equal(timed_history[1][self.entity_name]["type"], np.array([entities.EntityType.CREATURE.value]))
        self.assertIn(2, timed_history)
        self.assertIn(self.entity_name, timed_history[2])
        np.testing.assert_array_equal(timed_history[2][self.entity_name]["position"], np.array([5, 6]))
        np.testing.assert_array_equal(timed_history[2][self.entity_name]["direction"], np.array([7, 8]))
        np.testing.assert_array_equal(timed_history[2][self.entity_name]["type"], np.array([entities.EntityType.ITEM.value]))
