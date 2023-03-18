import unittest
import numpy as np

from parameterized import parameterized

from rlgameoflife import math_utils


class TestMathUtils(unittest.TestCase):
    def test_init(self):
        v = math_utils.Vector2D(1, 2)
        self.assertEqual(v.vector[0], 1)
        self.assertEqual(v.vector[1], 2)

    def test_magnitude(self):
        v = math_utils.Vector2D(3, 4)
        self.assertEqual(v.magnitude(), 5.0)

    def test_normalize(self):
        v = math_utils.Vector2D(3, 4)
        n = v.normalize()
        self.assertEqual(n.magnitude(), 1.0)

    def test_scale(self):
        v = math_utils.Vector2D(1, 1)
        s = v.scale(5)
        self.assertEqual(s.magnitude(), 5.0)

    def test_add(self):
        v1 = math_utils.Vector2D(1, 2)
        v2 = math_utils.Vector2D(3, 4)
        a = v1.add(v2)
        self.assertEqual(a.vector[0], 4)
        self.assertEqual(a.vector[1], 6)

    def test_subtract(self):
        v1 = math_utils.Vector2D(3, 4)
        v2 = math_utils.Vector2D(1, 2)
        s = v1.subtract(v2)
        self.assertEqual(s.vector[0], 2)
        self.assertEqual(s.vector[1], 2)

    def test_dot(self):
        v1 = math_utils.Vector2D(1, 2)
        v2 = math_utils.Vector2D(3, 4)
        d = v1.dot(v2)
        self.assertEqual(d, 11.0)

    def test_cross(self):
        v1 = math_utils.Vector2D(1, 2)
        v2 = math_utils.Vector2D(3, 4)
        c = v1.cross(v2)
        self.assertEqual(c, -2.0)

    @parameterized.expand(
        [
            [1, 0, 0, 1, np.pi / 2],
            [1, 0, 1, 0, 0],
            [1, 1, -1, 1, np.pi / 2],
            [1, 0, 0, -1, -np.pi / 2],
            [1, 0, -1, 0, np.pi],
        ]
    )
    def test_angle_between(self, vx1, vy1, vx2, vy2, expected):
        v1 = math_utils.Vector2D(vx1, vy1)
        v2 = math_utils.Vector2D(vx2, vy2)
        angle = v1.angle_between(v2)
        self.assertAlmostEqual(angle, expected, places=6)

    @parameterized.expand(
        [
            (1, 0, np.pi / 2, 0, 1),
            (1, 1, np.pi / 4, 0, np.sqrt(2)),
            (1, 0, -np.pi / 2, 0, -1),
        ]
    )
    def test_rotate(self, vx, vy, angle, ex, ey):
        v = math_utils.Vector2D(vx, vy)
        r = v.rotate(angle)
        self.assertAlmostEqual(r.vector[0], ex, places=6)
        self.assertAlmostEqual(r.vector[1], ey, places=6)
