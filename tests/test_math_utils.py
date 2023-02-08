

import numpy as np
import pytest

from rlgameoflife import math_utils

def test_init():
    v = math_utils.Vector2D(1, 2)
    assert v.vector[0] == 1
    assert v.vector[1] == 2

def test_magnitude():
    v = math_utils.Vector2D(3, 4)
    assert v.magnitude() == 5.0

def test_normalize():
    v = math_utils.Vector2D(3, 4)
    n = v.normalize()
    assert n.magnitude() == 1.0

def test_scale():
    v = math_utils.Vector2D(1, 1)
    s = v.scale(5)
    assert s.magnitude() == 5.0

def test_add():
    v1 = math_utils.Vector2D(1, 2)
    v2 = math_utils.Vector2D(3, 4)
    a = v1.add(v2)
    assert a.vector[0] == 4
    assert a.vector[1] == 6

def test_subtract():
    v1 = math_utils.Vector2D(3, 4)
    v2 = math_utils.Vector2D(1, 2)
    s = v1.subtract(v2)
    assert s.vector[0] == 2
    assert s.vector[1] == 2

def test_dot():
    v1 = math_utils.Vector2D(1, 2)
    v2 = math_utils.Vector2D(3, 4)
    d = v1.dot(v2)
    assert d == 11.0

def test_cross():
    v1 = math_utils.Vector2D(1, 2)
    v2 = math_utils.Vector2D(3, 4)
    c = v1.cross(v2)
    assert c == -2.0

def test_angle_between():
    v1 = math_utils.Vector2D(1, 0)
    v2 = math_utils.Vector2D(0, 1)
    angle = v1.angle_between(v2)
    assert np.isclose(angle, np.pi / 2)

@pytest.mark.parametrize("vx, vy, angle, ex, ey", [
    (1, 0, np.pi/2, 0, 1),
    (1, 1, np.pi/4, 0, np.sqrt(2)),
    (1, 0, -np.pi/2, 0, -1),
])
def test_rotate(vx, vy, angle, ex, ey):
    v = math_utils.Vector2D(vx, vy)
    r = v.rotate(angle)
    assert np.isclose(r.vector[0], ex, atol=1e-6)
    assert np.isclose(r.vector[1], ey, atol=1e-6)