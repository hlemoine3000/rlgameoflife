import numpy as np


class Vector2D:
    def __init__(self, x: float, y: float) -> None:
        self.vector = np.array([x, y])

    @property
    def x(self):
        return self.vector[0]

    @property
    def y(self):
        return self.vector[1]

    def __repr__(self):
        return f"Vector2D({self.vector[0]}, {self.vector[1]})"

    def magnitude(self):
        return np.linalg.norm(self.vector)

    def normalize(self):
        return Vector2D(*(self.vector / self.magnitude()))

    def scale(self, length):
        return Vector2D(*(self.vector * length / self.magnitude()))

    def add(self, other):
        return Vector2D(*(self.vector + other.vector))

    def subtract(self, other):
        return Vector2D(*(self.vector - other.vector))

    def dot(self, other):
        return np.dot(self.vector, other.vector)

    def cross(self, other):
        return self.vector[0] * other.vector[1] - self.vector[1] * other.vector[0]

    def angle_between(self, other):
        return np.arccos(
            np.clip(
                np.dot(self.normalize().vector, other.normalize().vector), -1.0, 1.0
            )
        )

    def rotate(self, angle):
        c, s = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([[c, -s], [s, c]])
        return Vector2D(*(np.dot(rotation_matrix, self.vector)))
