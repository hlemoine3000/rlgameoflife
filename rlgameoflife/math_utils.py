import numpy as np


class Vector2D:
    def __init__(self, x: float = 0., y: float = 0.) -> None:
        self.vector = np.array([x, y])

    @property
    def x(self) -> float:
        return self.vector[0]

    @property
    def y(self) -> float:
        return self.vector[1]

    def __repr__(self) -> str:
        return f"Vector2D({self.vector[0]}, {self.vector[1]})"

    def magnitude(self) -> float:
        return np.linalg.norm(self.vector)

    def normalize(self):
        return Vector2D(*(self.vector / self.magnitude()))

    def scale(self, length):
        return Vector2D(*(self.vector * length / self.magnitude()))

    def add(self, other):
        return Vector2D(*(self.vector + other.vector))

    def subtract(self, other):
        return Vector2D(*(self.vector - other.vector))

    def dot(self, other) -> float:
        return np.dot(self.vector, other.vector)

    def cross(self, other) -> float:
        return self.vector[0] * other.vector[1] - self.vector[1] * other.vector[0]

    def angle_between(self, other):
        dot = np.dot(self.vector, other.vector)
        det = np.linalg.det([self.vector, other.vector])
        angle = np.arctan2(det, dot)
        if angle < -np.pi:
            angle += 2 * np.pi
        elif angle > np.pi:
            angle -= 2 * np.pi
        return angle

    def rotate(self, angle):
        c, s = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([[c, -s], [s, c]])
        return Vector2D(*(np.dot(rotation_matrix, self.vector)))
