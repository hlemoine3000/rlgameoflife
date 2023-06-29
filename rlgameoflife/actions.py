import enum
import random


class BadActionTypeException(Exception):
    """Raise when action is not the correct type."""


class Actions(enum.Enum):
    """Base class for actions."""


class DiscreteMoveActions(Actions):
    """Discrete move actions."""
    STAY = 0
    ROTATE_RIGHT = 1
    ROTATE_LEFT = 2
    FORWARD = 3

def sample(action_space: Actions) -> int:
    return random.choice(list(action_space)).value