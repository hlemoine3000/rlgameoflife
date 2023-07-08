import random
import typing

import numpy as np

from rlgameoflife import actions
from rlgameoflife import collider
from rlgameoflife import entities
from rlgameoflife import events
from rlgameoflife import math_utils
from rlgameoflife import visual_pattern

from . import base_world


class BasicAgentWorld(base_world.BaseWorld):
    def __init__(
        self,
        total_ticks: int,
        output_dir: str,
        boundaries: typing.Tuple[int, int] = (100, 100),
        disable_history: bool = False,
    ) -> None:
        super().__init__(total_ticks, output_dir, boundaries, disable_history)
        
        self.agent_vision = visual_pattern.VisualConePattern(np.pi / 2, 1000.0, 9)
        self.observation_shape = self.agent_vision.shape
        self.action_space = actions.DiscreteMoveActions
    
    def _initialize(self) -> None:
        self.add_tick_event(events.EventType.SPAWN_FOOD_EVENT, 20)

    def _reinitialize(self) -> None:
        # Create initial entities
        self.food_group = entities.EntityGroup(
            [
                entities.Food(math_utils.Vector2D(50, 60), 0, self._history),
                entities.Food(math_utils.Vector2D(50, 50), 0, self._history),
                entities.Food(math_utils.Vector2D(50, 40), 0, self._history),
            ],
            "food_group",
        )
        self.add_entities_group(self.food_group)

        self.agent = entities.Creature(
            math_utils.Vector2D(10.0, 50.0),
            math_utils.Vector2D(1.0, 0.0),
            0,
            self._history,
        )
        self.agent_group = entities.EntityGroup([self.agent], "agent-group")
        self.add_entities_group(self.agent_group)
        self.agent_collider = collider.CreatureFoodCollider(self.agent_group)

    def spawn_food(self) -> None:
        self.food_group.add(
            entities.Food(
                math_utils.Vector2D(
                    random.randint(5, self._boundaries.x - 5),
                    random.randint(5, self._boundaries.y - 5),
                ),
                self._tick,
                self._history,
            )
        )

    def tick_events_actions(self, event: events.EventType) -> None:
        if event == events.EventType.SPAWN_FOOD_EVENT:
            self.spawn_food()

    def get_observation(self) -> np.ndarray:
        self.agent_vision.reset()
        self.agent_vision.update(self.agent, self._entities_group)
        return self.agent_vision.visual_pattern.flatten()

    def agent_actions(
        self, step_actions: actions.DiscreteMoveActions
    ) -> base_world.AgentParameters:
        if type(step_actions) is not actions.DiscreteMoveActions:
            self._logger.error("step action bad type %s", type(step_actions))
            raise actions.BadActionTypeException()
        reward = self.agent_collider.collide(self._entities_group)
        if step_actions == actions.DiscreteMoveActions.FORWARD:
            self.agent.move(self.agent.direction)
        elif step_actions == actions.DiscreteMoveActions.ROTATE_RIGHT:
            self.agent.rotate(-0.1)
        elif step_actions == actions.DiscreteMoveActions.ROTATE_LEFT:
            self.agent.rotate(0.1)

        return base_world.AgentParameters(
            observation=self.get_observation(),
            reward=reward,
            terminated=False,
            truncated=False,
            info={},
        )


class BasicEvalWorldAgent(BasicAgentWorld):
    def _initialize(self) -> None:
        pass
    def _reinitialize(self) -> None:
        # Create initial entities
        self.food_group = entities.EntityGroup(
            [
                entities.Food(math_utils.Vector2D(40, 60), 0, self._history),
                entities.Food(math_utils.Vector2D(40, 40), 0, self._history),
                entities.Food(math_utils.Vector2D(60, 40), 0, self._history),
                entities.Food(math_utils.Vector2D(60, 60), 0, self._history),
                entities.Food(math_utils.Vector2D(80, 40), 0, self._history),
                entities.Food(math_utils.Vector2D(80, 60), 0, self._history),
            ],
            "food_group",
        )
        self.add_entities_group(self.food_group)

        self.agent = entities.Creature(
            math_utils.Vector2D(10.0, 50.0),
            math_utils.Vector2D(1.0, 0.0),
            0,
            self._history,
        )
        self.agent_group = entities.EntityGroup([self.agent], "agent-group")
        self.add_entities_group(self.agent_group)
        self.agent_collider = collider.CreatureFoodCollider(self.agent_group)