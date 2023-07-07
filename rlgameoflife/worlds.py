from dataclasses import dataclass
import datetime
import json
import logging
import os
import random
import tqdm
import typing

import numpy as np

from rlgameoflife import actions
from rlgameoflife import collider
from rlgameoflife import entities
from rlgameoflife import events
from rlgameoflife import math_utils
from rlgameoflife import mover
from rlgameoflife import visual_pattern


@dataclass
class AgentParameters:
    observation: np.array
    reward: float
    terminated: bool
    truncated: bool
    info: dict


class BaseWorld:
    _movers: typing.List[mover.Mover]
    observation_shape: tuple[int, ...]

    def __init__(
        self,
        total_ticks: int,
        output_dir: str,
        boundaries: typing.Tuple[float, float] = (1000, 1000),
        disable_history: bool = False,
    ) -> None:
        self._logger = logging.getLogger(__class__.__name__)

        self._total_ticks = total_ticks
        self._history = entities.EntitiesHistoryLoader(output_dir, disable=disable_history)
        self._boundaries = math_utils.Vector2D(boundaries[0], boundaries[1])

        self._entities_group = entities.EntityGroup([], "all_entities_group")
        self._movers = []

        # Set up events
        self._tick = 0
        self._tick_events = events.TickEvents()

        self._init()

    def _init(self) -> None:
        self._logger.warning("_init not implemented.")
    
    def disable_history(self) -> None:
        self._history.disabled = True
    
    def enable_history(self) -> None:
        self._history.disabled = False

    def reset(self) -> None:
        self._tick = 0
        self._entities_group = entities.EntityGroup([], "all_entities_group")
        self._movers = []
        self._tick_events.reset()
        self._history.reset()
        self._init()

    def add_entities_group(self, entities_group: entities.EntityGroup) -> None:
        self._entities_group.add(entities_group)

    def add_mover(self, mv: mover.Mover):
        self._movers.append(mv)

    def add_tick_event(self, event_type: events.EventType, trigger_tick: int) -> None:
        self._tick_events.add_tick_event(event_type, trigger_tick)

    def tick_events_actions(self, event: events.EventType) -> None:
        if self._tick == 0:
            self._logger.warning("events action not implemented.")
        pass

    def events(self) -> None:
        for event in self._tick_events.get():
            self.tick_events_actions(event)
        self._tick_events.update()

    def update_groups(self) -> None:
        self._entities_group.update()

    def move(self) -> None:
        for mov in self._movers:
            mov.move(self._entities_group)

    def save_history(self) -> None:
        self._history.save()

    def save_parameters(self) -> None:
        parameters_filepath = os.path.join(self._output_dir, "parameters.json")
        parameters_dict = {
            "total_ticks": self._total_ticks,
            "boundaries": {"x": self._boundaries.x, "y": self._boundaries.y},
        }
        self._logger.info(f"Save simulation parameters at {parameters_filepath}")
        os.makedirs(self._output_dir, exist_ok=True)
        with open(parameters_filepath, "w") as parameters_file:
            json.dump(parameters_dict, parameters_file, indent=4)

    def save_simulation(self) -> None:
        self.save_parameters()
        self.save_history()

    def simulate(self):
        self._tick = 0
        pbar = tqdm.tqdm(range(self._total_ticks))
        for tick in pbar:
            self._tick = tick
            self.events()
            self.move()
            self.update_groups()

        self.save_simulation()
        self._logger.info("Simulation complete.")

    def agent_actions(self, step_actions: actions.Actions) -> AgentParameters:
        if self._tick == 0:
            self._logger.warning("not implemented.")
        return actions.Actions()

    def step(self, step_actions: actions.Actions) -> AgentParameters:
        agent_parameters = self.agent_actions(step_actions)
        self.events()
        self.move()
        self.update_groups()
        self._tick += 1
        return agent_parameters


class BasicWorld(BaseWorld):
    def __init__(
        self,
        total_ticks: int,
        output_dir: str,
        boundaries: typing.Tuple[int, int] = (1000, 1000),
    ) -> None:
        super().__init__(total_ticks, output_dir, boundaries)

        # Set up events
        self.add_tick_event(events.EventType.SPAWN_FOOD_EVENT, 200)

    def _init(self) -> None:
        # Create initial entities
        self.creature_group = entities.EntityGroup(
            [
                entities.Creature(
                    math_utils.Vector2D(100, 100),
                    math_utils.Vector2D(1.0, 0),
                    0,
                    self._history,
                )
            ],
            "creature_group",
        )
        self.add_entities_group(self.creature_group)
        self.food_group = entities.EntityGroup(
            [
                entities.Food(math_utils.Vector2D(500, 500), 0, self._history),
                entities.Food(math_utils.Vector2D(500, 400), 0, self._history),
                entities.Food(math_utils.Vector2D(500, 300), 0, self._history),
            ],
            "food_group",
        )
        self.add_entities_group(self.food_group)

        # Set up movers
        self.add_mover(mover.SimpleVisualCreatureMover(self.creature_group))
        
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


class BasicAgentWorld(BaseWorld):
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

        # Set up events
        self.add_tick_event(events.EventType.SPAWN_FOOD_EVENT, 20)

    def _init(self) -> None:
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
    ) -> AgentParameters:
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

        return AgentParameters(
            observation=self.get_observation(),
            reward=reward,
            terminated=False,
            truncated=False,
            info={},
        )
