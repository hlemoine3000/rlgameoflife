from collections import namedtuple, deque
from dataclasses import dataclass
import logging
import math
import random

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from rlgameoflife import worlds
from rlgameoflife import actions
from rlgameoflife import models


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


@dataclass
class AgentTrainerParameters:
    batch_size: int = (
        128  # BATCH_SIZE is the number of transitions sampled from the replay buffer
    )
    gamma: float = (
        0.99  # GAMMA is the discount factor as mentioned in the previous section
    )
    eps_start: float = 0.9  # is the starting value of epsilon
    eps_end: float = 0.05  # is the final value of epsilon
    eps_decay: int = 1000  # controls the rate of exponential decay of epsilon, higher means a slower decay
    tau: float = 0.005  # is the update rate of the target network
    lr: float = 1e-4  # is the learning rate of the AdamW optimizer
    num_episodes: int = 60
    max_steps_per_episode: int = 400
    eval_each_n_episode: int = 5
    replay_memory_size: int = 10000


class WorldParameters:
    max_ticks: int = 400
    boundaries: tuple[int, int] = (100, 100)


class AgentTrainer:
    def __init__(self, agent_parameters: AgentTrainerParameters) -> None:
        self._logger = logging.getLogger(__class__.__name__)
        self.world_parameters = WorldParameters()
        self.hyperparameters = agent_parameters

        self.world = worlds.BasicAgentWorld(
            self.world_parameters.max_ticks,
            "outputs",
            self.world_parameters.boundaries,
            disable_history=True,
        )
        self.eval_world = worlds.BasicEvalWorldAgent(
            self.world_parameters.max_ticks,
            "outputs",
            self.world_parameters.boundaries,
            disable_history=True,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get number of actions from gym action space
        n_actions = len(self.world.action_space)
        # Get the number of state observations
        observation = self.world.get_observation()
        n_observations = observation.shape[0]

        self.policy_net = models.DQN(n_observations, n_actions).to(self.device)
        self.target_net = models.DQN(n_observations, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=self.hyperparameters.lr, amsgrad=True
        )
        self.memory = ReplayMemory(self.hyperparameters.replay_memory_size)

        self.steps_done = 0

    def _select_action(self, state):
        return self.policy_net(state).max(1)[1].view(1, 1)

    def select_action(self, state, training: bool = True):
        if not training:
            return self._select_action(state)
        sample = random.random()
        eps_threshold = self.hyperparameters.eps_end + (
            self.hyperparameters.eps_start - self.hyperparameters.eps_end
        ) * math.exp(-1.0 * self.steps_done / self.hyperparameters.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self._select_action(state)
        else:
            return torch.tensor(
                [[actions.sample(self.world.action_space)]],
                device=self.device,
                dtype=torch.long,
            )

    def optimize_model(self) -> float:
        if len(self.memory) < self.hyperparameters.batch_size:
            return
        transitions = self.memory.sample(self.hyperparameters.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(
            self.hyperparameters.batch_size, device=self.device
        )
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(
                non_final_next_states
            ).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (
            next_state_values * self.hyperparameters.gamma
        ) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        return loss.item()

    def train(self, save_final_eval: bool = True):
        episode_bar = tqdm(range(self.hyperparameters.num_episodes))
        eval_rewards = None
        for episode in episode_bar:
            # Initialize the environment and get it's state
            state = self.world.get_observation()
            state = torch.tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            self.policy_net.train()
            for t in range(self.hyperparameters.max_steps_per_episode):
                action = self.select_action(state)
                step_parameters = self.world.step(
                    self.world.action_space(action.item())
                )
                reward = torch.tensor([step_parameters.reward], device=self.device)
                # done = step_parameters.terminated or step_parameters.truncated

                if step_parameters.terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(
                        step_parameters.observation,
                        dtype=torch.float32,
                        device=self.device,
                    ).unsqueeze(0)

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[
                        key
                    ] * self.hyperparameters.tau + target_net_state_dict[key] * (
                        1 - self.hyperparameters.tau
                    )
                self.target_net.load_state_dict(target_net_state_dict)

                episode_bar.set_description(
                    f"Train | Episode {episode} | Step {t} / {self.hyperparameters.max_steps_per_episode} | Eval reward {eval_rewards}"
                )

            if (
                self.hyperparameters.eval_each_n_episode > 0
                and episode % self.hyperparameters.eval_each_n_episode == 0
                and episode != self.hyperparameters.num_episodes - 1
            ):
                eval_rewards = self.evaluate()
            self.world.reset()
        final_rewards = self.evaluate(save_history=save_final_eval)
        self._logger.info("Training complete.")
        return final_rewards

    def evaluate(self, save_history: bool = False) -> int:
        if save_history:
            self.eval_world.enable_history()
        else:
            self.eval_world.disable_history()
        self.eval_world.reset()

        # Initialize the environment and get it's state
        state = self.eval_world.get_observation()
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(
            0
        )
        episode_reward = 0
        self.policy_net.eval()
        for t in range(self.hyperparameters.max_steps_per_episode):
            with torch.no_grad():
                action = self.select_action(state, training=False)
            step_parameters = self.eval_world.step(
                self.eval_world.action_space(action.item())
            )
            # done = step_parameters.terminated or step_parameters.truncated

            if step_parameters.terminated:
                next_state = None
            else:
                next_state = torch.tensor(
                    step_parameters.observation,
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)

            # Move to the next state
            state = next_state
            episode_reward += step_parameters.reward

        self.eval_world.save_history()
        return episode_reward
