from dataclasses import dataclass

import torch.nn.functional as F
import torch.nn as nn


@dataclass
class DQNParameters:
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


class DQN(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, output_size)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return F.tanh(self.layer3(x))