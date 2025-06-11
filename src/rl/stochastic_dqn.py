"""stochastic_q_learning.py

Stochastic Deep Action-Value Learning (commonly called *Stochastic DQN*) for
finite **discrete-action** environments implemented in PyTorch.

This agent follows the algorithm of Fourati et al. (2024), which speeds up
bootstrapping by evaluating only a logarithmic-sized subset of actions at each
step instead of the entire action space.  Optionally, Double Action-Value
Learning is used to reduce positive bias by decoupling action selection and
target evaluation.

References:
    — Fourati, F., et al. (2024). *Stochastic DQN: Leveraging Sub-Action
      Selection for Efficient Deep Reinforcement Learning.*
      https://arxiv.org/abs/2405.10310
    — Canonical implementation:
      https://github.com/fouratifares/stochdqn/blob/main/rl/stochdqn.py

Example
-------
>>> import gymnasium as gym
>>> from stochastic_dqn import (
...     StochasticQLearningAgent,
...     StochasticQLearningConfig,
... )
>>> env = gym.make("Taxi-v3")
>>> cfg = StochasticQLearningConfig()
>>> agent = StochasticQLearningAgent(
...     state_size  = env.observation_space.n,
...     action_size = env.action_space.n,
...     index2action = {i: np.eye(env.action_space.n)[i]
...                 for i in range(env.action_space.n)},
...     config = cfg,        
... )
>>> # you would now call `agent.act` & `env.step` inside your training loop
"""

from src.rl.common import has_converged
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from math import ceil, inf, log2
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger

# ───────────────────────────── Configuration ─────────────────────────────
@dataclass(slots=True)
class StochasticQLearningConfig:
    """Container for all hyper-parameters required by
    :class:`StochasticQLearningAgent`.

    Attributes:
        hidden_size: Width of the two hidden layers.
        learning_rate: Learning-rate used by the Adam optimiser.
        discount_factor: Discount factor γ for future rewards.

        epsilon_start: Initial exploration probability ε.  See the
            *epsilon-greedy* strategy.
        epsilon_decay: Multiplicative decay applied to ε after every
            environment step.
        epsilon_min: Lower bound on ε.

        subset_coefficient: Controls how many actions are evaluated during the
            stochastic maximisation phase:  
            `subset_size = floor(subset_coefficient * ceil(log₂|A|))`.
        batch_multiplier: Mini-batch size is
            `batch_multiplier * ceil(log₂|A|)`.
        buffer_multiplier: Replay-buffer capacity is
            `buffer_multiplier * batch_size`.

        deterministic: If *True* the maximisation step evaluates **all**
            actions (ordinary Deep Action-Value Learning).  If *False* use the
            stochastic subset.
        double_dqn: Enable Double Action-Value Learning to reduce estimation
        target_update_every: Number of learning steps between hard copies of
            the online network into the target network (default 1000, as
            in the original Deep Action-Value paper).

        use_cuda: Attempt to run on a CUDA device when available.
    """

    hidden_size: int = 64
    learning_rate: float = 1e-3
    discount_factor: float = 0.99

    # Exploration schedule
    epsilon_start: float = 1.0
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01

    # Buffer and subset sizing
    subset_coefficient: float = 2.0
    batch_multiplier: float = 2.0
    buffer_multiplier: float = 2.0

    # Optimisation switches
    deterministic: bool = False
    double_dqn: bool = True
    target_update_every: int = 1_000

    use_cuda: bool = True

# ────────────────────────────── Helpers ────────────────────────────── #
def _to_tensor(array: np.ndarray | Sequence[float], *,
               device: torch.device) -> torch.Tensor:
    """Convert a NumPy array or Python sequence to a `float32` tensor.

    Args:
        array: Data to be converted.
        device: Device on which to allocate the resulting tensor.

    Returns:
        A `torch.Tensor` with data type `torch.float32` residing on *device*.
    """
    return torch.as_tensor(array, dtype=torch.float32, device=device)

# ──────────────────────── Neural-network building block ────────────────────────
class _StateActionNetwork(nn.Module):
    """A two-hidden-layer multilayer perceptron that predicts a single
    action-value estimate *Q(s, a)* from the concatenation of a state vector and
    a one-hot action vector.

    Args:
        input_dim: Dimensionality of the concatenated input *(|S| + |A|)*.
        hidden_dim: Width of each hidden layer.
        device: PyTorch device on which to place the network.
    """
    def __init__(self, input_dim: int, hidden_dim: int,
                 device: torch.device) -> None:
        super().__init__()
        self._fc1 = nn.Linear(input_dim, hidden_dim)
        self._fc2 = nn.Linear(hidden_dim, hidden_dim)
        self._head = nn.Linear(hidden_dim, 1)
        self.to(device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # (B, input_dim)
        """Compute forward pass.

        Args:
            inputs: Batch of concatenated state-action vectors.

        Returns:
            Predicted scalar action-value estimates shape *(B, 1)*.
        """
        hidden = torch.relu(self._fc1(inputs))
        hidden = torch.relu(self._fc2(hidden))
        return self._head(hidden)


# ──────────────────────────── Replay buffer ────────────────────────────
class _ReplayBuffer:
    """Fixed-size cyclic buffer used for experience replay.

    A transition is a five-tuple *(state, action_index, reward,
    next_state, done_flag)*.  When the buffer is full, the oldest
    experience is overwritten.

    Args:
        capacity: Maximum number of stored transitions.
    """

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._storage: list[
            tuple[np.ndarray, int, float, np.ndarray, bool]
        ] = []

    def __len__(self) -> int:
        """Number of elements currently stored."""
        return len(self._storage)

    def add(self, transition: tuple[np.ndarray, int, float, np.ndarray,
                                    bool]) -> None:
        """Add a transition, discarding the oldest one if the buffer is full.

        Args:
            transition: Five-tuple as described in the class docstring.

        Returns:
            None
        """
        if len(self._storage) == self._capacity:
            self._storage.pop(0)
        self._storage.append(transition)

    def sample(
            self, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Uniformly sample a mini-batch without replacement.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            tuple: Five NumPy arrays
                states (float32): Array with shape (batch_size, state_size)
                actions (int64): Array with shape (batch_size,)
                rewards (float32): Array with shape (batch_size,)
                next_states (float32): Array with shape (batch_size, state_size)
                dones (bool): Boolean mask with shape (batch_size,)
        """
        states, actions, rewards, next_states, dones = zip(
            *random.sample(self._storage, batch_size)
        )
        return (
            np.vstack(states),
            np.fromiter(actions, dtype=np.int64),
            np.fromiter(rewards, dtype=np.float32),
            np.vstack(next_states),
            np.fromiter(dones, dtype=np.bool_),
        )


# ────────────────────────── Main agent class ──────────────────────────
class StochasticQLearningAgent:
    """Stochastic Deep Action-Value Learning agent with optional Double
    Action-Value targets.

    The agent is *environment-agnostic*: you call `act` to obtain the
    next action index and `step` to feed back the resulting transition.

    Args:
        state_size: Dimension of the environment's flattened state vector.
        action_size: Cardinality of the discrete action set |A|.
        index2action: Mapping from integer indices to one-hot action vectors.
        config: Optional user-supplied configuration.  If *None* a default
            `StochasticQLearningConfig` instance is constructed.
    """

    # ─────────────────────────── Construction ───────────────────────────
    def __init__(
        self,
        state_size: int,
        action_size: int,
        index2action: dict[int, Sequence[float]],
        config: StochasticQLearningConfig | None = None,
    ) -> None:
        self.config = config or StochasticQLearningConfig()
        self.state_size = state_size
        self.action_size = action_size
        self.index2action = index2action

        # Derived sizes
        log2_actions = max(1, ceil(log2(action_size)))
        self.batch_size = int(self.config.batch_multiplier * log2_actions)
        self.buffer_capacity = int(self.config.buffer_multiplier * self.batch_size)
        self.subset_size = (
            action_size
            if self.config.deterministic
            else max(1, int(self.config.subset_coefficient * log2_actions))
        )

        # Device placement
        self.device = torch.device(
            "cuda" if self.config.use_cuda and torch.cuda.is_available() else "cpu"
        )

        # Online and target networks
        input_dim = state_size + action_size
        self.online_network = _StateActionNetwork(
            input_dim, self.config.hidden_size, self.device
        )
        self.target_network = _StateActionNetwork(
            input_dim, self.config.hidden_size, self.device
        )
        self.sync_target()
        self.target_network.eval()

        # Optimiser and loss function
        self.optimizer = optim.Adam(
            self.online_network.parameters(),
            lr=self.config.learning_rate,
        )
        self._loss_function = nn.MSELoss()

        # Exploration bookkeeping
        self.epsilon = self.config.epsilon_start

        # Experience replay
        self.replay = _ReplayBuffer(self.buffer_capacity)

        # Miscellaneous runtime state
        self._one_hot_identity = torch.eye(action_size, device=self.device)
        self._learn_step_counter = 0
        self.log = logger.bind(agent="StochasticDQN")

    # ────────────────────────── Private helpers ──────────────────────────
    def _one_hot(self, indices: torch.Tensor) -> torch.Tensor:
        """Return one-hot vectors for integer indices without allocation."""
        return self._one_hot_identity[indices]

    def _concat_state_action(
        self,
        state_tensor: torch.Tensor,
        action_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate state and one-hot action along the feature axis."""
        return torch.cat((state_tensor, self._one_hot(action_indices)), dim=1)

    def _stochastic_max(
        self,
        next_states: torch.Tensor,
        candidate_actions: Iterable[int],
        use_target_net: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute maximal action-value and corresponding action index.

        Implements the Double Action-Value trick: the online network chooses
        *argmax*; the target network provides the evaluation value.

        Args:
            next_states: Batch of next-state tensors of shape *(B, |S|)*.
            candidate_actions: Iterable of integer action indices considered
                for each sample.
            use_target_net: When *True* evaluation is done with the target
                network; otherwise the online network is used.

        Returns:
            Tuple of tensors *(max_action_values, best_action_indices)*, both
            with shape *(B,)*.
        """
        action_tensor = torch.tensor(
            list(set(candidate_actions)), dtype=torch.long, device=self.device
        )

        # Evaluate every candidate with the online network (selection phase)
        batch_size, num_actions = next_states.size(0), action_tensor.numel()
        expanded_states = next_states.unsqueeze(1).repeat(1, num_actions, 1)
        expanded_actions = action_tensor.unsqueeze(0).repeat(batch_size, 1)

        online_values = self.online_network(
            self._concat_state_action(
                expanded_states.view(-1, self.state_size),
                expanded_actions.view(-1),
            )
        ).view(batch_size, num_actions)

        best_indices = online_values.argmax(dim=1)
        best_actions = action_tensor[best_indices]

        # Evaluate selected actions with evaluation network (evaluation phase)
        evaluation_network = self.target_network if use_target_net else self.online_network
        evaluated_values = evaluation_network(
            self._concat_state_action(next_states, best_actions)
        ).squeeze(1)

        return evaluated_values, best_actions

    def _epsilon_greedy(
        self, state_np: np.ndarray, random_subset: np.ndarray
    ) -> int:
        """Return an action index chosen by an epsilon-greedy behaviour policy.

        Uniform random actions are produced until replay is sufficiently filled
        to allow learning.  When *deterministic* is *True*, all actions are
        evaluated; otherwise a logarithmic subset is used.

        Args:
            state_np: Current environment state as a NumPy array.
            random_subset: Fresh array of uniformly drawn action indices used
                to diversify the candidate pool.

        Returns:
            Integer index of the selected action.
        """
        should_explore = (
            np.random.rand() <= self.epsilon or len(self.replay) < self.batch_size
        )
        if should_explore:
            return random.randrange(self.action_size)

        state_tensor = _to_tensor(state_np, device=self.device).unsqueeze(0)

        if self.config.deterministic:
            candidates = range(self.action_size)
        else:
            k = min(len(self.replay), self.subset_size)
            replay_actions = (
                [a for _, a, *_ in random.sample(self.replay._storage, k)]
                if k
                else []
            )
            candidates = np.concatenate((replay_actions, random_subset)).tolist()

        _, best_action = self._stochastic_max(
            state_tensor,
            candidates,
            use_target_net=self.config.double_dqn,
        )
        return int(best_action.item())

    # ────────────────────────── Public interface ──────────────────────────
    def act(self, state: np.ndarray, random_actions: np.ndarray) -> int:
        """Select an action for *state* under the current ε-greedy policy.

        Args:
            state: One-dimensional NumPy array representing the current
                environment state (shape ``(state_size,)``).
            random_actions: One-dimension array of uniformly sampled
                action indices used to augment the candidate set when
                stochastic maximisation is enabled.

        Returns:
            int: Index of the chosen discrete action in the range
            ``[0, action_size - 1]``.
        """        
        return self._epsilon_greedy(state, random_actions)

    def step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        random_actions: np.ndarray,
    ) -> None:
        """Store a transition and perform one optimisation step.

        Args:
            state: Original state before the action.
            action: Integer action index taken.
            reward: Scalar reward received.
            next_state: Environment state after performing *action*.
            done: Whether the episode terminated at *next_state*.
            random_actions: Array of random action indices for the subset
                selection mechanism.
        """
        self.replay.add((state, action, reward, next_state, done))
        self._learn(random_actions)

        if self.epsilon > self.config.epsilon_min:
            self.epsilon *= self.config.epsilon_decay

    def _learn(self, random_actions: np.ndarray) -> None:
        """Perform one optimisation step using replayed experience.

        Outline:
            1. Sample a mini-batch from the replay buffer.
            2. Build bootstrapped targets via stochastic maximisation
               (Double Action-Value Learning optional).
            3. Minimise mean-squared error loss and update parameters.
            4. Hard-update the target network every
               `config.target_update_every` steps

        Args:
            random_actions: One-dimensional `np.ndarray` containing
                uniformly sampled action indices.  When the agent
                operates in stochastic mode (*config.deterministic is
                False*), this array is merged with the actions observed
                in the current mini-batch to form the candidate set used
                by `_stochastic_max`.

        Returns:
            None.  The method updates the online network's parameters
            in place and, at the specified cadence, also updates the
            target network.

        Raises:
            RuntimeError: Propagated from PyTorch if the backward pass
                fails (for example, due to non-finite gradients).
        """
        if len(self.replay) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay.sample(
            self.batch_size
        )

        state_tensor = _to_tensor(states, device=self.device)
        action_tensor = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        reward_tensor = _to_tensor(rewards, device=self.device)
        next_state_tensor = _to_tensor(next_states, device=self.device)
        done_tensor = torch.as_tensor(dones.astype(np.float32), device=self.device)

        if self.config.deterministic:
            candidate_pool = range(self.action_size)
        else:
            candidate_pool = np.concatenate((actions, random_actions)).tolist()

        with torch.no_grad():
            next_values, _ = self._stochastic_max(
                next_state_tensor,
                candidate_pool,
                use_target_net=self.config.double_dqn,
            )
            targets = (
                reward_tensor
                + self.config.discount_factor * next_values * (1 - done_tensor)
            )

        current_values = self.online_network(
            self._concat_state_action(state_tensor, action_tensor)
        ).squeeze(1)

        loss = self._loss_function(current_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Periodically update target network
        self._learn_step_counter += 1
        if self._learn_step_counter % self.config.target_update_every == 0:
            self.sync_target()

    def sync_target(self) -> None:
        """Synchronise the target network with the online network.

        This is a *hard* update: every parameter tensor in
        `target_network` is assigned the corresponding tensor
        from `online_network`.  The method is invoked
        automatically by `_learn` every
        `config.target_update_every` optimisation steps but can also be
        called manually, for example, at the start of training or before
        evaluation.

        Returns:
            None.  The parameters of `target_network` are
            modified in place.
        """
        self.target_network.load_state_dict(self.online_network.state_dict())