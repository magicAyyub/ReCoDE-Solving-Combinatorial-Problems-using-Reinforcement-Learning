"""stochastic_q_learning.py

Tabular Stochastic Q-learning implementation based on Fourati et al. (2024):
    - https://arxiv.org/pdf/2405.10310v1
    - https://github.com/fouratifares/stochdqn/blob/main/rl/stochdqn.py

The API mirrors `QLearningAgent` (from `src/rl/tabular_q_learning.py`) so you can drop-in replace it.

Example
-------
>>> env = gym.make("Taxi-v3")
>>> cfg = StochasticQLearningConfig()
>>> agent = StochasticQLearningAgent(env, cfg)
>>> agent.train(episodes=5_000)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from math import inf, log2
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ─────────────────────────── Low‑level utilities ──────────────────────────── #

def to_tensor(ndarray: np.ndarray | Sequence[float], *, device: torch.device) -> torch.Tensor:
    """Convert `ndarray`/`list` to *float32* tensor on the given **device**."""
    return torch.as_tensor(ndarray, dtype=torch.float32, device=device)


# ────────────────────────────────── Network ────────────────────────────────── #
class SQNetwork(nn.Module):
    """*State-Action* Q-value network: f(s, a) → Q_θ(s, a)."""

    def __init__(self, in_dim: int, hidden: int, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.q = nn.Linear(hidden, 1)  # scalar output
        self.to(device)

    # Forward pass --------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, in_dim)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.q(x)  # (B, 1)

# ───────────────────────────────── Buffer ──────────────────────────────────── #
class ReplayBuffer:
    """Fixed-size cyclic buffer ⟨s, a, r, s', done⟩."""

    def __init__(self, capacity: int):
        self._cap = capacity
        self._buf: list[tuple[np.ndarray, int, float, np.ndarray, bool]] = []

    # --------------- Public API ----------------
    def __len__(self) -> int:  # e.g. `len(buffer)`
        return len(self._buf)

    def add(self, exp: tuple[np.ndarray, int, float, np.ndarray, bool]) -> None:
        if len(self._buf) == self._cap:
            self._buf.pop(0)
        self._buf.append(exp)

    def sample(
        self, batch: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Uniform random sample sans replacement."""
        states, actions, rewards, next_states, dones = zip(*random.sample(self._buf, batch))
        return (
            np.vstack(states),
            np.fromiter(actions, dtype=np.int64),
            np.fromiter(rewards, dtype=np.float32),
            np.vstack(next_states),
            np.fromiter(dones, dtype=np.bool_),
        )


# ───────────────────────────── Config struct ──────────────────────────────── #
@dataclass(slots=True)
class StochasticQLearningConfig:
    """Hyperparameters (defaults reproduce the original). """

    hidden_size: int = 64
    lr: float = 1e-3
    gamma: float = 0.99

    # ε‑greedy exploration
    epsilon_start: float = 1.0
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01

    # Replay & subset sizes
    subset_coef: float = 2.0  # subset = coef · log₂|A|
    batch_mul: float = 2.0    # batch = mul · log₂|A|
    buffer_mul: float = 2.0   # buffer = mul · batch

    # Optimisation toggles
    deterministic: bool = True  # => subset = all actions during maximisation
    double_dqn: bool = True     # use Double‑DQN target

    use_cuda: bool = True

# ───────────────────────────── Stoch DQN Agent ────────────────────────────── #
class StochasticQLearningAgent:
    """Deep **Stochastic DQN** with optional Double-DQN targets.

    Parameters
    ----------
    state_size
        Dimensionality of *state* vector (|S|).
    action_size
        Total number of discrete actions (|A|).
    index2action
        Mapping *index → one-hot action-vector* (shape = |A|). Allows us to
        concatenate state & action into one input.
    cfg
        `StochDQNConfig` hyperparameters.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        index2action: dict[int, Sequence[float]],
        cfg: StochasticQLearningConfig = StochasticQLearningConfig(),
    ) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.idx2act = index2action
        self.cfg = cfg

        # Derived sizes ---------------------------------------------------- #
        self._log2A = max(1, round(log2(action_size)))
        self.batch_size = int(cfg.batch_mul * self._log2A)
        self.buffer_size = int(cfg.buffer_mul * self.batch_size)
        self.subset_size = (
            action_size  # deterministic ⇒ evaluate all actions
            if cfg.deterministic
            else int(cfg.subset_coef * self._log2A)
        )

        # Device & networks ------------------------------------------------- #
        self.device = torch.device(
            "cuda" if cfg.use_cuda and torch.cuda.is_available() else "cpu"
        )
        in_dim = state_size + action_size  # s ⊕ one‑hot(a)
        self.q_net = SQNetwork(in_dim, cfg.hidden_size, self.device)
        self.tgt_net = SQNetwork(in_dim, cfg.hidden_size, self.device)
        self.tgt_net.load_state_dict(self.q_net.state_dict())
        self.tgt_net.eval()  # freeze target params

        self.optim = optim.Adam(self.q_net.parameters(), lr=cfg.lr)
        self.loss_fn = nn.MSELoss()

        # Exploration ------------------------------------------------------- #
        self.epsilon = cfg.epsilon_start

        # Replay buffer ----------------------------------------------------- #
        self.replay = ReplayBuffer(self.buffer_size)

    # ──────────────────────────── Core helpers ──────────────────────────── #
    def _concat_sa(self, s: torch.Tensor, a_idx: torch.Tensor) -> torch.Tensor:
        """Concatenate state & *one-hot* action index along dim-1."""
        a_oh = torch.eye(self.action_size, device=self.device)[a_idx]
        return torch.cat((s, a_oh), dim=1)

    def _stochastic_max(
        self,
        next_s: torch.Tensor,
        candidate_actions: Iterable[int],
        use_target: bool,
    ) -> torch.Tensor:
        """Max-Q over **subset** of actions (batch aware)."""
        q_max = torch.full((next_s.size(0),), -inf, device=self.device)
        # Initially choose first action as placeholder best
        best_a = torch.zeros(next_s.size(0), dtype=torch.long, device=self.device)

        net = self.tgt_net if use_target else self.q_net

        for a in set(candidate_actions):
            qa = net(self._concat_sa(next_s, torch.full_like(best_a, a)))  # (B,1)
            better = qa.squeeze(1) > q_max
            q_max[better] = qa.squeeze(1)[better]
            best_a[better] = a
        return q_max, best_a

    def _epsilon_greedy(self, state: np.ndarray, rand_subset: np.ndarray) -> int:
        """ε-greedy action selection; falls back to replay/random if needed."""
        if np.random.rand() <= self.epsilon or len(self.replay) < self.batch_size:
            return random.randrange(self.action_size)

        s_tensor = to_tensor(state, device=self.device).unsqueeze(0)

        # Candidate actions -------------------------------------------------
        if self.cfg.deterministic:
            candidate = range(self.action_size)
        else:
            # Current replay actions + on‑the‑fly randoms ensure diversity
            replay_actions = [a for _, a, *_ in random.sample(self.replay._buf, self.subset_size)]
            candidate = np.concatenate((replay_actions, rand_subset))

        q_max, best_a = self._stochastic_max(s_tensor, candidate, use_target=self.cfg.double_dqn)
        return int(best_a.item())
    
    @staticmethod
    def _has_converged(reward_history: list[float], window: int = 100) -> bool:
        """Simple convergence heuristic: recent rewards stable & high."""
        if len(reward_history) < window:
            return False
        window_rewards = reward_history[-window:]
        return np.std(window_rewards) < 1e-3 and np.mean(window_rewards) > 0.9

    # ───────────────────────────── Public API ────────────────────────────── #
    def act(self, state: np.ndarray, rand_actions: np.ndarray) -> int:
        """Public wrapper around `_epsilon_greedy`."""
        return self._epsilon_greedy(state, rand_actions)

    def step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        rand_actions: np.ndarray,
    ) -> None:
        """Store transition & **one** gradient‑descent update."""
        self.replay.add((state, action, reward, next_state, done))
        self._learn(rand_actions)
        # ε decay ----------------------------------------------------------- #
        if self.epsilon > self.cfg.epsilon_min:
            self.epsilon *= self.cfg.epsilon_decay

    # -------------------------- optimisation loop ------------------------- #
    def _learn(self, rand_actions: np.ndarray) -> None:
        if len(self.replay) < self.batch_size:
            return  # not enough experience yet

        # Sample mini‑batch -------------------------------------------------- #
        s, a, r, s2, d = self.replay.sample(self.batch_size)
        s_tensor = to_tensor(s, device=self.device)
        a_tensor = torch.as_tensor(a, dtype=torch.long, device=self.device)
        r_tensor = to_tensor(r, device=self.device)
        s2_tensor = to_tensor(s2, device=self.device)
        d_tensor = torch.as_tensor(d.astype(np.float32), device=self.device)

        # Compute targets ---------------------------------------------------- #
        if self.cfg.deterministic:
            cand = range(self.action_size)
        else:
            cand = np.concatenate((a, rand_actions))

        with torch.no_grad():
            q_max_next, _ = self._stochastic_max(
                s2_tensor, cand, use_target=self.cfg.double_dqn
            )
            y = r_tensor + self.cfg.gamma * q_max_next * (1 - d_tensor)

        # Current Q estimates ---------------------------------------------- #
        q_sa = self.q_net(self._concat_sa(s_tensor, a_tensor)).squeeze(1)

        # Loss & SGD step --------------------------------------------------- #
        loss = self.loss_fn(q_sa, y)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    # ------------------------- Target‑network sync ------------------------ #
    def sync_target(self) -> None:
        """Hard update: θ_target ← θ."""
        self.tgt_net.load_state_dict(self.q_net.state_dict())