from __future__ import annotations

import time
import random
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo
from loguru import logger

@dataclass
class QLearningConfig:
    """Hyper-parameters for tabular Q-learning."""
    learning_rate: float = 0.1
    discount_factor: float = 0.99
    epsilon_start: float = 1.0
    epsilon_decay: float = 0.9985
    epsilon_min: float = 0.05
    max_steps_per_episode: int = 100
    seed: int | None = 42
    video_dir: str | None = None


class QLearningAgent:
    """
    Tabular Q-learning agent for discrete state & action spaces.

    Usage
    -----
    >>> env = gym.make("Taxi-v3")
    >>> cfg = QLearningConfig()
    >>> agent = QLearningAgent(env, cfg)
    >>> agent.train(episodes=5_000)
    >>> agent.test(episodes=100)
    """

    def __init__(self, env: gym.Env, cfg: QLearningConfig) -> None:
        # ――― Environment ――― #
        if cfg.video_dir:
            env = RecordVideo(
                env,
                video_folder=cfg.video_dir,
                episode_trigger=lambda ep: ep % 100 == 0,  # record every 100th episode
            )
        self.env = env

        # ――― Hyper-parameters ――― #
        self.cfg = cfg
        self.lr = cfg.learning_rate
        self.gamma = cfg.discount_factor
        self.epsilon = cfg.epsilon_start

        # ――― Spaces & Q-table ――― #
        self.n_actions = env.action_space.n
        self.n_states = env.observation_space.n
        self.q_table = np.zeros((self.n_states, self.n_actions), dtype=np.float32)

        # ――― Reproducible seeding ――― #
        if cfg.seed is not None:
            random.seed(cfg.seed)
            np.random.seed(cfg.seed)
            env.reset(seed=cfg.seed)

        # ――― Logging ――― #
        self.log = logger.bind(agent="TabularQLearning")

    # ─────────────────────────────── Agent helpers ─────────────────────────────── #
    def _choose_action(self, state: int) -> int:
        """ε-greedy action selection."""
        if random.random() < self.epsilon:          # explore
            return self.env.action_space.sample()
        return int(np.argmax(self.q_table[state]))  # exploit

    def _update_q_table(
        self, state: int, action: int, reward: float, next_state: int
    ) -> None:
        """Standard one-step Q-learning update."""
        best_next = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state, best_next]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error

    @staticmethod
    def _has_converged(reward_history: list[float], window: int = 100) -> bool:
        """Simple convergence heuristic: recent rewards stable & high."""
        if len(reward_history) < window:
            return False
        window_rewards = reward_history[-window:]
        return np.std(window_rewards) < 1e-3 and np.mean(window_rewards) > 0.9

    # ─────────────────────────────── Public API ─────────────────────────────── #
    def train(
        self,
        episodes: int = 2_000,
        log_interval: int = 100,
    ) -> tuple[list[float], list[float], int, float]:
        """
        Train the agent.

        Returns
        -------
        rewards : list[float]
            Reward obtained in each episode.
        converged_ep : int
            Episode at which convergence was detected (-1 if never).
        converged_time : float
            Wall-clock seconds to convergence (0.0 if never).
        """
        rewards: list[float] = []

        start_t = time.perf_counter()
        converged_ep, converged_time = -1, 0.0

        for ep in range(episodes):
            state, _ = self.env.reset(seed=self.cfg.seed)
            episode_reward = 0.0

            for step in range(self.cfg.max_steps_per_episode):
                action = self._choose_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)

                self._update_q_table(state, action, reward, next_state)
                state = next_state
                episode_reward = reward  # keep final reward (per env spec)

            # ε decay (after each episode)
            self.epsilon = max(self.cfg.epsilon_min, self.epsilon * self.cfg.epsilon_decay)

            # bookkeeping
            rewards.append(episode_reward)

            if (ep + 1) % log_interval == 0:
                self.log.info(
                    "Episode {ep:>5d} | R: {r:+.1f} | TC: {tc:>6.2f}% | ε: {eps:.3f}",
                    ep=ep + 1,
                    r=episode_reward,
                    eps=self.epsilon,
                )

            # convergence detection
            if converged_ep == -1 and self._has_converged(rewards):
                converged_ep = ep
                converged_time = time.perf_counter() - start_t
                self.log.success(
                    "Converged at episode {ep} after {t:.2f}s",
                    ep=converged_ep,
                    t=converged_time,
                )

        return rewards, converged_ep, converged_time

    def test(self, episodes: int = 10) -> None:
        """
        Run greedy evaluation episodes (no learning, no ε-greedy exploration)."""
        for ep in range(1, episodes + 1):
            state, _ = self.env.reset(seed=self.cfg.seed)
            self.env.render()

            for step in range(self.cfg.max_steps_per_episode):
                action = int(np.argmax(self.q_table[state]))
                state, reward, terminated, truncated, info = self.env.step(action)
                self.env.render()

                if terminated or truncated:
                    break

            self.log.info("Test-Ep {ep:>3d} | Reward: {r:+.1f}", ep=ep, r=reward)

        self.env.close()