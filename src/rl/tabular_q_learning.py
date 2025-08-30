"""tabular_q_learning.py

Vanilla tabular Q-learning agent with NumPy.

This module exposes a minimal, yet complete, implementation of the **one-step
Q-learning** algorithm (Watkins, 1989) for environments that feature *discrete*
state and action spaces. It is designed for pedagogical clarity rather than raw
performance and, therefore, uses an explicit table to store state-action
values.

Example:
    >>> import gymnasium as gym
    >>> from tabular_q_learning import QLearningAgent, QLearningConfig
    >>> env = gym.make("Taxi-v3")
    >>> cfg = QLearningConfig()
    >>> agent = QLearningAgent(env, cfg)
    >>> agent.train(episodes=5_000)
    >>> agent.test(episodes=100)
"""
from __future__ import annotations
from src.rl.common import has_converged


import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo
from loguru import logger

@dataclass(slots=True)
class QLearningConfig:
    """Hyperparameters and miscellaneous settings for :class:`QLearningAgent`.

    Attributes:
        learning_rate (float): Temporal-difference step size $\alpha$.
        discount_factor (float): Future reward discount $\gamma \in [0, 1]`.
        epsilon_start (float): Initial exploration rate $\varepsilon$.
        epsilon_decay (float): Multiplicative decay applied to *ε* after every episode.
        epsilon_min (float): Lower bound for *ε*.
        max_steps_per_episode (int): Hard limit on the length of an episode.
        seed (int | None): RNG seed for reproducibility. ``None`` disables seeding.
        video_dir (str | None): If provided, every 100th episode is recorded to this
            directory using :class:`gymnasium.wrappers.RecordVideo`.
    """
    learning_rate: float = 0.1
    discount_factor: float = 0.99
    epsilon_start: float = 1.0
    epsilon_decay: float = 0.9985
    epsilon_min: float = 0.05
    max_steps_per_episode: int = 100
    seed: int | None = 42
    video_dir: str | None = None


class QLearningAgent:
    """Tabular Q-learning agent for finite Markov Decision Processes.

    The agent stores its estimates of the **optimal action-value function** in a
    2D NumPy array—often referred to as the *Q-table* in RL literature—and
    updates those estimates with the canonical one-step TD rule.

    Args:
        environment (gym.Env): An environment whose observation and action
            spaces are both instances of :class:`gym.spaces.Discrete`.
        config (QLearningConfig): Hyperparameter bundle.

    Raises:
        ValueError: If the environment does not have discrete observation *and*
            action spaces.
    """

    def __init__(self, environment: gym.Env, config: QLearningConfig) -> None:  # noqa: D401,E501
        # ─── Environment & video capture ─── #
        if config.video_dir is not None:
            environment = RecordVideo(
                environment,
                video_folder=config.video_dir,
                episode_trigger=lambda episode_idx: episode_idx % 100 == 0,
            )

        if not isinstance(environment.observation_space, gym.spaces.Discrete):
            raise ValueError("QLearningAgent requires a *discrete* observation space.")
        if not isinstance(environment.action_space, gym.spaces.Discrete):
            raise ValueError("QLearningAgent requires a *discrete* action space.")

        self.environment: gym.Env = environment

        # ─── Hyper‑parameters ─── #
        self.config: QLearningConfig = config
        self.learning_rate: float = config.learning_rate
        self.discount_factor: float = config.discount_factor
        self.epsilon: float = config.epsilon_start

        # ─── State–action value table ─── #
        self.num_actions: int = environment.action_space.n
        self.num_states: int = environment.observation_space.n
        self.action_value_table: np.ndarray = np.zeros(
            (self.num_states, self.num_actions), dtype=np.float32
        )

        # ─── Reproducibility ─── #
        if config.seed is not None:
            random.seed(config.seed)
            np.random.seed(config.seed)
            environment.reset(seed=config.seed)

        # ─── Logging ─── #
        self.log = logger.bind(agent="TabularQLearning")


    # ──────────────────────────────────────────────────────────────────────────────── #
    #                                 Private helpers                                  #
    # ──────────────────────────────────────────────────────────────────────────────── #
    def _select_action(self, state_idx: int, action_mask: np.ndarray | None = None) -> int:
        """Sample an action via an $\varepsilon$-greedy policy.

        Args:
            state_idx (int): Discrete state identifier obtained from the environment.
            action_mask : 1-D np.ndarray | None
                Binary vector (1 → legal, 0 → illegal).  If None, we assume
                every action is legal.

        Returns:
            int: Index of the chosen action.
        """
        # • Exploration
        if random.random() < self.epsilon:
            return int(self.environment.action_space.sample(mask=action_mask))
        
        # • Exploitation: arg-max over the same legal set
        if action_mask is not None:
            legal = np.flatnonzero(action_mask)          # indices where mask==1
            best_idx = np.argmax(self.action_value_table[state_idx, legal])
            return int(legal[best_idx])
        return int(np.argmax(self.action_value_table[state_idx]))

    def _update_action_value_table(
        self,
        state_idx: int,
        action_idx: int,
        reward: float,
        next_state_idx: int,
        next_mask: np.ndarray | None
    ) -> None:
        """Apply the one-step Q-learning update rule.

        Args:
            state_idx (int): Index of the pre-transition state $s_t$.
            action_idx (int): Index of the action $a_t$ taken in $s_t$.
            reward (float): Immediate scalar reward $r_{t+1}$.
            next_state_idx (int): Index of the successor state $s_{t+1}$.
            next_mask (np.ndarray): boolean array of next action mask
        """
        if next_mask is not None:
            legal_next = np.flatnonzero(next_mask)
            next_max = (np.max(self.action_value_table[next_state_idx, legal_next])
                        if legal_next.size else 0.0)
        else:
            next_max = np.max(self.action_value_table[next_state_idx])
            
        td_target = reward + self.discount_factor * next_max
        td_error  = td_target - self.action_value_table[state_idx, action_idx]
        self.action_value_table[state_idx, action_idx] += self.learning_rate * td_error

    # ──────────────────────────────────────────────────────────────────────────────── #
    #                                    Public API                                    #
    # ──────────────────────────────────────────────────────────────────────────────── #
    def train(
        self,
        episodes: int = 2_000,
        log_interval: int = 100,
    ) -> tuple[list[float], int, float]:
        """Perform on-policy learning with $\varepsilon$-greedy exploration.

        Args:
            episodes (int, optional): Number of training episodes. Defaults to ``2000``.
            log_interval (int, optional): Frequency (in episodes) at which progress
                is written to the log. Defaults to ``100``.

        Returns:
            tuple[list[float], int, float]:
                * episode_rewards - Reward after each episode.
                * converged_episode - Index of the first episode that satisfies
                  the convergence test (or ``-1`` if never converged).
                * converged_wall_time - Seconds elapsed until convergence (or
                  ``0.0`` if never converged).
        """
        episode_rewards: list[float] = []
        start_time: float = time.perf_counter()

        converged_episode: int = -1
        converged_wall_time: float = 0.0

        for episode_idx in range(episodes):
            state_idx, info = self.environment.reset(seed=self.config.seed)
            action_mask = info.get("action_mask")
            final_reward: float = 0.0

            for _ in range(self.config.max_steps_per_episode):
                action_idx: int = self._select_action(state_idx, action_mask)
                next_state_idx, reward, terminated, truncated, info = self.environment.step(
                    action_idx
                )

                self._update_action_value_table(
                    state_idx, action_idx, reward, next_state_idx, info.get("action_mask")
                )
                state_idx = next_state_idx
                action_mask = info.get("action_mask")
                final_reward = reward

                if terminated or truncated:
                    break

            # Anneal exploration rate
            self.epsilon = max(
                self.config.epsilon_min, self.epsilon * self.config.epsilon_decay
            )

            # Bookkeeping & logging
            episode_rewards.append(final_reward)
            if (episode_idx + 1) % log_interval == 0:
                self.log.info(
                    "Episode {idx:>5d} | R: {reward:+.1f} | ε: {eps:.3f}",
                    idx=episode_idx + 1,
                    reward=final_reward,
                    eps=self.epsilon,
                )

            # Early stopping
            if converged_episode == -1 and has_converged(episode_rewards):
                converged_episode = episode_idx
                converged_wall_time = time.perf_counter() - start_time
                self.log.success(
                    "Converged at episode {idx} after {t:.2f}s",
                    idx=converged_episode,
                    t=converged_wall_time,
                )

        return episode_rewards, converged_episode, converged_wall_time

    def test(self, episodes: int = 10) -> None:
        """Evaluate a policy (no exploration, no learning), a quick demo run using a rollout of a trained policy.

        Args:
            episodes (int, optional): Number of evaluation episodes. Defaults to ``10``.
        """
        for episode_idx in range(1, episodes + 1):
            state_idx, info = self.environment.reset(seed=self.config.seed)
            mask = info["action_mask"]
            self.environment.render()

            for _ in range(self.config.max_steps_per_episode):
                legal = np.flatnonzero(mask)
                best = legal[np.argmax(self.action_value_table[state_idx, legal])]
                action_idx = int(best)
                
                state_idx, reward, terminated, truncated, info = self.environment.step(
                    action_idx
                )
                mask = info["action_mask"]
                self.environment.render()

                if terminated or truncated:
                    break

            self.log.info(
                "Test Episode {idx:>3d} | Reward: {reward:+.1f}",
                idx=episode_idx,
                reward=reward,
            )

        self.environment.close()
