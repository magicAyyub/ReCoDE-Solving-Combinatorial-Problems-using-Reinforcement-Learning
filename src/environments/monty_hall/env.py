"""
A Monty Hall environment implementation in Gymnasium, customizable with the number of doors and cars.
"""

from .renderer import MontyHallPygameRenderer
from .state import DoorState, Phase

from typing import Optional, Literal

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

class MontyHallEnv(gym.Env):
    """A full Monty Hall implementation that follows Gymnasium's API."""

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 4,
    }

    def __init__(
        self,
        *,
        n_doors: int = 3,
        n_cars: int = 1,
        render_mode: Literal["human", "rgb_array"] | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialises the customizable Monty Hall environment.

        Args:
            n_doors (int, optional): Number of total doors in the environment. Defaults to 3.
            n_cars (int, optional): Number of cars behind doors, the rest of doors will be goats.
              Defaults to 1.
            render_mode (Literal or None): rendering mode of the environment. Defaults to None (no rendering needed).
            seed (int or None): controls the random number generation. Note that, setting this
             will cause deterministic behaviour, mostly useful for debugging only. Defaults to None (random seed).

        Raises:
            ValueError: for any logical errors, such as having less than 3 doors, invalid number of cars,
             or an invalid render mode
        """
        # ─── Logical assertions ─── #
        if n_doors < 3:
            raise ValueError("Monty Hall requires at least 3 doors.")
        if not (0 < n_cars < n_doors):
            raise ValueError(f"n_cars must be between 1 and {n_doors - 1}.")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode '{render_mode}'.")

        # ─── Core parameters ───
        self.n_doors = int(n_doors)
        self.n_cars = int(n_cars)
        self.render_mode = render_mode

        # ─── Gym spaces ───
        # Observation/State Space: 1D NumPy vector of doors with value (0-3) from DoorState
        self.observation_space = spaces.MultiDiscrete(
            np.full(self.n_doors, len(DoorState), dtype=np.int64)
        )
        # Action Space: Discrete choice, given the number of doors
        self.action_space = spaces.Discrete(self.n_doors)


        # ─── renderer (optional) ───
        self._renderer: MontyHallPygameRenderer | None = None
        if render_mode is not None:
            self._renderer = MontyHallPygameRenderer(self.n_doors, self.metadata, render_mode)
            
        # ─── Initial episode state ───
        self.reset(seed=seed)

    # ──────────────────────────────────────────────────────────────────────────────── #
    #                                 Gymnasium API                                    #
    # ──────────────────────────────────────────────────────────────────────────────── #
    def reset(self, *, seed: Optional[int] = None, options=None):
        """Resets the environment, whenever an episode has terminated.

        Args:
            seed (Optional[int]): reset the environment with a specific seed value. Defaults to None.
            options: unused, mandated by the Gymnasium interface. Defaults to None.

        Returns:
            Pair: 1D state vector of DoorStates, and info (dict) consisting of auxiliary information from _get_info()
        """
        super().reset(seed=seed)
        self._reset_state()

        if self.render_mode is not None:
            self.render()
        return self._get_obs(), self._get_info()

    def step(self, action: int):
        """Corresponds to env.step() in Gymnasium, corresponds to an atomic action taken in the environment.

        Args:
            action (int): the integer index of the action taken.

        Raises:
            RuntimeError: if an action is performed on an already completed episode.
            ValueError: if an action ID is provided that is not valid.

        Returns:
            observation (1d Numpy): next observation of door states
            reward (float): scalar of the reward value for the current action
            terminated (bool): whether the episode is terminated
            truncated (bool): whether the episode was truncated (not applicable in this env)
            info (dict): auxiliary information of episode progress from _get_info()
        """
        if self._phase is Phase.DONE:
            raise RuntimeError(
                "Episode is already completed! Call reset() to start a new one."
            )
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action!r}.")

        reward: float = 0.0
        terminated = False
        truncated = False  # We provide no truncation logic for this environment — it's a 2-step environment

        if self._phase is Phase.AWAITING_FIRST_PICK:
            self._choose_initial(action)
        elif self._phase is Phase.AFTER_REVEAL:
            terminated, reward = self._choose_final(action)

        self.render()
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self, mode: str | None = None):
        """ Handles the rendering functionality by composition

        Args:
            mode (str | None, optional): _description_. Defaults to None.

        Raises:
            ValueError: if render() is called despite initially setting render_mode as None.

        Returns:
            np.ndarray | None: np.ndarray corresponds to render_mode=rgb_array,
              and None corresponds to human (renders in the designated PyGame instance)
        """
        if self.render_mode is None:
            return None
            
        if not self._renderer:
            raise ValueError("render_mode was None when env was created.")
        return self._renderer.render(self._state)

    def close(self):
        """Closes the environment, performing some basic cleanup of resources."""
        if self._renderer:
            self._renderer.close()
            self._renderer = None
    # ──────────────────────────────────────────────────────────────────────────────── #
    #                                 Private helpers                                  #
    # ──────────────────────────────────────────────────────────────────────────────── #
    def _reset_state(self):
        """Creates a fresh set-up behind the scenes of all internal state"""
        self._state = np.full(self.n_doors, DoorState.CLOSED, dtype=int)
        self._car_doors = self.np_random.choice(
            self.n_doors, size=self.n_cars, replace=False
        )
        self._chosen_door: Optional[int] = None
        self._phase = Phase.AWAITING_FIRST_PICK

    def _choose_initial(self, door: int) -> None:
        if self._state[door] != DoorState.CLOSED:
            raise ValueError("First pick must be an unopened, unchosen door.")

        self._chosen_door = door
        self._state[door] = DoorState.CHOSEN

        closed = [i for i, s in enumerate(self._state) if s == DoorState.CLOSED]
        goats_available = [d for d in closed if d not in self._car_doors]

        # Host reveals as many goats as possible while leaving exactly two
        # closed doors (the player's pick and one other).
        n_to_open = min(len(goats_available), len(closed) - 1)
        if n_to_open:
            to_open = self.np_random.choice(
                goats_available, size=n_to_open, replace=False
            )
            self._state[to_open] = DoorState.GOAT

        self._phase = Phase.AFTER_REVEAL

    def _choose_final(self, door: int):
        if self._state[door] not in (DoorState.CLOSED, DoorState.CHOSEN):
            raise ValueError("Final pick must be one of the remaining closed doors.")

        reward = 1.0 if door in self._car_doors else 0.0
        self._state[door] = DoorState.CAR if reward else DoorState.GOAT

        self._phase = Phase.DONE
        return True, reward

    def _get_obs(self):
        """Returns a copy of the current state vector"""
        return self._state.copy()

    def _get_info(self):
        """Provides full information of the currently running instance.

        Returns:
            dict: consisting of
              - which doors have cars behind them,
              - the chosen door,
              - and the progress step of the env._
        """
        return {
            "car_doors": self._car_doors.tolist(),
            "chosen_door": self._chosen_door,
            "phase": self._phase.name,
        }


# Register the environment to allow usage with `gym.make``
register(
    id="MontyHall-v0",
    entry_point="environments.monty_hall.env:MontyHallEnv",
)
