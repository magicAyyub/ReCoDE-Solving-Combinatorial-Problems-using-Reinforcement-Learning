"""env.py

A Monty Hall environment implementation in Gymnasium, customizable with the number of doors and cars.
"""

from enum import Enum, IntEnum, auto
import numpy as np
from typing import Optional, Literal

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register


class DoorState(IntEnum):
    """State of each door in the observation vector."""

    CLOSED = 0  # Unopened & unchosen
    GOAT = 1  # Opened and reveals a goat
    CAR = 2  # Opened and reveals a car (a win)
    CHOSEN = 3  # Still closed but currently selected by the player


class Phase(Enum):
    """Progress phase of an episode."""

    AWAITING_FIRST_PICK = auto()
    AFTER_REVEAL = auto()
    DONE = auto()


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
        render_mode: None | Literal["human", "rgb_array"] = None,
        seed: int | None = None,
    ) -> None:
        """Initialises the customizable Monty Hall environment.

        Args:
            n_doors (int, optional): Number of total doors in the environment. Defaults to 3.
            n_cars (int, optional): Number of cars behind doors, the rest of doors will be goats.
              Defaults to 1.
            render_mode (Literal or None): rendering mode of the environment. Defaults to None (no rendering needed).
            seed (Int or None): controls the random number generation. Note that, setting this
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

        # ─── PyGame placeholders (lazily initialised) ───
        self._pygame = self._window = self._surface = self._font = None

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

        if self.render_mode is not None:
            self.render()
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self, mode: str | None = None):
        mode = mode or self.render_mode or "human"
        if mode not in self.metadata["render_modes"]:
            raise NotImplementedError(f"Unsupported render mode: {mode}")

        # Lazily set up pygame window/surface/font once the first time we render
        if self._pygame is None:
            self._init_pygame()

        self._draw_frame()

        if mode == "human":
            assert self._window is not None
            self._window.blit(self._surface, (0, 0))
            self._pygame.display.flip()
            self._pygame.time.delay(int(1000 / self.metadata["render_fps"]))
            return None
        elif mode == "rgb_array":
            arr = self._pygame.surfarray.array3d(self._surface)  # (W,H,3)
            return np.transpose(arr, (1, 0, 2))  # (H,W,3) like gymnasium expects

    def close(self):
        """Closes the environment, performing some basic cleanup of resources."""
        if self._pygame is not None:
            self._pygame.quit()
            self._pygame = self._window = self._surface = self._font = None

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

    # ──────────────────────────────────────────────────────────────────────────────── #
    #                                 PyGame helpers                                   #
    # ──────────────────────────────────────────────────────────────────────────────── #
    def _init_pygame(self):
        """(Lazily) Initialises PyGame, setting up the display window and surface as per the set render_mode"""
        import pygame  # local import, allowing keeping the dependency optional until used

        pygame.init()
        pygame.font.init()

        self._pygame = pygame  # stash module reference for later use

        width = self.n_doors * 80 + 20  # 80px per door + margins
        height = 160
        if self.render_mode == "human":
            self._window = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Monty Hall")
        else:
            self._window = None

        # Off‑screen surface that we draw every frame
        self._surface = pygame.Surface((width, height))
        self._font = pygame.font.SysFont(None, 36)

    def _draw_frame(self):
        """Draw current environment state onto ``self._surface``."""
        pygame = self._pygame  # type: ignore
        assert pygame is not None and self._surface is not None
        self._surface.fill((30, 30, 30))  # background

        for idx, symbol in enumerate(self._state):
            x = 10 + idx * 80
            y = 20
            rect = pygame.Rect(x, y, 60, 100)

            if symbol == 0:  # closed, unknown
                color = (160, 160, 160)
                pygame.draw.rect(self._surface, color, rect)
            elif symbol == 1:  # open goat
                color = (240, 240, 240)
                pygame.draw.rect(self._surface, color, rect)
                text = self._font.render("G", True, (0, 0, 0))
                self._surface.blit(text, text.get_rect(center=rect.center))
            elif symbol == 2:  # open car
                color = (240, 240, 240)
                pygame.draw.rect(self._surface, color, rect)
                text = self._font.render("C", True, (0, 0, 0))
                self._surface.blit(text, text.get_rect(center=rect.center))
            elif symbol == 3:  # closed & currently chosen
                color = (80, 160, 240)
                pygame.draw.rect(self._surface, color, rect)
            else:  # should never occur
                color = (255, 0, 0)
                pygame.draw.rect(self._surface, color, rect)

            # Door outline
            pygame.draw.rect(self._surface, (0, 0, 0), rect, width=2)

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
