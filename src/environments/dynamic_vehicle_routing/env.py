"""
A customisable Dynamical Vehicle Routing problem environment implementation satisfying the Gymnasium interface.

Based on the example given by OR-Gym (Balaji et al.), with some cleaned up code.
    - Paper: https://arxiv.org/abs/1911.10641
    - GitHub: https://github.com/hubbs5/or-gym/blob/master/or_gym/envs/classic_or/vehicle_routing.py
"""

from typing import Optional, Literal

import numpy as np
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

@dataclass
class DVRConfig():
    n_restaurants = 2

class DynamicVehicleRoutingEnv(gym.Env):
    """A full Dynamic Vehicle Routing problem environment implementation that follows Gymnasium's API."""

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 4,
    }

    def __init(
        *,
        self
    ) -> None:
        """ This environment simulates a driver working with a food delivery app
        to move through a city, accept orders, pick them up from restaurants,
        and deliver them to waiting customers.

        Each order has some known associated information:
        * delivery value
        * restaurant (pickup location)
        * delivery location

        An order is timed out within 60 minutes, or may be accepted by another driver before then.
        
        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
    

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
        pass

    def render(self, mode: str | None = None):
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
        pass

    def _get_obs(self):
        pass

    def _get_info(self):
        pass 

# Register the environment to allow usage with `gym.make``
register(
    id="DynamicVehicleRouting-v0",
    entry_point="environments.dynamic_vehicle_routing.env:DynamicVehicleRoutingEnv",
)
