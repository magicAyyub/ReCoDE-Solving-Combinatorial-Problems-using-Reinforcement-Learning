import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MontyHallDiscreteWrapper(gym.ObservationWrapper):
    """
    Flattens Monty Hall's MultiDiscrete observation (length = n_doors,
    values ∈ {0,1,2,3}) into one Discrete index ∈ [0, 4**n_doors - 1].

    For n_doors = 3 the mapping is simply the base-4 number formed
    by the door states, e.g. [0,2,3] → 0·4² + 2·4¹ + 3·4⁰ = 11.
    Mapping is bijective (no information is lost).
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.n_doors = env.unwrapped.n_doors
        self.observation_space = spaces.Discrete(4**self.n_doors)

        # Pre-compute 4**[0 … n_doors-1] for fast dot-product mapping
        self._powers = (4 ** np.arange(self.n_doors)).astype(np.int64)

    def observation(self, obs: np.ndarray) -> int:
        """Converts a 1D array of int-door states (0-3) to an integer index

        Args:
            obs (np.ndarray): 1D numpy array of integer door states

        Returns:
            int: integer index from the numpy array
        """
        return int(np.dot(obs, self._powers))
