from typing import List
import numpy as np

def has_converged(reward_history: list[float], window_size: int = 100) -> bool:
    """Return ``True`` if recent rewards are both high and stable.

    Args:
        reward_history (list[float]): Episode-level rewards collected so far.
        window_size (int, optional): Number of most-recent episodes to inspect.
            Defaults to ``100``.

    Returns:
        bool: ``True`` when the standard deviation of recent rewards is below
        ``1e-3`` *and* their mean exceeds ``0.9``.
    """
    if len(reward_history) < window_size:
        return False
    recent = reward_history[-window_size:]
    return np.std(recent) < 1e-3 and np.mean(recent) > 0.9