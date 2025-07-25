import numpy as np

def epsilon_greedy(
    values: np.ndarray,
    epsilon: float,
    mask: np.ndarray | None = None,
    seed: int | None = None,
) -> int:
    """Return an action index drawn from an ε-greedy policy.

    Args:
        q_values (1d NumPy array): current estimated q-values.
        epsilon (float scalar): Exploration rate between 0 and 1
        mask (np.ndarray | None, optional):
            Binary vector marking **legal** actions (``1`` → legal, ``0`` → illegal).
            If *None* every action is assumed to be legal.
        seed (int | None, optional):
            If provided, a fresh RNG initialised with this seed is used, making
            the call fully deterministic.
    Returns:
            int: Index of the chosen action.
    """
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    # Determine the set of legal actions --------------------------------------
    legal = np.arange(values.size) if mask is None else np.flatnonzero(mask)
    if legal.size == 0:
        raise ValueError("No legal actions available")

    # Exploration -------------------------------------------------------------
    if rng.random() < epsilon:
        return int(rng.choice(legal))

    # Exploitation ------------------------------------------------------------
    best_idx_within_legal = legal[np.argmax(values[legal])]
    return int(best_idx_within_legal)

def td_update(
    q_table: np.ndarray,
    state_idx: int,
    action_idx: int,
    reward: float,
    next_state_idx: int,
    next_mask: np.ndarray | None = None,
    learning_rate: float = 0.1,
    discount_factor: float = 0.99,
) -> float:
    """Perform one 1-step TD (Q-learning) update.

    Args:
        q_table (np.ndarray): Q-value table with shape ``(num_states, num_actions)``.
            The updated value is written in-place.
        state_idx (int): Index of the current state $s_t$.
        action_idx (int): Index of the action taken $a_t$.
        reward (float): Immediate reward $r_{t+1}$.
        next_state_idx (int): Index of the next state $s_{t+1}$.
        next_mask (np.ndarray | None, optional): Binary mask for legal actions
            in $s_{t+1}$ (``1``= legal, 0 = illegal). If *None*,
            every action is considered legal.
        learning_rate (float, optional): Step‑size $\alpha$.
            Defaults to ``0.1``.
        discount_factor (float, optional): Discount factor $\gamma$.
            Defaults to ``0.99``.

    Returns:
        float: The **new** Q-function ``q_table[state_idx, action_idx]``.

    Raises:
        ValueError: If *next_mask* is provided but has no ``1`` entries, i.e.
            there are no legal actions in the next state.
    """
    # ── 1. Determine bootstrap value ─────────────────────────────────────────
    if next_mask is None:
        next_max = float(np.max(q_table[next_state_idx]))
    else:
        legal = np.flatnonzero(next_mask)
        if legal.size == 0:
            next_max = 0.0  # As in original tabular_q_learning.py
        else:
            next_max = float(np.max(q_table[next_state_idx, legal]))

    td_target = reward + discount_factor * next_max

    # ── 2. Compute TD error and apply update ────────────────────────────────
    td_error = td_target - q_table[state_idx, action_idx]
    q_table[state_idx, action_idx] += learning_rate * td_error

    return float(q_table[state_idx, action_idx])