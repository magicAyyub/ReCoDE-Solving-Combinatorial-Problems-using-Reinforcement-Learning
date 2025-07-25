import numpy as np

epsilon_greedy_cases = [
    # ─── Pure exploitation (ε = 0) ───
    ((np.array([0.0, 1.0]), 0.0, None, 0), 1, "greedy best=1 (seed 0)"),
    ((np.array([1.0, 0.0]), 0.0, None, 1), 0, "greedy best=0 (seed 1)"),

    # ─── Exploitation with a legality mask ───
    ((np.array([0.0, 0.0]), 0.0, np.array([0, 1]), 7),
     1, "mask forces 1 (seed 7)"),

    # ─── Pure exploration (ε = 1) ───
    ((np.array([0.0, 0.0, 0.0]), 1.0, None, 0),
     1, "random choice with seed 0 → first sample is 1"),
]

td_update_cases = [
    # ── 1. reward‑only update (next‑state has no value) ──
    ((np.array([[0.0, 0.0],
                [0.0, 0.0]], dtype=np.float32),
      0, 1, 1.0, 1, None,         0.5, 1.0),
     0.5,
     "α=0.5, γ=1, reward-only"),

    # ── 2. bootstrapping from the best next action ──
    ((np.array([[0.0, 0.0],
                [0.2, 0.8]], dtype=np.float32),
      0, 0, 0.0, 1, None,         1.0, 1.0),
     0.8,
     "target = γ·maxₐ′Q(s',a') = 0.8"),

    # ── 3. legality mask restricts the next‑state arg‑max ──
    ((np.array([[0.0, 0.0],
                [0.4, 0.6]], dtype=np.float32),
      0, 1, 0.0, 1, np.array([1, 0]), 1.0, 1.0),
     0.4,
     "mask forbids action 1 in s'"),

    # ── 4. no legal actions in the next state (next_max = 0) ──
    ((np.array([[0.0, 0.0],
                [0.0, 0.0]], dtype=np.float32),
      0, 0, 2.0, 1, np.array([0, 0]), 0.5, 0.9),
     1.0,
     "empty mask ⇒ bootstrap value 0"),
]