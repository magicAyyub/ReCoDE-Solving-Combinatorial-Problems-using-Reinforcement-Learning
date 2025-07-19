"""
A customisable News Vendor problem environment implementation satisfying the Gymnasium interface.

Adapted from the example given by OR-Gym (Balaji et al.), with some cleaned up code and visualisation.
    - Paper: https://arxiv.org/abs/1911.10641
    - GitHub: https://github.com/hubbs5/or-gym/blob/master/or_gym/envs/classic_or/newsvendor.py
"""

from .renderer import NewsVendorPygameRenderer
from dataclasses import dataclass

import numpy as np
from collections.abc import Iterable
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register


# ──────────────────────────────────────────────────────────────────────────────── #
#                                 Configuration                                    #
# ──────────────────────────────────────────────────────────────────────────────── #
@dataclass
class NewsVendorConfig:
    # ── Demand / system limits ──
    lead_time: int = 5                 # periods before an order arrives
    max_inventory: int = 4_000         # cap on pipeline + on‑hand units
    max_order_quantity: int = 2_000    # upper bound for a single action

    # ── Cost parameters (upper bounds for random sampling) ──
    max_sale_price: float = 100.0
    max_holding_cost: float = 5.0
    max_lost_sales_penalty: float = 10.0
    max_demand_mean: float = 200.0              # Poisson mean upper bound

    # ── Episode & discount ──
    max_steps: int = 40
    gamma: float = 1.0                 # discount on purchase cost

class NewsVendorEnv(gym.Env):
    """Gymnasium implementation of the classic multi-period news-vendor problem."""

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        *,
        config: NewsVendorConfig = NewsVendorConfig(),
        render_mode: str | None = None,
    ) -> None:
        """ A Gymnasium interface of the Multi-Period News Vendor with Lead Times from Balaji et. al.

        The News Vendor problem is a seminal problem within inventory management, see: 
            Foundations of Inventory Management (Zipkin, 2000)

        Inventory orders are not instantaneous and have multi-period leadtimes. There are costs inccured for holding
        unsold inventory, although unsold inventory expires at the end of each period. There are also penalties
        associated with losing goodwill by having unsold inventory.

        Observation:
            Type: Box
            State Vector: S = (p, c, h, k, mu, x_l, x_l-1)
            p = price
            c = cost
            h = holding cost
            k = lost sales penalty
            mu = mean of demand distribution
            x_l = order quantities in the queue

        Initial state:
            Parameters p, c, h, k, and mu, with no inventory in the pipeline.

        Actions:
            Type: Box
            Amount of product to order.

        Reward:
            Sales minus discounted purchase price, minus holding costs for
            unsold product or penalties associated with insufficient inventory.

        Episode termination:
            By default, the environment terminates within 40 time steps.
        """
        
        self.config = config
        self.render_mode = render_mode

        # ╭─────────────────── Observation space ───────────────────╮
        # state = [p, c, h, k, μ, pipeline_1,  …, pipeline_n]
        #                          ↑        ↑
        #                          newest   next to arrive
        self.obs_dim = self.config.lead_time + 5
        self.observation_space = spaces.Box(
            low=np.zeros(self.obs_dim, dtype=np.float32),
            high=np.array(
                [self.config.max_sale_price,        # p
                 self.config.max_sale_price,        # c  (bounded by price)
                 self.config.max_holding_cost,      # h
                 self.config.max_lost_sales_penalty,# k
                 self.config.max_demand_mean]                # μ
                + [self.config.max_order_quantity] * self.config.lead_time,
                dtype=np.float32,
            ),
        )

        # ─── Action space ─── #
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([self.config.max_order_quantity], dtype=np.float32),
        )

        # ─── renderer (optional) ───
        self._renderer: NewsVendorPygameRenderer | None = None
        if render_mode is not None:
            self._renderer = NewsVendorPygameRenderer(
                lead_time=self.config.lead_time,
                config=self.config,
                metadata=self.metadata,
                render_mode=self.render_mode,
            )

        # ─── Initialisation ─── #
        self.state = self._reset_state()

    # ──────────────────────────────────────────────────────────────────────────────── #
    #                                Private helpers                                   #
    # ──────────────────────────────────────────────────────────────────────────────── #
   
    def _sample_economic_parameters(self) -> None:
        """Draw random price/cost/penalties for a fresh episode."""
        self.price = max(1.0, self.np_random.random() * self.config.max_sale_price)
        self.cost  = max(1.0, self.np_random.random() * self.price)  # Cannot exceed price
        self.holding_cost_rate     = self.np_random.random() * min(self.cost, self.config.max_holding_cost)
        self.lost_sales_penalty     = self.np_random.random() * self.config.max_lost_sales_penalty
        self.mu    = self.np_random.random() * self.config.max_demand_mean # mu = max demand mean

    def _reset_state(self) -> np.ndarray:
        """Reset economic parameters and clear the pipeline."""
        self._sample_economic_parameters()
        pipeline = np.zeros(self.config.lead_time, dtype=np.float32)
        self.current_step = 0
        return np.concatenate((
            np.array([self.price, self.cost, self.holding_cost_rate, self.lost_sales_penalty, self.mu], dtype=np.float32),
            pipeline,
        ))

    # ──────────────────────────────────────────────────────────────────────────────── #
    #                                 Gymnasium API                                    #
    # ──────────────────────────────────────────────────────────────────────────────── #
    def reset(self, *, seed: int | None = None, options=None):
        """Resets the environment, whenever an episode has terminated or is beginning.

        Args:
            seed (int or None): reset the environment with a specific seed value. Defaults to None.
            options: unused, mandated by the Gymnasium interface. Defaults to None.

        Returns:
            State vector
        """
        super().reset(seed=seed)

        self.state = self._reset_state()
        return self.state

    def step(self, action):
        # ── 1: Ensure scalar float ──
        if isinstance(action, (np.ndarray, list)):
            action = float(np.asarray(action).flatten()[0])
        order_qty = np.clip(
            action,
            0.0,
            min(
                self.config.max_order_quantity,
                self.config.max_inventory - self.state[5:].sum(),
            ),
        )

        # ── 2: Demand for this period ──
        demand = self.np_random.poisson(lam=self.mu)

        # ── 3: Inventory available today ──
        pipeline = self.state[5:]
        inv_on_hand = order_qty if self.config.lead_time == 0 else pipeline[0]

        # ── 4: Sales outcomes ──
        sales_revenue   = min(inv_on_hand, demand) * self.price
        excess_inventory = max(0.0, inv_on_hand - demand)
        short_inventory  = max(0.0, demand - inv_on_hand)

        # ── 5: Costs ──
        purchase_cost      = order_qty * self.cost * (self.config.gamma ** self.config.lead_time)
        holding_cost       = excess_inventory * self.holding_cost_rate
        lost_sales_penalty = short_inventory * self.lost_sales_penalty

        # ── 6: Reward ──
        reward = sales_revenue - purchase_cost - holding_cost - lost_sales_penalty
        if isinstance(reward, Iterable):
            reward = float(np.squeeze(reward))

        # ── 7: Advance pipeline ──
        new_pipeline = np.zeros(self.config.lead_time, dtype=np.float32)
        if self.config.lead_time > 0:
            new_pipeline[:-1] = pipeline[1:]
            new_pipeline[-1]  = order_qty
        self.state = np.hstack([self.state[:5], new_pipeline]).astype(np.float32)

        # ── 8: Episode termination ──
        self.current_step += 1
        terminated = self.current_step >= self.config.max_steps
        truncated  = False  # No time‑limit truncation beyond max_steps

        return self.state, reward, terminated, truncated, {}

    def render(self, mode: str | None = None):
        if self.render_mode is None:
            return None
            
        if not self._renderer:
            raise ValueError("render_mode was None when env was created.")
        return self._renderer.render(self.state)

    def close(self):
        """Closes the environment, performing some basic cleanup of resources."""
        if self._renderer:
            self._renderer.close()
            self._renderer = None

# ──────────────────────────────────────────────────────────────────────────────── #
#               Register so users can call gymnasium.make()                        #
# ──────────────────────────────────────────────────────────────────────────────── #
register(
    id="NewsVendor-v0",
    entry_point="environments.news_vendor.env:NewsVendorEnv",
    max_episode_steps=NewsVendorConfig().max_steps,
)