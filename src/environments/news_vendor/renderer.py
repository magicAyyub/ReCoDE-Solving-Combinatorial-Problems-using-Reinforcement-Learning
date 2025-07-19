from typing import Literal

import numpy as np


class NewsVendorPygameRenderer:
    """Self-contained PyGame renderer for the multi-period News Vendor Gymnasium environment.

    The frame shows two sections:
        • Economic parameters (price, cost, holding cost, lost-sales penalty, μ)
        • A horizontal bar chart representing the order-pipeline.  Each bar's height
          is proportional to the corresponding order quantity relative to
          ``config.max_order_quantity``.
    """

    _BG_COLOR = (30, 30, 30)
    _TEXT_COLOR = (230, 230, 230)
    _PIPE_BG = (60, 60, 60)
    _PIPE_FILL = (80, 160, 240)
    _PIPE_BORDER = (0, 0, 0)

    def __init__(
        self,
        *,
        lead_time: int,
        config,
        metadata: dict,
        render_mode: Literal["human", "rgb_array"],
    ) -> None:
        import pygame  # defer heavy import until absolutely necessary

        self._pg = pygame
        self._lead_time = lead_time
        self._cfg = config
        self._fps = metadata.get("render_fps", 4)
        self._mode = render_mode

        pygame.init()
        pygame.font.init()

        # ── Layout constants ──
        self._margin = 10
        self._cell_w = 70
        self._cell_h = 200  # max bar height
        text_area_h = 120

        width = self._margin * 2 + lead_time * self._cell_w
        height = text_area_h + self._cell_h + self._margin * 2

        self._surface = pygame.Surface((width, height))
        self._window = None
        if render_mode == "human":
            self._window = pygame.display.set_mode((width, height))
            pygame.display.set_caption("News Vendor")

        self._font_big = pygame.font.SysFont(None, 30)
        self._font_small = pygame.font.SysFont(None, 22)

    # ────────────────────────────────────────────────────────────────────────── #
    #  Public API                                                              #
    # ────────────────────────────────────────────────────────────────────────── #
    def render(self, state: np.ndarray) -> None | np.ndarray:
        """Render **one** frame from the environment state.

        Args:
            state (np.ndarray): shape ``(5 + lead_time,)``

        Returns:
            If ``render_mode == 'rgb_array'``, a ``H×W×3`` uint8 RGB array suitable
            for video capture.
        """
        self._draw_frame(state)

        if self._mode == "human":
            self._window.blit(self._surface, (0, 0))
            self._pg.display.flip()
            self._pg.time.delay(int(1000 / self._fps))
            return None

        arr = self._pg.surfarray.array3d(self._surface)
        return np.transpose(arr, (1, 0, 2))

    def close(self) -> None:
        """Release PyGame resources."""
        if self._pg:
            self._pg.quit()
        self._pg = self._window = self._surface = self._font_big = self._font_small = None

    # ────────────────────────────────────────────────────────────────────────── #
    #  Internal helpers                                                         #
    # ────────────────────────────────────────────────────────────────────────── #
    def _draw_frame(self, s: np.ndarray) -> None:
        pg = self._pg
        surf = self._surface
        surf.fill(self._BG_COLOR)

        # ── 1. Economic parameters ──
        price, cost, h_cost, k_pen, mu = s[:5]
        txt_lines = [
            f"Price:          {price:6.2f}",
            f"Cost:           {cost:6.2f}",
            f"Holding cost:   {h_cost:6.2f}",
            f"Lost‑sales pen.:{k_pen:6.2f}",
            f"μ (demand mean):{mu:6.2f}",
        ]
        for i, line in enumerate(txt_lines):
            txt = self._font_big.render(line, True, self._TEXT_COLOR)
            surf.blit(txt, (self._margin, self._margin + i * 24))

        # ── 2. Pipeline visualisation ──
        pipeline = s[5:]
        bar_y_base = surf.get_height() - self._margin  # bottom of bars
        for idx, qty in enumerate(pipeline):
            x = self._margin + idx * self._cell_w + self._cell_w // 4
            bar_w = self._cell_w // 2
            # background slot
            slot_rect = pg.Rect(x, bar_y_base - self._cell_h, bar_w, self._cell_h)
            pg.draw.rect(surf, self._PIPE_BG, slot_rect)
            pg.draw.rect(surf, self._PIPE_BORDER, slot_rect, width=1)

            # filled portion
            if qty > 0:
                h = int(qty / self._cfg.max_order_quantity * self._cell_h)
                h = max(1, min(h, self._cell_h))
                fill_rect = pg.Rect(x, bar_y_base - h, bar_w, h)
                pg.draw.rect(surf, self._PIPE_FILL, fill_rect)

            # numeric label above bar
            label = self._font_small.render(str(int(qty)), True, self._TEXT_COLOR)
            label_rect = label.get_rect(midbottom=(x + bar_w // 2, bar_y_base - self._cell_h - 4))
            surf.blit(label, label_rect)

            # step annotation
            step_lbl = self._font_small.render(f"t-{self._lead_time - idx}", True, self._TEXT_COLOR)
            step_rect = step_lbl.get_rect(midtop=(x + bar_w // 2, bar_y_base + 4))
            surf.blit(step_lbl, step_rect)
