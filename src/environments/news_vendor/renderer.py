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

    def __init__(self, *, lead_time, config, metadata, render_mode):
        import pygame
        self._pg = pygame
        scale       = 1.75
        scale_font  = 1.25
        self._lead_time = lead_time
        self._cfg       = config
        self._fps       = metadata.get("render_fps", 4)
        self._mode      = render_mode

        pygame.init()
        pygame.font.init()

        # --- Layout constants -------------------------------------------------
        self._margin  = int(10 * scale)
        self._cell_w  = int(70 * scale)
        self._cell_h  = int(200 * scale)
        text_area_h   = int(120 * scale)

        # create fonts BEFORE we size the window so we can ask their height
        self._font_big   = pygame.font.SysFont(None, int(30 * scale_font))
        self._font_small = pygame.font.SysFont(None, int(22 * scale_font))

        self._step_lbl_h = self._font_small.get_height() + 6          # <─▼ NEW

        width  = self._margin * 2 + lead_time * self._cell_w
        height = (text_area_h + self._cell_h +
                  self._step_lbl_h +                 # extra room for “t‑*”
                  self._margin * 2)                  # top & bottom margins

        self._surface = pygame.Surface((width, height))
        self._window  = None
        if render_mode == "human":
            self._window = pygame.display.set_mode((width, height), pygame.RESIZABLE)
            pygame.display.set_caption("News Vendor")

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
            f"Lost sales penalty:   {k_pen:6.2f}",
            f"μ (demand mean):  {mu:6.2f}",
        ]
        for i, line in enumerate(txt_lines):
            txt = self._font_big.render(line, True, self._TEXT_COLOR)
            surf.blit(txt, (self._margin, self._margin + i * 24))

        # ── 2. Pipeline visualisation ──
        pipeline = s[5:]
        bar_y_base = surf.get_height() - self._margin - self._step_lbl_h  # <─▲ NEW
        for idx, qty in enumerate(pipeline):
            x      = self._margin + idx * self._cell_w + self._cell_w // 4
            bar_w  = self._cell_w // 2
            slot_r = pg.Rect(x, bar_y_base - self._cell_h, bar_w, self._cell_h)
            pg.draw.rect(surf, self._PIPE_BG,     slot_r)
            pg.draw.rect(surf, self._PIPE_BORDER, slot_r, width=1)

            if qty > 0:
                h = int(qty / self._cfg.max_order_quantity * self._cell_h)
                h = max(1, min(h, self._cell_h))
                pg.draw.rect(surf, self._PIPE_FILL,
                             pg.Rect(x, bar_y_base - h, bar_w, h))

            label = self._font_small.render(str(int(qty)), True, self._TEXT_COLOR)
            surf.blit(label,
                      label.get_rect(midbottom=(x + bar_w // 2,
                                                 bar_y_base - self._cell_h - 4)))

            # step annotation
            step_lbl  = self._font_small.render(f"t-{self._lead_time - idx}",
                                                True, self._TEXT_COLOR)
            surf.blit(step_lbl,
                      step_lbl.get_rect(midtop=(x + bar_w // 2,
                                                bar_y_base + 4)))