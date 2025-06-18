"""renderer.py

A minimal PyGame renderer for the Monty Hall environment.

Usage (inside an env):
    self.renderer = PygameRenderer(n_doors, metadata, render_mode)
    ...
    rgb = self.renderer.render(state)      # returns np.ndarray if render_mode=="rgb_array"
    self.renderer.close()                  # clean-up
"""
from .state import DoorState

from typing import Literal, Optional

import numpy as np


class MontyHallPygameRenderer:
    """Self-contained PyGame renderer.

    Parameters
    ----------
    n_doors : int
        Number of doors to draw.
    metadata : Mapping
        Environment metadata containing "render_fps".
    render_mode : {"human", "rgb_array"}
        Behaviour on each `render()` call.
    """

    def __init__(
        self,
        n_doors: int,
        metadata: dict,
        render_mode: Literal["human", "rgb_array"],
    ):
        import pygame # Lazily imported, as and when required

        self._pygame = pygame
        self._n_doors = n_doors
        self._fps = metadata["render_fps"]
        self._mode = render_mode

        pygame.init()
        pygame.font.init()

        width = n_doors * 80 + 20
        height = 160

        self._font = pygame.font.SysFont(None, 36)
        self._surface = pygame.Surface((width, height))

        self._window = None
        if render_mode == "human":
            self._window = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Monty Hall")

    # ──────────────────────────────────────────────────────────────────────────────── #
    #                                 Public API                                       #
    # ──────────────────────────────────────────────────────────────────────────────── #
    def render(self, state: np.ndarray) -> Optional[np.ndarray]:
        """Draw `state` and, depending on mode, either display or return RGB."""
        self._draw_frame(state)

        if self._mode == "human":
            self._window.blit(self._surface, (0, 0))
            self._pygame.display.flip()
            self._pygame.time.delay(int(1000 / self._fps))
            return None

        # rgb_array
        arr = self._pygame.surfarray.array3d(self._surface)  # (W,H,3)
        return np.transpose(arr, (1, 0, 2))  # (H,W,3)

    def close(self) -> None:
        self._pygame.quit()
        self._pygame = self._window = self._surface = self._font = None

    # ──────────────────────────────────────────────────────────────────────────────── #
    #                                 Private helpers                                  #
    # ──────────────────────────────────────────────────────────────────────────────── #
    def _draw_frame(self, state: np.ndarray) -> None:
        pg = self._pygame
        self._surface.fill((30, 30, 30))

        for idx, symbol in enumerate(state):
            x = 10 + idx * 80
            y = 20
            rect = pg.Rect(x, y, 60, 100)

            match DoorState(symbol):
                case DoorState.CLOSED:
                    color = (160, 160, 160)
                    pg.draw.rect(self._surface, color, rect)
                case DoorState.GOAT:
                    color = (240, 240, 240)
                    pg.draw.rect(self._surface, color, rect)
                    txt = self._font.render("G", True, (0, 0, 0))
                    self._surface.blit(txt, txt.get_rect(center=rect.center))
                case DoorState.CAR:
                    color = (240, 240, 240)
                    pg.draw.rect(self._surface, color, rect)
                    txt = self._font.render("C", True, (0, 0, 0))
                    self._surface.blit(txt, txt.get_rect(center=rect.center))
                case DoorState.CHOSEN:
                    color = (80, 160, 240)
                    pg.draw.rect(self._surface, color, rect)

            pg.draw.rect(self._surface, (0, 0, 0), rect, width=2)
