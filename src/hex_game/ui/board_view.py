from __future__ import annotations

from collections import OrderedDict
from math import cos, pi, sin, tan

import numpy as np
import pygame
import pygame.gfxdraw
from pygame import Surface

from hex_game.game.board import BLACK, WHITE, Board
from hex_game.ui.hex_utils import (
    calculate_diamond_layout,
    find_closest_hex,
    get_arc_points,
    get_hex_points,
)
from hex_game.ui.players import BasePlayer, HumanPlayer


class HexBoard:
    """Print 2D board"""

    MAP_COLOR = {
        1: (255, 255, 255),
        -1: (0, 0, 0),
    }

    def __init__(
        self,
        n=11,
        player_white: BasePlayer | None = None,
        player_black: BasePlayer | None = None,
    ) -> None:

        pygame.init()
        self.n = n
        self.screen = pygame.display.set_mode((1280, 720), pygame.RESIZABLE)
        pygame.display.set_caption("Jeu de Hex")
        self.clock = pygame.time.Clock()
        self.running = True

        self.board = Board(self.n)
        self.pawn_dict: dict[int, int] = OrderedDict()

        self.hex_pos: list[list[tuple[float, float]]] = [
            [(0, 0)] * self.n for _ in range(self.n)
        ]

        # Players
        self.players = {
            WHITE: player_white or HumanPlayer(),
            BLACK: player_black or HumanPlayer(),
        }
        self.can_play = True
        self.hover_index: tuple[int, int] | None = None
        self.turn = WHITE
        self.quit = False
        self.move_count = 0

        # Geometry placeholders
        self.radius = 0.0
        self.hex_radius = 0.0
        self.center = (0, 0)
        self.x_val = 0
        self.diamond_height = 0
        self.diamond_top_left = (0, 0)
        self.diamond_top_right = (0, 0)
        self.diamond_bottom_left = (0, 0)
        self.diamond_bottom_right = (0, 0)

        self.update_window()

    def handle_events(self):
        """Handle event"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEMOTION:
                self.hover_index = find_closest_hex(
                    pygame.mouse.get_pos(), np.array(self.hex_pos), self.hex_radius
                )
            elif event.type == pygame.WINDOWRESIZED:
                self.update_window()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if (
                    isinstance(self.players[self.board.turn], HumanPlayer)
                    and self.hover_index
                ):
                    i, j = self.hover_index
                    index = i * self.n + j
                    if self.board.can_play(index):
                        self.pawn_dict[index] = self.board.turn
                        self.board.play(index)
                        self.move_count += 1

    def computer_move(self):
        """Request a move from the current player if it's not human."""
        if self.board.has_won != 0:
            return

        current_player = self.players[self.board.turn]
        if isinstance(current_player, HumanPlayer):
            return

        action = current_player.get_move(self.board, move_count=self.move_count)

        if action is not None:
            if isinstance(action, int):
                if self.board.can_play(action):
                    self.pawn_dict[action] = self.board.turn
                    self.board.play(action)
                    self.move_count += 1
            elif action == "reset":
                self.reset()

    def reset(self):
        """Reset board"""
        self.turn = WHITE
        self.board.reset()
        self.pawn_dict = OrderedDict()
        self.move_count = 0

    def update_window(self, safey: float = 0.12, safex: float = 0.02):
        """Update window settings and positions"""
        width, height = self.screen.get_width(), self.screen.get_height()
        layout = calculate_diamond_layout(width, height, self.n, safey, safex)

        self.radius = layout["radius"]
        self.x_val = layout["x_val"]
        self.diamond_height = layout["diamond_height"]
        self.center = layout["center"]
        self.hex_radius = layout["hex_radius"]
        self.diamond_top_left = layout["top_left"]
        self.diamond_top_right = layout["top_right"]
        self.diamond_bottom_left = layout["bottom_left"]
        self.diamond_bottom_right = layout["bottom_right"]

        # Update Hex center position
        for i in range(self.n):
            t_i = (i + 1) / (self.n + 1)
            for j in range(self.n):
                t_j = (j + 1) / (self.n + 1)
                self.hex_pos[i][j] = (
                    self.center[0]
                    - 3 * self.x_val / 4 * (1 - t_i)
                    + self.x_val / 4 * t_i
                    + t_j * self.x_val / 2,
                    self.center[1] + (t_j - 1 / 2) * self.diamond_height,
                )

    def _draw_bg(self):
        """Draw background diamond"""
        self.screen.fill((254, 194, 31))

        adjacent30 = self.radius / tan(pi / 6)
        adjacent60 = self.radius / tan(pi / 3)

        center_top_left = (
            self.diamond_top_left[0] + adjacent30,
            self.diamond_top_left[1] + self.radius,
        )
        center_bottom_right = (
            self.diamond_bottom_right[0] - adjacent30,
            self.diamond_bottom_right[1] - self.radius,
        )
        center_top_right = (
            self.diamond_top_right[0] - adjacent60,
            self.diamond_top_right[1] + self.radius,
        )
        center_bottom_left = (
            self.diamond_bottom_left[0] + adjacent60,
            self.diamond_bottom_left[1] - self.radius,
        )

        n_arc = 10
        # PART black
        points = get_arc_points(center_top_left, self.radius, -pi, -5 * pi / 6, n_arc)
        points = []
        for angle in np.linspace(-pi / 6, pi / 6, n_arc):
            points.append(
                (
                    center_top_left[0] - self.radius * cos(angle),
                    center_top_left[1] - self.radius * sin(angle),
                ),
            )
        for angle in np.linspace(-pi / 6, pi / 6, n_arc):
            points.append(
                (
                    center_bottom_right[0] + self.radius * cos(angle),
                    center_bottom_right[1] - self.radius * sin(angle),
                ),
            )
        for angle in np.linspace(-pi / 3, -pi / 6, n_arc):
            points.append(
                (
                    center_top_right[0] - self.radius * sin(angle),
                    center_top_right[1] - self.radius * cos(angle),
                ),
            )
        for angle in np.linspace(pi / 6, pi / 3, n_arc):
            points.append(
                (
                    center_bottom_left[0] - self.radius * sin(angle),
                    center_bottom_left[1] + self.radius * cos(angle),
                ),
            )
        pygame.gfxdraw.filled_polygon(self.screen, points, (40, 40, 40))

        # PART white
        points = []
        for angle in np.linspace(pi / 6, pi / 2, n_arc):
            points.append(
                (
                    center_top_left[0] - self.radius * cos(angle),
                    center_top_left[1] - self.radius * sin(angle),
                ),
            )
        for angle in np.linspace(0, -pi / 6, n_arc):
            points.append(
                (
                    center_top_right[0] - self.radius * sin(angle),
                    center_top_right[1] - self.radius * cos(angle),
                ),
            )
        for angle in np.linspace(pi / 6, 0, n_arc):
            points.append(
                (
                    center_bottom_left[0] - self.radius * sin(angle),
                    center_bottom_left[1] + self.radius * cos(angle),
                ),
            )
        for angle in np.linspace(-pi / 2, -pi / 6, n_arc):
            points.append(
                (
                    center_bottom_right[0] + self.radius * cos(angle),
                    center_bottom_right[1] - self.radius * sin(angle),
                ),
            )
        pygame.gfxdraw.filled_polygon(self.screen, points, (255, 255, 255))

    def _draw_text(self):
        font = pygame.font.SysFont(
            "Liberation Serif",
            int(self.diamond_height / (1.5 * self.n)),
        )

        letters, digits = self._render_labels(font, self.n)

        # Letters & numbers
        for i in range(self.n):
            t_i = (i + 1) / (self.n + 1)
            pos = (
                self.center[0] - 3 * self.x_val / 4 * (1 - t_i) + self.x_val / 4 * t_i,
                self.diamond_top_left[1] - self.diamond_height / (self.n * 3),
            )
            text_rect = letters[i].get_rect(center=(pos[0], pos[1]))
            self.screen.blit(letters[i], text_rect)

            pos = (
                self.center[0]
                - 3 * self.x_val / 4
                + t_i * self.x_val / 2
                - self.x_val / (self.n * 4),
                self.center[1] + (t_i - 1 / 2) * self.diamond_height,
            )
            text_rect = digits[i].get_rect(right=pos[0], centery=pos[1])
            self.screen.blit(digits[i], text_rect)

    def _render_labels(
        self, font: pygame.font.Font, n: int
    ) -> tuple[list[Surface], list[Surface]]:
        ord_a = ord("A")
        letters, numbers = [], []
        for i in range(n):
            label = font.render(chr(ord_a + i), True, (0, 0, 0))
            num = font.render(str(i + 1), True, (0, 0, 0))
            letters.append(label)
            numbers.append(num)
        return letters, numbers

    def _draw_hex(self):
        """Draw hexagones"""
        for i in range(self.n):
            for j in range(self.n):
                coords = self.hex_pos[i][j]
                points, points_inside = get_hex_points(coords, self.hex_radius)
                pygame.draw.polygon(self.screen, (0, 0, 0), points)
                pygame.draw.polygon(self.screen, (241, 219, 170), points_inside)

    def _draw_debug(self):
        font = pygame.font.SysFont(
            "Liberation Serif",
            int(self.diamond_height / (2 * self.n)),
        )
        for i in range(self.n):
            for j in range(self.n):
                index = i * self.n + j
                color = (0, 0, 0) if self.board[index] >= 0 else (255, 255, 255)
                num = font.render(str(self.board[index]), True, color)
                text_rect = num.get_rect(center=self.hex_pos[i][j])
                self.screen.blit(num, text_rect)

    def _draw_pawn(self):
        """Draw pawns"""
        if self.can_play:
            if self.hover_index:
                time_s = pygame.time.get_ticks() / 100
                alpha = (1 + sin(time_s)) / 2
                color = (
                    (30, 30, 30, int(32 * alpha) + 64)
                    if self.board.turn == BLACK
                    else (255, 255, 255, int(64 * alpha) + 96)
                )

                i, j = self.hover_index
                diam = self.hex_radius * 2
                circle_surf = pygame.Surface((int(diam), int(diam)), pygame.SRCALPHA)
                pygame.draw.circle(
                    circle_surf,
                    color,
                    (int(diam // 2), int(diam // 2)),
                    int(self.hex_radius / 1.5),
                )
                pos_draw = (
                    self.hex_pos[i][j][0] - diam // 2,
                    self.hex_pos[i][j][1] - diam // 2,
                )
                self.screen.blit(circle_surf, pos_draw)

        # Draw pawns
        for index, turn in self.pawn_dict.items():
            pawn_color = self.MAP_COLOR[turn]
            i, j = divmod(index, self.n)
            pygame.draw.circle(
                self.screen,
                pawn_color,
                self.hex_pos[i][j],
                self.hex_radius / 1.5,
            )

    def draw_board(self, debug=False):
        """Draw board"""
        self._draw_bg()
        self._draw_text()
        self._draw_hex()
        self._draw_pawn()
        if debug:
            self._draw_debug()

    def draw(self, debug=False):
        """Draw graphics"""
        self.draw_board(debug=debug)

    def run(self, debug=False):
        """Run main pygame loop"""
        # Main Loop
        while self.running:
            self.computer_move()
            self.handle_events()
            self.draw(debug=debug)
            pygame.display.flip()
            self.clock.tick(60)

        # Quit
        pygame.display.quit()


if __name__ == "__main__":
    HexBoard().run()
