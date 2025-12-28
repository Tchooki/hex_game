"""
Hex Game Graphics
"""
from typing import List, Tuple, Optional
from math import tan, cos, sin, sqrt, pi
from collections import OrderedDict

import numpy as np
# pylint: disable=no-member
import pygame
import pygame.gfxdraw
from pygame import Surface

from game.board import Board, Pos, WHITE, BLACK

class HexBoard:
    """Print 2D board
    """
    MAP_COLOR = {
        1 : (255,255,255),
        -1 : (0,0,0),
    }

    def __init__(self, n=11) -> None:
        pygame.init()
        self.n = n
        self.screen = pygame.display.set_mode((1280, 720), pygame.RESIZABLE)
        pygame.display.set_caption("Jeu de Hex")
        self.clock = pygame.time.Clock()
        self.running = True

        self.board = Board(self.n)
        self.pawn_dict = OrderedDict()

        self.hex_pos : List[List[Tuple[float, float]]] = [[(0,0)] * self.n for _ in range(self.n)]

        self.can_play = True
        self.hover_index = None
        self.turn = WHITE
        self.update_window()

    def handle_events(self):
        """Handle event
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEMOTION:
                self.hover_index = self.closest_hex()
            elif event.type == pygame.WINDOWRESIZED:
                self.update_window()
                print("Resize:", event.x, event.y)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                
                if self.can_play and self.hover_index:
                    pos = Pos(self.hover_index, self.n)
                    if self.board.can_play(pos):
                        self.pawn_dict[pos] = self.board.turn
                        self.board.play(pos)


    def update_window(self,
                       safey : float = 0.12,
                       safex : float = 0.02):
        """Update window settings and positions

        Args:
            safey (float, optional): Safe zone up and down. Defaults to 0.12.
            safex (float, optional): Safe zone left and right. Defaults to 0.02.
        """
        width, height = self.screen.get_width(), self.screen.get_height()
        safe_height = int((1-safey)*height)
        safe_width = int((1-safex)*width)
        diamond_height = safe_height
        x = int(diamond_height*1.1547005383792517)
        if 1.5*x > safe_width:
            x = int(safe_width/1.5)
            diamond_height = int(x/1.1547005383792517)
        
        self.radius = x/(self.n+2)

        center = (width//2, height//2)
        self.diamond_top_left = (center[0]-x*3/4, center[1]-diamond_height/2)
        self.diamond_top_right = (center[0]+x/4, center[1]-diamond_height/2)
        self.diamond_bottom_left = (center[0]-x/4, center[1]+diamond_height/2)
        self.diamond_bottom_right = (center[0]+x*3/4, center[1]+diamond_height/2)

        self.x = x
        self.diamond_height = diamond_height
        self.center = center
        self.hex_radius = self.x/(self.n+1)/sqrt(3)

        # Update Hex center position
        for i in range(self.n):
            t_i = (i+1)/(self.n+1)
            for j in range(self.n):
                t_j = (j+1)/(self.n+1)
                self.hex_pos[i][j] = (
                    self.center[0] - 3*self.x/4*(1-t_i) + +self.x/4*t_i + t_j*self.x/2,
                    self.center[1] + (t_j-1/2)*self.diamond_height
                )

    def closest_hex(self) -> Optional[Tuple]:
        """Test if mouse hover a hexagone

        Returns:
            Optional[Tuple]: id coords if found one, else None
        """
        mx, my = pygame.mouse.get_pos()
        mouse_pos = np.array([mx, my])
        diffs = self.hex_pos - mouse_pos
        diffs = np.sum(diffs**2, axis=2)
        min_index = np.argmin(diffs)
        i, j = np.unravel_index(min_index, diffs.shape)
        if diffs[i][j] <= self.hex_radius**2:
            return (int(i),int(j))
        return None

    def _draw_bg(self):
        """Draw background diamond
        """
        self.screen.fill((254, 194, 31))

        adjacent30 = self.radius/tan(pi/6)
        adjacent60 = self.radius/tan(pi/3)

        center_top_left = (self.diamond_top_left[0] + adjacent30,
                           self.diamond_top_left[1] + self.radius)
        center_bottom_right = (self.diamond_bottom_right[0] - adjacent30,
                               self.diamond_bottom_right[1] - self.radius)
        center_top_right = (self.diamond_top_right[0] - adjacent60,
                            self.diamond_top_right[1] + self.radius)
        center_bottom_left = (self.diamond_bottom_left[0] + adjacent60,
                              self.diamond_bottom_left[1] -self.radius)

        #### BG PARTS  #####
        n_arc_points = 10
        # PART BG
        points = []
        #corner top left
        for angle in np.linspace(-pi/6,pi/6, n_arc_points):
            points.append((
                center_top_left[0]-self.radius*cos(angle),
                center_top_left[1]-self.radius*sin(angle)
            ))
        # PART black
        #corner top left
        points = []
        for angle in np.linspace(-pi/6,pi/6, n_arc_points):
            points.append((
                center_top_left[0]-self.radius*cos(angle),
                center_top_left[1]-self.radius*sin(angle)
            ))
        #center
        #corner bottom right
        for angle in np.linspace(-pi/6,pi/6, n_arc_points):
            points.append((
                center_bottom_right[0]+self.radius*cos(angle),
                center_bottom_right[1]-self.radius*sin(angle)
            ))
        #corner top right
        for angle in np.linspace(-pi/3,-pi/6, n_arc_points):
            points.append((
                center_top_right[0]-self.radius*sin(angle),
                center_top_right[1]-self.radius*cos(angle)
            ))
        #corner bottom left
        for angle in np.linspace(pi/6,pi/3, n_arc_points):
            points.append((
                center_bottom_left[0]-self.radius*sin(angle),
                center_bottom_left[1]+self.radius*cos(angle)
            ))
        pygame.gfxdraw.filled_polygon(self.screen, points, (40,40,40))

        ######

        # PART white
        #corner top left
        points = []
        for angle in np.linspace(pi/6,pi/2, n_arc_points):
            points.append((
                center_top_left[0]-self.radius*cos(angle),
                center_top_left[1]-self.radius*sin(angle)
            ))
        #corner top left
        for angle in np.linspace(0,-pi/6, n_arc_points):
            points.append((
                center_top_right[0]-self.radius*sin(angle),
                center_top_right[1]-self.radius*cos(angle)
            ))
        #corner bottom left
        for angle in np.linspace(pi/6,0, n_arc_points):
            points.append((
                center_bottom_left[0]-self.radius*sin(angle),
                center_bottom_left[1]+self.radius*cos(angle)
            ))
        #corner bottom right
        for angle in np.linspace(-pi/2,-pi/6, n_arc_points):
            points.append((
                center_bottom_right[0]+self.radius*cos(angle),
                center_bottom_right[1]-self.radius*sin(angle)
            ))
        pygame.gfxdraw.filled_polygon(self.screen, points, (255,255,255))

    def _draw_text(self):
        font = pygame.font.SysFont("Liberation Serif", int(self.diamond_height/(1.5*self.n)))

        def render_text(font : pygame.font.Font, n : int) -> Tuple[List[Surface], List[Surface]]:
            ord_a = ord('A')
            letters, numbers = [], []
            for i in range(n):
                l = font.render(chr(ord_a+i), True, (0, 0, 0))
                num = font.render(str(i+1), True, (0,0,0))
                letters.append(l)
                numbers.append(num)
            return letters, numbers

        letters, digits = render_text(font, self.n)

        # Letters & numbers
        for i in range(self.n):
            t_i = (i+1)/(self.n+1)
            pos = (
                self.center[0] - 3*self.x/4*(1-t_i) + +self.x/4*t_i,
                self.diamond_top_left[1] - self.diamond_height/(self.n*3)
                )
            text_rect = letters[i].get_rect(center=(pos[0], pos[1]))
            self.screen.blit(letters[i], text_rect)

            pos = (
                self.center[0] - 3*self.x/4 + t_i*self.x/2 - self.x/(self.n*4),
                self.center[1] + (t_i-1/2)*self.diamond_height
                )
            text_rect = digits[i].get_rect(right=pos[0], centery=pos[1])
            self.screen.blit(digits[i], text_rect)

    def _draw_hex(self):
        """Draw hexagones
        """

        for i in range(self.n):
            t_i = (i+1)/(self.n+1)
            for j in range(self.n):
                t_j = (j+1)/(self.n+1)
                coords = (
                    self.center[0] - 3*self.x/4*(1-t_i) + +self.x/4*t_i + t_j*self.x/2,
                    self.center[1] + (t_j-1/2)*self.diamond_height
                )
                self.hex_pos[i][j] = coords
                points, points_inside = [], []
                percent = 0.06
                for angle in np.arange(pi/2,-3*pi/2,-pi/3):
                    points.append((
                        self.hex_radius*(1+percent)*cos(angle)+coords[0],
                        self.hex_radius*(1+percent)*sin(angle)+coords[1]
                        ))
                    points_inside.append((
                        self.hex_radius*(1-percent)*cos(angle)+coords[0],
                        self.hex_radius*(1-percent)*sin(angle)+coords[1]
                    ))
                pygame.draw.polygon(self.screen, (0,0,0), points)
                pygame.draw.polygon(self.screen, (241, 219,170), points_inside)


    def _draw_debug(self):
        font = pygame.font.SysFont("Liberation Serif", int(self.diamond_height/(2*self.n)))
        for i in range(self.n):
            for j in range(self.n):
                color = (0, 0, 0) if self.board[Pos(i,j)] >= 0 else (255, 255, 255)
                num = font.render(str(self.board[Pos(i,j)]), True, color)
                text_rect = num.get_rect(center=self.hex_pos[i][j])
                self.screen.blit(num, text_rect)

    def _draw_pawn(self):
        """Draw pawns"""
        # Draw hover
        
        if self.can_play:
            if self.hover_index:
                
                time_s = pygame.time.get_ticks()/100
                alpha = (1 + sin(time_s))/2
                color = (30, 30, 30, int(32*alpha)+64) if self.board.turn == BLACK else (255, 255, 255, int(64*alpha)+96)

                i, j = self.hover_index
                diam = self.hex_radius*2
                circle_surf = pygame.Surface((diam, diam), pygame.SRCALPHA)
                pygame.draw.circle(circle_surf, color, (diam//2, diam//2), self.hex_radius/1.5)
                pos = (self.hex_pos[i][j][0] - diam//2,
                    self.hex_pos[i][j][1] - diam//2)
                self.screen.blit(circle_surf, pos)

        # Draw pawns
        for pos, turn in self.pawn_dict.items():
            color = self.MAP_COLOR[turn]
            pygame.draw.circle(self.screen, color, self.hex_pos[pos.x][pos.y], self.hex_radius/1.5)


    def draw_board(self, debug=False):
        """Draw board
        """
        self._draw_bg()
        self._draw_text()
        self._draw_hex()
        self._draw_pawn()
        if debug:
            self._draw_debug()

    def draw(self, debug=False):
        """Draw graphics
        """
        self.draw_board(debug=debug)

    def run(self, debug=False):
        """Run main pygame loop
        """
        while self.running:
            self.handle_events()
            self.draw(debug=debug)
            pygame.display.flip()
            self.clock.tick(60)


if __name__ == '__main__':
    HexBoard().run()
