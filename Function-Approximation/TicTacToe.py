import pygame as pg
import sys
from Environment import Environment
from Agent import Agent

class TicTacToe:
    def __init__(self, model):
        self.WIN_SIZE = 500
        self.CELL_SIZE = self.WIN_SIZE // 3
        self.VEC2 = pg.math.Vector2

        pg.init()
        self.screen = pg.display.set_mode([self.WIN_SIZE] * 2)
        pg.display.set_caption('Tic Tac Toe')
        self.clock = pg.time.Clock()

        self.board_image = self.get_scaled_image('resources/field.png', [self.WIN_SIZE] * 2)
        self.o_image = self.get_scaled_image('resources/o.png', [self.CELL_SIZE] * 2)
        self.x_image = self.get_scaled_image('resources/x.png', [self.CELL_SIZE] * 2)

        self.env = Environment()
        self.ag1 = Agent('x', model)
        self.ag2 = Agent('o', model)
        self.agents = [self.ag1, self.ag2]
        self.features = [[], []]
        self.turn = 0
        self.model = model

    def get_scaled_image(self, path, res):
        img = pg.image.load(path)
        return pg.transform.smoothscale(img, res)

    def display_message(self, message):
        if not pg.font.get_init():
            pg.font.init()
        font = pg.font.SysFont('Verdana', self.CELL_SIZE // 8, True)
        text = font.render(message, True, (255, 255, 255))  # White text
        self.screen.blit(text, (self.WIN_SIZE // 2 - text.get_width() // 2, self.WIN_SIZE // 4))
        pg.display.flip()

    def wait_for_player_decision(self):
        while True:
            for event in pg.event.get():
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_y:
                        return True
                    elif event.key == pg.K_n:
                        return False
                elif event.type == pg.QUIT:
                    pg.quit()
                    sys.exit()

    def draw_board(self, board):
        self.screen.blit(self.board_image, (0, 0))
        for y in range(3):
            for x in range(3):
                if board[y][x] == 'x':
                    self.screen.blit(self.x_image, self.VEC2(x, y) * self.CELL_SIZE)
                elif board[y][x] == 'o':
                    self.screen.blit(self.o_image, self.VEC2(x, y) * self.CELL_SIZE)

    def game_loop(self):
        running = True
        while running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                elif event.type == pg.MOUSEBUTTONDOWN and self.turn == 0:
                    current_cell = self.VEC2(pg.mouse.get_pos()) // self.CELL_SIZE
                    col, row = map(int, current_cell)
                    action = [row, col]
                    if self.env.is_valid_move(action):
                        self.features[self.turn].append(action)
                        self.env.make_action(action)
                        self.turn = 1 - self.turn

            stat = self.env.finish()
            if stat == 0 and self.turn == 1:
                self.clock.tick(60)
                curr_board = self.env.get_state()
                action, feature = self.agents[self.turn].predict_action(curr_board)
                self.features[self.turn].append(feature)
                self.env.make_action(action)
                self.turn = 0

            self.draw_board(self.env.get_state())
            pg.display.flip()

            stat = self.env.finish()
            if stat > 0:
                if stat in [1, 2, 3]:
                    messages = {
                        1: "A draw! Play again? (Y/N)",
                        2: "You Won! Play again? (Y/N)",
                        3: "Game Over! Play again? (Y/N)"
                    }
                    self.display_message(messages[stat])
                    if self.wait_for_player_decision():
                        self.env = Environment()
                        self.ag1 = Agent('x', self.model)
                        self.ag2 = Agent('o', self.model)
                        self.agents = [self.ag1, self.ag2]
                        self.features = [[], []]
                        self.turn = 0
                        continue
                    else:
                        running = False

            self.clock.tick(60)

        pg.quit()
        sys.exit()