import numpy as np

class Environment:
    def __init__(self):
        self.__board = np.empty((3, 3), dtype = str)
        self.__board.fill('e')
        self.__turn = 'x'

    def get_state(self):
        return self.__board.copy()

    def get_successor(self, action):
        b = self.__board.copy()
        b[action[0], action[1]] = self.__turn
        return b

    def make_action(self, action):
        self.__board[action[0], action[1]] = self.__turn
        self.__turn = 'o' if self.__turn == 'x' else 'x'


    def is_valid_move(self, action):
        if 0 <= action[0] < 3 and 0 <= action[1] < 3:
            return self.__board[action[0], action[1]] == 'e'
        return False
        
    def finish(self):
        if np.any(np.sum(self.__board=='x',axis=0)==3) or np.any(np.sum(self.__board=='x',axis=1)==3) or np.sum(self.__board.diagonal()=='x',axis=0)==3 or np.sum(self.__board[::-1].diagonal()=='x',axis=0)==3:
            return 2 #x win

        elif np.any(np.sum(self.__board=='o',axis=0)==3) or np.any(np.sum(self.__board=='o',axis=1)==3) or np.sum(self.__board.diagonal()=='o',axis=0)==3 or np.sum(self.__board[::-1].diagonal()=='o',axis=0)==3:
            return 3 #o win

        elif np.sum(self.__board == 'e') == 0:
            return 1 #draw

        else:
            return 0 #continue