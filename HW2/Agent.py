import numpy as np

class Agent:
    def __init__(self, label, model):
        self.label = label
        self.opponent = 'x' if label == 'o' else 'o'
        self.model = model

    def _encode_board(self, board):
        new_board = np.zeros_like(board, dtype = int)
        new_board[board == self.label] = 1
        new_board[board == 'e'] = 0
        new_board[board == self.opponent] = -1
        return new_board.astype(int)

    def _get_features(self, board):
        f1 = np.sum(board.sum(axis=0)==2)+ np.sum(board.sum(axis=1)==2)+ np.sum(board.diagonal().sum(axis=0)==2)+ np.sum(board[::-1].diagonal().sum(axis=0)==2)
        f2 = np.sum(board.sum(axis=0)==-2)+np.sum(board.sum(axis=1)==-2)+np.sum(board.diagonal().sum(axis=0)==-2)+np.sum(board[::-1].diagonal().sum(axis=0)==-2)
        f3 = 1 if board[1, 1] == 1 else 0
        f4 = np.sum([board[0,0]==1, board[0,2]==1, board[2,0]==1, board[2,2]==1])
        f5 = ((board.sum(axis=0)==1)&((board==1).sum(axis=0)==1)).sum()+ ((board.sum(axis=1)==1)&((board==1).sum(axis=1)==1)).sum()+ np.sum((board.diagonal().sum()==1)&((board==1).diagonal().sum()==1))+ np.sum((board[::-1].diagonal().sum()==1)&((board[::-1]==1).diagonal().sum()==1))
        f6 = 1 if any((board.sum(axis=0)==3)|(board.sum(axis=1)==3)) or board.diagonal().sum()==3 or board[::-1].diagonal().sum()==3 else 0
        return np.stack([1,f1,f2,f3,f4,f5,f6])

    def _evaluate_board(self, features):
        return self.model.predict(features)

    def predict_action(self, board):
        curr_board = self._encode_board(board.copy())
        all_actions = np.stack(np.where(curr_board == 0)).T
        action = all_actions[0]
        best = -1e5
        best_f = None
        for act in all_actions:
            b = curr_board.copy()
            b[act[0], act[1]] = 1
            f = self._get_features(b)
            score = self._evaluate_board(f[None])[0,0]
            if score > best:
                action = act
                best = score
                best_f = f
        return action, best_f
