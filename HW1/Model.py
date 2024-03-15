import numpy as np

class RoteLearning:
    def fit(self, train_data, train_labels):
        self.train_data = train_data
        self.train_labels = train_labels

    def predict(self, val_data):
        pred = [self.train_labels[np.where(np.all(self.train_data == val_data[i], axis = 1))].tolist() for i in range(len(val_data))]
        return [x[0] if x else None for x in pred]

    
class FindS:
    def fit(self, train_data, train_labels):
        positive = train_data[np.where(train_labels == '+')]
        self.h = np.array([positive[0, i] if np.all(positive[:, i] == positive[0, i]) else '?' for i in range(positive.shape[1])])

    def predict(self, data):
        return np.where(np.all((self.h == data) | (self.h == '?'), axis = 1), '+', '-')


class CandidElimination:
    def __init__(self,fs):
        self.fs = fs
        
        
    def get_all_h(self, uni):
        lens = list(map(len, uni))
        h = [np.concatenate([[uni[i][j]] * int(np.prod(lens[i + 1:])) for j in range(len(uni[i]))]).tolist() * int(np.prod(lens[:i])) for i in range(len(uni))]
        return np.array(h).T

    def evaluate(self, data, h):
        return np.where(np.all((h == data) | (h == '?'), axis = 1), '+', '-')

    def fit(self, train_data, train_labels):
        positive = train_data[np.where(train_labels == '+')]
        negative = train_data[np.where(train_labels == '-')]
        self.specific = np.array([positive[0, i] if np.all(positive[:, i] == positive[0, i]) else '?' for i in range(positive.shape[1])])

        # generating more general h for each h
        uni = np.array(list(zip(self.fs.h.tolist(), ['?'] * len(self.fs.h))))
        general = np.unique(self.get_all_h(uni), axis = 0)[1:-1]
        g = []
        for h in general:
            if np.all(self.evaluate(negative, h) == '-'):
                g.append(h)

        self.general = np.stack(g) if g else np.array([])
        self.vs = np.concatenate([self.specific[None], self.general], axis = 0) if g else self.specific[None]
        self.status = 'consistent' if np.all(self.evaluate(negative, self.specific) == '-') else 'inconsistent'

    def predict(self, val_data):
        preds = np.stack([self.evaluate(val_data, h) for h in self.vs]).T
        final_preds = []
        for p in preds:
            a, b = np.unique(p, return_counts = True)
            if len(a) == 2 and b[0] == b[1]:
                final_preds.append(None)
            else:
                final_preds.append(a[b.argmax()])
        return np.array(final_preds)
    
    
    
    
    
class ListThenEliminate:
    def get_all_h(self, uni):
        lens = list(map(len, uni))
        h = [np.concatenate([[uni[i][j]] * int(np.prod(lens[i + 1:])) for j in range(len(uni[i]))]).tolist() * int(np.prod(lens[:i])) for i in range(len(uni))]
        return np.array(h).T

    def evaluate(self, data, h):
        return np.where(np.all((h == data) | (h == '?'), axis = 1), '+', '-')

    def fit(self, train_data, train_labels):
        uni = [np.unique(train_data[:, i]).tolist() + ['?'] for i in range(train_data.shape[1])]
        all_h = self.get_all_h(uni)
        vs = []

        for h in all_h:
            if np.all(self.evaluate(train_data, h) == train_labels):
                vs.append(h)

        self.vs = np.stack(vs) if vs else np.array([])

    def predict(self, val_data):
        preds = np.stack([self.evaluate(val_data, h) for h in self.vs]).T
        final_preds = []
        for p in preds:
            a, b = np.unique(p, return_counts = True)
            if len(a) == 2 and b[0] == b[1]:
                final_preds.append(None)
            else:
                final_preds.append(a[b.argmax()])
        return np.array(final_preds)