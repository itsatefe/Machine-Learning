import numpy as np
from scipy.stats import mode
import gdown

class NearestNeighbors:
    def fit(self, data):
        self.data = data

    def get_neighbors(self, points, n_neighbors = 1):
        dist = np.stack([np.linalg.norm(self.data - point, axis = 1) for point in points])
        ind = np.argsort(dist, axis = 1)[:, 1:1+n_neighbors]
        return dist[np.arange(len(points))[:, None], ind], ind

class SMOTE:
    def __init__(self, k = 5, threshold = 0.3):
        self.k = k
        self.threshold = threshold

    def fit(self, data, labels):
        self.data = data
        self.labels = labels
        self.nn = NearestNeighbors()
        self.nn.fit(data)
        u, n = np.unique(labels, return_counts = True)
        self.minority = u[n.argmin()]
        self.majority = u[n.argmax()]

    def r1_risk(self, data, labels):
        dist, ind = self.nn.get_neighbors(data, n_neighbors = self.k)
        is_in = (self.labels[ind] == labels[:, None])
        return np.sum(is_in, axis = 1) / self.k

    def _intra_extra(self, data, labels):
        dist, ind = self.nn.get_neighbors(data, n_neighbors = self.k)
        is_in = (self.labels[ind] == labels[:, None])
        return np.sum(is_in.astype(int) * dist, axis = 1) / (np.sum((~is_in).astype(int) * dist, axis = 1) + 1e-9)

    def r2_risk(self, data, labels):
        int_ext = self._intra_extra(data, labels)
        return int_ext / (1 + int_ext)

    def risk_score(self, data, labels):
        r = self.r2_risk(data, labels)
        return np.where(r > self.threshold, (r - self.threshold)/(1 - self.threshold), 0)

    def cls_score(self, data, labels, weights, iteration, max_iter):
        r1 = self.r1_risk(data, labels)
        return ((max_iter - iteration - 2) * r1 + iteration * weights) / max_iter

    def sample_prob(self, data, labels, weights, iteration, max_iter):
        s_c = self.cls_score(data, labels, weights, iteration, max_iter)
        s_r = self.risk_score(data, labels)
        logits = s_c * (1 - s_r)
        return logits / logits.sum()

    def resample(self):
        min_data = self.data[self.labels == self.minority]
        min_labels = self.labels[self.labels == self.minority]
        prob = self.sample_prob(min_data, min_labels, np.ones((min_data.shape[0],)), 0, 1)
        ind_plus = np.random.choice(len(min_data), 2 * np.sum(self.labels == self.majority) - len(self.labels), p = prob)
        dist, ind = self.nn.get_neighbors(min_data, n_neighbors = self.k)
        is_in = (self.labels[ind] == self.minority)
        n_ind = ind[ind_plus]
        n_is_in = is_in[ind_plus]
        samples_plus = []
        for i in range(len(ind_plus)):
            sample = self.data[np.random.choice(n_ind[i][n_is_in[i]])]
            samples_plus.append(min_data[ind_plus[i]] + np.random.rand() * (sample - min_data[ind_plus[i]]))
        samples_plus = np.array(samples_plus)
        new_samples = np.concatenate([self.data, samples_plus], axis = 0)
        new_labels = np.concatenate([self.labels, np.ones((len(samples_plus),))])
        return new_samples, new_labels


class Bagging:
    def __init__(self, base_model, n_models = 5, **kwargs):
        self.base_model = base_model
        self.n_models = n_models
        self.kwargs = kwargs

    def get_params(self, deep = True):
        dic = {'base_model':self.base_model, 'n_models':self.n_models}
        dic.update(self.kwargs)
        return dic

    def set_params(self, **kwargs):
        for parameter, value in kwargs.items():
            setattr(self, parameter, value)
        return self

    def fit(self, data, labels):
        self.models = [self.base_model(**self.kwargs) for i in range(self.n_models)]

        for i, model in enumerate(self.models):
            new_ind = np.random.choice(len(data), len(data))
            new_data = data[new_ind]
            new_labels = labels[new_ind]
            model.fit(new_data, new_labels)

    def predict(self, data):
        probs = np.concatenate([model.predict(data)[:, None] for model in self.models], axis = 1)
        preds = mode(probs, axis = 1)[0]
        return preds

    def predict_prob(self, data):
        probs = np.concatenate([model.predict_prob(data)[:, None] for model in self.models], axis = 1)
        preds = probs.mean(axis = 1)
        return preds

    def score(self, data, labels):
        preds = self.predict(data)
        return np.sum(preds.flatten() == labels.flatten()) / len(labels)

class Boosting:
    def __init__(self, base_model, n_models = 5, **kwargs):
        self.base_model = base_model
        self.n_models = n_models
        self.kwargs = kwargs
        self.smote = SMOTE()

    def get_params(self, deep = True):
        dic = {'base_model':self.base_model, 'n_models':self.n_models}
        dic.update(self.kwargs)
        return dic

    def set_params(self, **kwargs):
        for parameter, value in kwargs.items():
            setattr(self, parameter, value)
        return self

    def fit(self, data, labels):
        self.smote.fit(data, labels)
        self.unique_labels = np.unique(labels)
        self.models = [self.base_model(max_depth = 1, **self.kwargs) for i in range(self.n_models)]
        new_data = data.copy()
        new_labels = labels.copy()
        say_amount = []

        for i, model in enumerate(self.models):
            weights = np.ones_like(labels) / len(labels)
            model.fit(new_data, new_labels)
            preds = model.predict(new_data)
            errors = (preds != new_labels)
            err = np.sum(errors.astype(int) * weights)
            say_amount.append(0.5 * np.log2((1 - err) / (err + 1e-9)) + 1e-9)
            new_weights = weights * np.exp(np.where(errors, 1, -1) * say_amount[-1])
            new_weights = new_weights / np.sum(new_weights)
            prob = self.smote.sample_prob(new_data, new_labels, weights / weights.max(), i, self.n_models)
            new_ind = np.random.choice(len(new_data), len(new_data), p = prob)
            new_data = new_data[new_ind]
            new_labels = new_labels[new_ind]
            self.smote = SMOTE()
            self.smote.fit(new_data, new_labels)
        self.say_amount = np.array(say_amount)

    def predict(self, data):
        probs = np.concatenate([model.predict(data)[:, None] for model in self.models], axis = 1)
        probs = np.sum(np.where(probs.astype(bool), 1, -1) * self.say_amount, axis = 1)
        preds = np.where(probs > 0, 1, 0)
        return preds

    def predict_prob(self, data):
        probs = np.concatenate([model.predict_prob(data)[:, None] for model in self.models], axis = 1)
        probs = np.sum(probs * self.say_amount / self.say_amount.sum(), axis = 1)
        return probs

    def score(self, data, labels):
        preds = self.predict(data)
        return np.sum(preds.flatten() == labels.flatten()) / len(labels)


