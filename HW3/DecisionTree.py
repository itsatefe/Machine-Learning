import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
import gdown
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import mode

class DecisionTree:
    def __init__(self, min_samples = 11, method = 'c4.5', max_depth = -1):
        self.min_sample = min_samples
        self.method = method
        self.max_depth = max_depth
        self.__rec_fit = self.rec_fit_c4 if method == 'c4.5' else self.rec_fit_cart
        self.__rec_predict = self.rec_predict_c4 if method == 'c4.5' else self.rec_predict_cart
        self.__rec_rules = self.rec_rules_c4 if method == 'c4.5' else self.rec_rules_cart

    def _entropy(self, labels):
        _, c = np.unique(labels, return_counts = True)
        if np.prod(c) == 0:
            return 0.0
        return np.sum(-c * np.log2(c / np.sum(c)) / np.sum(c))

    def _split_info(self, data, split_point = None):
        if not split_point:
            new_data = data.copy()
        else:
            new_data = np.zeros_like(data)
            new_data[data >= split_point] = 1

        _, c = np.unique(new_data, return_counts = True)
        return np.sum(-c * np.log2(c / np.sum(c)) / np.sum(c)) + 1e-9

    def _gain(self, data, labels, d_type):
        cur_ent = self._entropy(labels)
        if d_type == 'disc':
            return [(cur_ent-np.sum([self._entropy(labels[np.where(data==x)])*len(np.where(data==x)[0])/len(labels) for x in np.unique(data)]))/self._split_info(data), np.nan]

        else:
            s_data = np.sort(np.unique(data))
            cut_points = np.mean(np.stack([s_data[:-1], s_data[1:]]), axis = 0) if len(s_data) > 1 else s_data
            gains = [cur_ent-self._entropy(labels[np.where(data < x)])*len(np.where(data < x)[0])/len(labels)-self._entropy(labels[np.where(data >= x)])*len(np.where(data >= x)[0])/len(labels) for x in cut_points]
            max_arg = np.argmax(gains)
            return [gains[max_arg] / self._split_info(data, split_point = cut_points[max_arg]), cut_points[max_arg]]

    def _gini(self, labels):
        _, c = np.unique(labels, return_counts = True)
        return 1 - np.sum((c / len(labels)) ** 2)

    def _impurity(self, data, labels, d_type):
        if d_type == 'disc':
            ginis = np.array([[self._gini(labels[np.where(data==x)])*len(np.where(data==x)[0])/len(labels) + self._gini(labels[np.where(data!=x)])*len(np.where(data!=x)[0])/len(labels), x] for x in np.unique(data)])
            return ginis[ginis[:, 0].argmin()]

        else:
            s_data = np.sort(np.unique(data))
            cut_points = np.mean(np.stack([s_data[:-1], s_data[1:]]), axis = 0) if len(s_data) > 1 else s_data
            ginis = [self._gini(labels[np.where(data < x)])*len(np.where(data < x)[0])/len(labels)+self._gini(labels[np.where(data >= x)])*len(np.where(data >= x)[0])/len(labels) for x in cut_points]
            min_arg = np.argmin(ginis)
            return [ginis[min_arg] ,cut_points[min_arg]]

    def _feature2expand_c4(self, data, labels, d_types):
        gains = np.array([list(self._gain(data[:, i], labels, d_types[i])) for i in range(data.shape[1])])
        ind = gains[:, 0].argmax()
        return gains[ind][0], gains[ind][1], gains[:, 0].argmax()

    def _feature2expand_cart(self, data, labels, d_types):
        ginis = np.array([list(self._impurity(data[:, i], labels, d_types[i])) for i in range(data.shape[1])])
        ind = ginis[:, 0].argmin()
        return ginis[ind][0], ginis[ind][1], ginis[:, 0].argmin()

    def rec_fit_c4(self, data, labels, d_types, f_names, dep):
        if len(data) <= self.min_sample or self._entropy(labels) == 0 or (dep >= 0 and dep >= self.max_depth) or data.shape[1] < 2:
            pred_label = mode(labels)[0]
            return (pred_label, np.sum(labels == pred_label) / len(labels) if pred_label == 1 else 1 - (np.sum(labels == pred_label) / len(labels)))

        gain, split_point, gain_ind = self._feature2expand_c4(data, labels, d_types)
        return {f_names[gain_ind]:{(None, x):self.rec_fit_c4(np.delete(data[data[:, gain_ind] == x], [gain_ind], axis = 1), labels[data[:, gain_ind] == x], d_types[:gain_ind] + d_types[gain_ind+1:], f_names[:gain_ind] + f_names[gain_ind+1:], dep + 1 if dep >= 0 else dep) for x in np.unique(data[:, gain_ind])}} if d_types[gain_ind] == 'disc' else \
        {f_names[gain_ind]:{('<', split_point):self.rec_fit_c4(data[data[:, gain_ind] < split_point], labels[data[:, gain_ind] < split_point], d_types, f_names, dep + 1 if dep >= 0 else dep),
                            ('>=', split_point):self.rec_fit_c4(data[data[:, gain_ind] >= split_point], labels[data[:, gain_ind] >= split_point], d_types, f_names, dep + 1 if dep >= 0 else dep)}}

    def rec_fit_cart(self, data, labels, d_types, f_names, dep):
        if len(data) <= self.min_sample or self._gini(labels) == 0 or (dep >= 0 and dep >= self.max_depth) or (data.shape[1] < 2 and self._gini(data) == 0):
            pred_label = mode(labels)[0]
            return (pred_label, np.sum(labels == pred_label) / len(labels) if pred_label == 1 else 1 - (np.sum(labels == pred_label) / len(labels)))

        gain, split_point, gain_ind = self._feature2expand_cart(data, labels, d_types)
        return {f_names[gain_ind]:{(1, split_point):self.rec_fit_cart(np.delete(data[data[:, gain_ind] == split_point], [gain_ind], axis = 1), labels[data[:, gain_ind] == split_point], d_types[:gain_ind] + d_types[gain_ind+1:], f_names[:gain_ind] + f_names[gain_ind+1:], dep + 1 if dep >= 0 else dep),
                                   (0, split_point):self.rec_fit_cart(data[data[:, gain_ind] != split_point], labels[data[:, gain_ind] != split_point], d_types, f_names, dep + 1 if dep >= 0 else dep)}} if d_types[gain_ind] == 'disc' else \
        {f_names[gain_ind]:{('<', split_point):self.rec_fit_cart(data[data[:, gain_ind] < split_point], labels[data[:, gain_ind] < split_point], d_types, f_names, dep + 1 if dep >= 0 else dep),
                            ('>=', split_point):self.rec_fit_cart(data[data[:, gain_ind] >= split_point], labels[data[:, gain_ind] >= split_point], d_types, f_names, dep + 1 if dep >= 0 else dep)}}


    def fit(self, data, labels):
        self.d_types = ['disc' if len(np.unique(data[:, i])) < 20 and len(np.unique(data[:, i])) < 2*np.log2(len(data)) else 'cont' for i in range(data.shape[1])]
        f_names = np.arange(data.shape[1]).tolist()
        self.tree = self.__rec_fit(data, labels, deepcopy(self.d_types), f_names, 0 if self.max_depth > 0 else -1)

    def rec_predict_c4(self, tree, data, ind):
        if type(tree) != dict:
            return np.stack([ind, np.ones_like(ind) * tree[0], np.ones_like(ind) * tree[1]])

        k = list(tree.keys())[0]
        vals = tree[k]
        return np.concatenate([self.rec_predict_c4(vals[ke], data[np.where(data[:, k] == ke[1])], ind[np.where(data[:, k] == ke[1])]) for ke, v in vals.items()], axis = 1) if self.d_types[k] == 'disc' else \
        np.concatenate([self.rec_predict_c4(vals[list(vals.keys())[0]], data[np.where(data[:, k] < list(vals.keys())[0][1])], ind[np.where(data[:, k] < list(vals.keys())[0][1])]), self.rec_predict_c4(vals[list(vals.keys())[1]], data[np.where(data[:, k] >= list(vals.keys())[1][1])], ind[np.where(data[:, k] >= list(vals.keys())[1][1])])], axis = 1)

    def rec_predict_cart(self, tree, data, ind):
        if type(tree) != dict:
            return np.stack([ind, np.ones_like(ind) * tree[0], np.ones_like(ind) * tree[1]])

        k = list(tree.keys())[0]
        vals = tree[k]
        return np.concatenate([self.rec_predict_cart(vals[ke], data[np.where(np.bitwise_xor(data[:, k] == ke[1], bool(1-ke[0])))], ind[np.where(np.bitwise_xor(data[:, k] == ke[1], bool(1-ke[0])))]) for ke, v in vals.items()], axis = 1) if self.d_types[k] == 'disc' else \
        np.concatenate([self.rec_predict_cart(vals[list(vals.keys())[0]], data[np.where(data[:, k] < list(vals.keys())[0][1])], ind[np.where(data[:, k] < list(vals.keys())[0][1])]), self.rec_predict_cart(vals[list(vals.keys())[1]], data[np.where(data[:, k] >= list(vals.keys())[1][1])], ind[np.where(data[:, k] >= list(vals.keys())[1][1])])], axis = 1)

    def predict(self, data):
        preds = self.__rec_predict(deepcopy(self.tree), data, np.arange(data.shape[0]))
        return preds[1][preds[0].argsort()]

    def predict_prob(self, data):
        preds = self.__rec_predict(deepcopy(self.tree), data, np.arange(data.shape[0]))
        return preds[2][preds[0].argsort()]

    def score(self, data, labels):
        preds = self.predict(data)
        return np.sum(preds == labels) / len(labels)

    def rec_rules_c4(self, tree, rule):
        if type(tree) != dict:
            self.rules.append(rule + f' THEN {tree}')
            return []

        k = list(tree.keys())[0]
        vals = tree[k]
        return [self.rec_rules_c4(vals[ke], (rule + f' AND X{k} IS {ke[1]}') if not ke[0] else (rule + f' AND X{k} {ke[0]} {ke[1]}')) for ke, v in vals.items()]

    def rec_rules_cart(self, tree, rule):
        if type(tree) != dict:
            self.rules.append(rule + f' THEN {tree}')
            return []

        k = list(tree.keys())[0]
        vals = tree[k]
        return [self.rec_rules_cart(vals[ke], (rule + f' AND X{k} {"IS" if ke[0]==1 else "IS NOT"} {ke[1]}') if not type(ke[0])==str else (rule + f' AND X{k} {ke[0]} {ke[1]}')) for ke, v in vals.items()]

    def get_rules(self):
        self.rules = []
        self.__rec_rules(self.tree, '')
        new_rules = [' '.join(x.split(' ')[2:]) for x in self.rules]
        self.rules = new_rules
        return new_rules