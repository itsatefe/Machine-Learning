import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mode
from copy import deepcopy
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay, precision_score, recall_score, auc as AUC
from sklearn.tree import DecisionTreeClassifier

class OnevsAll:
    def __init__(self, base_model, **kwargs):
        self.base_model = base_model
        self.kwargs = kwargs

    def fit(self, data, labels):
        self.unique_labels = np.unique(labels)
        self.models = [self.base_model(**self.kwargs) for i in range(len(self.unique_labels))]

        for i, model in enumerate(self.models):
            new_labels = np.where(labels == self.unique_labels[i], 1, 0)
            model.fit(data, new_labels)

    def predict(self, data):
        probs = np.concatenate([model.predict_prob(data)[:, None] for model in self.models], axis = 1)
        preds = self.unique_labels[probs.argmax(axis = 1)]
        return preds

    def score(self, data, labels):
        preds = self.predict(data)
        return np.sum(preds.flatten() == labels.flatten()) / len(labels)
    
    

class OnevsOne:
    def __init__(self, base_model, **kwargs):
        self.base_model = base_model
        self.kwargs = kwargs

    def fit(self, data, labels):
        self.unique_labels = np.unique(labels)
        self.models = {}

        for i in range(len(self.unique_labels) - 1):
            for j in range(i + 1, len(self.unique_labels)):
                ind = np.where(np.bitwise_or(labels == self.unique_labels[i], labels == self.unique_labels[j]))[0]
                new_data = data[ind]
                new_labels = np.where(labels[ind] == self.unique_labels[i], 0, 1)

                model = self.base_model(**self.kwargs)
                model.fit(new_data, new_labels)
                self.models[(i, j)] = model

    def predict(self, data):
        probs = np.concatenate([self.unique_labels[np.array(k)[model.predict(data).astype(int)], None] for k, model in self.models.items()], axis = 1)
        preds = mode(probs, axis = 1)[0]
        return preds

    def score(self, data, labels):
        preds = self.predict(data)
        return np.sum(preds.flatten() == labels.flatten()) / len(labels)