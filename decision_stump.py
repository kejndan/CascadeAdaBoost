import numpy as np
from config import *


class DecisionStump:
    def __init__(self, weights=None):
        self.weights = weights
        self.error = None
        self.threshold = None
        self.polar = None
        self.idx_feature = None

    def fit(self, X, y, random_sample_features=False, print_info=True):
        nb_features_take = int(X.shape[1]*percent_feature_take/100)
        if random_sample_features:  # Случайный выбор признаков
            working_features = np.random.choice(np.arange(X.shape[1]), nb_features_take, replace=False)
        else:  # Выбор признаков на основе дисперсии
            variance_features = np.var(X, axis=0)
            sorted_indexes_variance_features = np.argsort(variance_features)
            working_features = sorted_indexes_variance_features[-nb_features_take:]

        stumps = []
        for i, index_feature in enumerate(working_features):
            if np.max(X[:, index_feature]) != np.min(X[:, index_feature]):
                stumps.append(self.__search_params_for_stump(index_feature, X, y))
                if print_info and i % 500 == 0:
                    print(f'Stump {i}/{len(working_features)} {stumps[-1]}')

        stumps.sort()
        self.error, self.threshold, self.polar, self.idx_feature = stumps[0]
        if print_info:
            print(f'Best stump:{stumps[0]}')
            print(f'Bad stump:{stumps[-1]}')

    def predict(self, X):
        class_1 = np.where(X[:, self.idx_feature] >= self.threshold)[0]
        class_0 = np.where(X[:, self.idx_feature] < self.threshold)[0]
        if self.polar == -1:
            class_1, class_0 = class_0, class_1
        predicted = np.zeros(len(X))
        predicted[class_1] = 1
        predicted[class_0] = -1
        return predicted

    def __search_params_for_stump(self, idx, X, y):
        feature = X[:, idx]
        step = (np.max(feature) - np.min(feature)) / 100
        threshold = np.min(feature) - 0.01
        min_err = 1

        opt_threshold = threshold
        opt_polar = 1

        while threshold < np.max(feature):
            polar = 1
            class_1 = np.where(X[:, idx] >= threshold)[0]
            class_0 = np.where(X[:, idx] < threshold)[0]

            predicted = np.zeros(len(X))
            predicted[class_1] = 1
            predicted[class_0] = -1

            err = (~np.equal(predicted, y) * self.weights).sum()
            if err > 0.5:
                err = 1 - err
                polar = -1

            if err < min_err:
                opt_threshold = threshold
                opt_polar = polar
                min_err = err

            threshold += step

        return min_err,  opt_threshold, opt_polar, idx

if __name__ == "__main__":
    print('Decision Stump model')