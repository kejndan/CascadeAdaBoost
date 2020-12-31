from adaboost import *
from utils import *


class CascadeAdaBoost:
    def __init__(self, const_f, const_d, total_fpr, total_dr):
        self.const_f = const_f
        self.const_d = const_d
        self.total_fpr = total_fpr
        self.total_dr = total_dr
        self.adaboosts = []
        self.FPRs = []
        self.DRs = []
        self.idxs_feature_stumps = None

    def fit(self, X, y, path_save=None):
        x_train, y_train = X, y
        if len(self.adaboosts) != 0:
            general_fpr = np.prod(np.array(self.FPRs))
            general_dr = np.prod(np.array(self.DRs))
        else:
            general_fpr = 1
            general_dr = 1
        k = len(self.adaboosts)
        while general_fpr > self.total_fpr:
            ab = AdaBoost()
            weights = np.array([1 / len(x_train) for i in range(len(x_train))])
            adaboost_completed = False
            thr = 0
            thr_step = 0.01
            i = 0
            while not adaboost_completed:
                print(f'Classif {i}')
                weights = ab.help_train(x_train, y_train, weights)
                y_pred, prob = ab.predict(x_train, thr)
                dr = detection_rate(y_pred, y_train)
                while self.const_d > dr:
                    thr -= thr_step
                    y_pred, prob = ab.predict(x_train, thr)
                    dr = detection_rate(y_pred, y_train)
                fpr = false_positive_rate(y_pred, y_train)
                print(f'FPR={fpr} DR={dr}')
                i += 1
                if self.const_f > fpr:
                    adaboost_completed = True
            ab.thr = thr

            self.adaboosts.append(ab)
            y_pred, prob = ab.predict(x_train, ab.thr)
            fpr = false_positive_rate(y_pred, y_train)
            self.FPRs.append(fpr)
            dr = detection_rate(y_pred, y_train)
            self.DRs.append(dr)

            if path_save is not None:
                save(path_save, f'ab_{k}_block', ab)
            general_fpr = np.prod(np.array(self.FPRs))

            print(f'General FPR {general_fpr} {k}')
            k += 1
            y_pred, prob = ab.predict(x_train, ab.thr)
            mask = np.logical_or(np.logical_and(y_pred == 1, y_train == -1), y_train == 1)
            print(f'Sample train before: {len(y_train)} Class 1:{np.sum(y_train==1)} Class 0:{np.sum(y_train==-1)}')
            x_train = x_train[mask]
            y_train = y_train[mask]
            print(f'Sample train before: {len(y_train)} Class 1:{np.sum(y_train==1)} Class 0:{np.sum(y_train==-1)}')

    def predict(self, X):
        prob_general = 1
        for i in range(len(self.adaboosts)):
            y_pred, prob = self.adaboosts[i].predict(X, self.adaboosts[i].thr)
            prob_general *= prob
            if y_pred == -1:
                break
        return y_pred, prob_general

    def get_idxs_feature_stumps(self):
        self.idxs_feature_stumps = []
        for i in range(len(self.adaboosts)):
            self.idxs_feature_stumps.extend(self.adaboosts[i].get_idxs_feature_stumps())
        self.idxs_feature_stumps = set(self.idxs_feature_stumps)
        return self.idxs_feature_stumps

    def load_adaboosts(self, loading_path, X=None, y=None):
        dirs = os.listdir(loading_path)
        for i in range(1, len(dirs) + 1):
            self.adaboosts.append(load(loading_path, f'ab_{i}_block'))
            print(f'Add {i} block of cascade. Numbers of WCL: {len(self.adaboosts[-1].alphs)}')
            if X is not None and y is not None:
                y_pred, _ = self.adaboosts[-1].predict(X)
                self.DRs.append(detection_rate(y_pred, y))
                self.FPRs.append(false_positive_rate(y_pred, y))
        # self.adaboosts[-1].thr = 6
        self.get_idxs_feature_stumps()

