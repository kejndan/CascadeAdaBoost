from utils import *
from decision_stump import *
import time


class AdaBoost:
    def __init__(self):
        self.stumps = []
        self.alphs = []
        self.T = None
        self.score_train = [[], [], [], [], []]
        self.score_valid = [[], [], [], [], []]
        self.thr = 0
        self.idxs_feature_stumps = None

    def fit(self, X, y, nums_iter, x_valid=None, y_valid=None, begin_iter=0):
        self.T = nums_iter
        weights = np.array([1 / len(X) for i in range(len(X))])
        for t in range(begin_iter, nums_iter):
            print(f'Num of iteration: {t}')
            start_iter = time.time()
            weights = self.help_train(X, y, weights)
            print(f'Time iter{(time.time() - start_iter) / 60} min')
            if x_valid is not None and y_valid is not None:
                self.__get_stat(X, y, x_valid, y_valid, t)

    def help_train(self, X, y, weights):
        stump = DecisionStump(weights)
        stump.fit(X, y)

        self.stumps.append(stump)
        y_pred = stump.predict(X)
        err = (~np.equal(y_pred, y) * weights).sum()

        alph = 1 / 2 * np.log((1 - err) / err)
        self.alphs.append(alph)
        weights_new = []

        for i in range(len(weights)):
            weights_new.append(weights[i] / 2
                               * (1 / err * (y_pred[i] != y[i]) + 1 / (1 - err) * (y_pred[i] == y[i])))
        weights = np.array(weights_new)
        return weights

    def __get_stat(self, X, y, x_valid, y_valid, iter):
        y_pred = self.predict(X, self.thr)[0]
        print('Train set')
        precision_train, recall_train, accuracy_train, fpr_train = eval_metrics(y_pred, y, printed=True)
        self.score_train[0].append(iter)
        self.score_train[1].append(precision_train)
        self.score_train[2].append(recall_train)
        self.score_train[3].append(accuracy_train)
        self.score_train[4].append(fpr_train)

        y_pred = self.predict(x_valid, self.thr)[0]
        print('Valid set')
        precision_valid, recall_valid, accuracy_valid, fpr_valid = eval_metrics(y_pred, y_valid, printed=True)

        self.score_valid[0].append(iter)
        self.score_valid[1].append(precision_valid)
        self.score_valid[2].append(recall_valid)
        self.score_valid[3].append(accuracy_valid)
        self.score_valid[4].append(fpr_valid)

        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(15, 15))
        axes[0, 0].plot(self.score_train[0], self.score_train[1], color='red', linestyle='--', label='Train precision')
        axes[0, 1].plot(self.score_train[0], self.score_train[2], color='blue', linestyle='--', label='Train recall')
        axes[1, 0].plot(self.score_train[0], self.score_train[3], color='green', linestyle='--', label='Train accuracy')
        axes[1, 1].plot(self.score_train[0], self.score_train[4], color='orange', linestyle='--',
                        label='Train False positive rate')

        axes[0, 0].plot(self.score_valid[0], self.score_valid[1], color='red', label='Valid precision')
        axes[0, 0].legend()
        axes[0, 1].plot(self.score_valid[0], self.score_valid[2], color='blue', label='Valid recall')
        axes[0, 1].legend()
        axes[1, 0].plot(self.score_valid[0], self.score_valid[3], color='green', label='Valid accuracy')
        axes[1, 0].legend()
        axes[1, 1].plot(self.score_valid[0], self.score_valid[4], color='orange',
                        label='Valid False positive rate')
        axes[1, 1].legend()

        axes[0, 0].set_title(f'Precision(Valid = {np.round(self.score_valid[1][-1], 3)})')
        axes[0, 1].set_title(f'Recall(True positive rate)(Valid = {np.round(self.score_valid[2][-1], 3)})')
        axes[1, 0].set_title(f'Accuracy(Valid = {np.round(self.score_valid[3][-1], 3)})')
        axes[1, 1].set_title(f'False positive rate(Valid = {np.round(self.score_valid[4][-1], 3)})')

        plt.show()

    def predict(self, X, thr=0):
        s = 0
        for t in range(len(self.alphs)):
            s += self.alphs[t] * self.stumps[t].predict(X)
        return np.sign(s - thr), 1 / (1 + np.exp(-s+thr))

    def get_idxs_feature_stumps(self):
        self.idxs_feature_stumps = []
        for i in range(len(self.stumps)):
            self.idxs_feature_stumps.append(self.stumps[i].idx_feature)
        return self.idxs_feature_stumps


if __name__ == "__main__":
    print()
    # dataset = read_dataset('faces/train.pickle')
    # print(dataset.shape)
    # X, y = processing_dataset(dataset)
    # indexes_nan = np.any(np.isnan(X), axis=1)
    # X = X[~indexes_nan]
    # y = y[~indexes_nan]
    # x_train, x_valid, y_train, y_valid = splitter(X, y, 10)
    # # nb_features_take = 10
    # # # y[-2429:] = 1.0
    # # #
    # # # # x= np.array([6,5,2,4,8,1,2,5,10,7])
    # # # # y=np.array([1,0,0,0,1,0,0,0,1,1])
    # # # # weights = np.array([1/len(x) for i in range(len(x))])
    # # # # print(search_params_for_stump(x,y,weights))
    # # # print(X.shape, y[:,np.newaxis].shape)
    # # # with open('train.pickle', 'wb') as f:
    # # #     pickle.dump(np.concatenate((X, y[:,np.newaxis]), axis=1), f)
    # # create_stump(X,y, )
    # ab = AdaBoost()
    # ab.train(x_train, y_train,15, x_valid, y_valid)
    # save('adaboosts','ada1', ab)
    # # ab = load('adaboosts','ada1')
    # # print(ab.test(x_valid))
