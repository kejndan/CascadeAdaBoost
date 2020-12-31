from cascade_adaboost import *
from config import *
from adaboost import *

if __name__ == "__main__":
    dataset = read_dataset(path_train_dataset)
    X, y = processing_dataset(dataset)
    indexes_nan = np.any(np.isnan(X), axis=1)
    X = X[~indexes_nan]
    y = y[~indexes_nan]

    cascade = CascadeAdaBoost(0.5, 0.995, 0.001, 0.9)
    cascade.load_adaboosts('ab_cascade', X,y)
    cascade.fit(X, y)
