from sklearn.neighbors import KNeighborsClassifier


class Estimator:
    @staticmethod
    def fit(train_x, train_y):
        return KNeighborsClassifier(n_neighbors=5).fit(train_x, train_y)

    @staticmethod
    def predict(trained, test_x):
        return trained.predict(test_x)
