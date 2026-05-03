import numpy as np

class NaiveBayes:
    def fit(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        self._classes = np.unique(y_train)

        n_classes = len(self._classes)

        self._mean = np.zeros((n_classes, n_features))
        self._var = np.zeros((n_classes, n_features))
        self._priors = np.zeros(n_classes)

        for idx, c in enumerate(self._classes):
            X_c = X_train[y_train == c]

            self._mean[idx] = X_c.mean(axis=0)
            self._var[idx] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / n_samples

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx] + 1e-9

        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)

        return numerator / denominator

    def predict(self, X_test):
        return np.array([self._predict_single(x) for x in X_test])

    def _predict_single(self, x):
        posteriors = []

        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x) + 1e-9))
            posteriors.append(prior + posterior)

        return self._classes[np.argmax(posteriors)]