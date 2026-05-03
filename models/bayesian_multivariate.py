import numpy as np

class BayesianMultivariate:
    def fit(self, X_train, y_train):
        self._classes = np.unique(y_train)
        self.params = []

        for c in self._classes:
            X_c = X_train[y_train == c]

            self.params.append({
                "mean": np.mean(X_c, axis=0),
                "cov": np.cov(X_c.T),
                "prior": X_c.shape[0] / X_train.shape[0]
            })

    def _pdf(self, x, mean, cov):
        d = len(mean)

        cov = cov + np.eye(d) * 1e-6

        det = np.linalg.det(cov)
        inv = np.linalg.inv(cov)

        diff = (x - mean).reshape(-1, 1)

        exponent = np.exp(-0.5 * diff.T @ inv @ diff)
        denom = ((2 * np.pi) ** (d / 2)) * np.sqrt(det)

        return (exponent / (denom + 1e-9)).item()

    def predict(self, X_test):
        return np.array([self._predict_single(x) for x in X_test])

    def _predict_single(self, x):
        posteriors = []

        for p in self.params:
            prob = np.log(self._pdf(x, p["mean"], p["cov"]) + 1e-9)
            posteriors.append(np.log(p["prior"]) + prob)

        return self._classes[np.argmax(posteriors)]