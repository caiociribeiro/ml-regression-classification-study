import numpy as np


# Funções de ativação e derivadas


def _sigmoid(x):
    # clip previne overflow em exp
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _sigmoid_deriv(x):
    s = _sigmoid(x)
    return s * (1.0 - s)


def _tanh(x):
    return np.tanh(x)


def _tanh_deriv(x):
    return 1.0 - np.tanh(x) ** 2


def _relu(x):
    return np.maximum(0.0, x)


def _relu_deriv(x):
    return (x > 0).astype(float)


class MLPRegressor:
    def __init__(
        self,
        hidden_layer_sizes=(100,),
        lr=0.01,
        epochs=200,
        activation="sigmoid",
        random_state=None,
    ):
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.lr = lr
        self.epochs = epochs
        self.activation = activation
        self.random_state = random_state
        self.coefs_ = []
        self.intercepts_ = []

    def _act(self, x):
        if self.activation == "tanh":
            return _tanh(x)
        if self.activation == "relu":
            return _relu(x)
        return _sigmoid(x)

    def _act_deriv(self, x):
        if self.activation == "tanh":
            return _tanh_deriv(x)
        if self.activation == "relu":
            return _relu_deriv(x)
        return _sigmoid_deriv(x)

    def _init_weights(self, n_inputs, n_outputs):
        rng = np.random.RandomState(self.random_state)
        layer_sizes = [n_inputs] + list(self.hidden_layer_sizes) + [n_outputs]

        self.coefs_ = []
        for a, b in zip(layer_sizes[:-1], layer_sizes[1:]):
            # Inicialização de Glorot/Xavier: distribui o gradiente
            # uniformemente entre camadas, evitando vanishing/exploding
            limit = np.sqrt(6.0 / (a + b))
            self.coefs_.append(rng.uniform(-limit, limit, size=(a, b)))

        self.intercepts_ = [np.zeros(b) for b in layer_sizes[1:]]

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        n_samples, n_features = X.shape
        n_outputs = 1 if y.ndim == 1 else y.shape[1]
        y2d = y.reshape(-1, n_outputs)

        self._init_weights(n_features, n_outputs)

        for _ in range(self.epochs):
            # Forward pass
            activations = [X]
            zs = []  # pré-ativações das camadas ocultas

            for W, b in zip(self.coefs_[:-1], self.intercepts_[:-1]):
                z = activations[-1].dot(W) + b
                zs.append(z)
                activations.append(self._act(z))

            # camada de saída: linear (regressão)
            outputs = activations[-1].dot(self.coefs_[-1]) + self.intercepts_[-1]

            # Backward pass
            # gradiente do MSE: dL/d_output = (2/n) * (ŷ - y)
            upstream = (2.0 / n_samples) * (outputs - y2d)

            grads_W = []
            grads_b = []

            # gradiente da camada de saída
            # dL/dW = activations[-1]ᵀ @ upstream  (soma sobre amostras)
            # dL/db = Σ upstream  (soma — upstream já tem fator 1/n de MSE)
            grads_W.append(activations[-1].T.dot(upstream))
            grads_b.append(np.sum(upstream, axis=0))  # FIX: era np.mean → escala errada

            # gradiente das camadas ocultas (última → primeira)
            for l in range(len(self.coefs_) - 2, -1, -1):
                upstream = upstream.dot(self.coefs_[l + 1].T) * self._act_deriv(zs[l])
                grads_W.append(activations[l].T.dot(upstream))
                grads_b.append(np.sum(upstream, axis=0))

            # reverter para ordem crescente (camada 0 → camada L)
            grads_W = grads_W[::-1]
            grads_b = grads_b[::-1]

            all_grads = grads_W + grads_b
            total_norm = np.sqrt(sum(np.sum(g**2) for g in all_grads))
            max_norm = 5.0
            if total_norm > max_norm:
                clip = max_norm / (total_norm + 1e-8)
                grads_W = [g * clip for g in grads_W]
                grads_b = [g * clip for g in grads_b]

            # ── Atualização SGD ───────────────────────────────────────────────
            for i in range(len(self.coefs_)):
                self.coefs_[i] -= self.lr * grads_W[i]
                self.intercepts_[i] -= self.lr * grads_b[i]

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        a = X
        for W, b in zip(self.coefs_[:-1], self.intercepts_[:-1]):
            a = self._act(a.dot(W) + b)
        return (a.dot(self.coefs_[-1]) + self.intercepts_[-1]).ravel()
