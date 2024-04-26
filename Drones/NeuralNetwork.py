import numpy as np
import warnings
warnings.filterwarnings("ignore")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class _layer(object):
    def __init__(self, n_in: int, n_out: int):
        self.weights: np.ndarray = np.random.rand(n_out, n_in).astype(dtype=np.float64)
        self.bias: np.ndarray = np.random.rand(n_out, 1).astype(dtype=np.float64)

    def forward(self, inp) -> np.ndarray:
        if type(inp) is not np.ndarray:
            raise TypeError('Input must be of type: Numpy ndarray')
        try:
            return np.dot(self.weights, inp) + self.bias
        except ValueError:
            print('Mismatched layer shape')

    def __call__(self, *args, **kwargs):
        return self.forward(*args)


class neuralNet(object):
    def __init__(self, shape: list[int]):
        self.shape = shape
        self.layers = []
        self.activationFunc = sigmoid
        self.weight_activations = []
        self.node_activations = []
        for i in range(len(shape)-1):
            _l = _layer(shape[i], shape[i+1])
            self.layers.append(_l)

    def forward(self, inp: np.ndarray) -> np.ndarray:
        if type(inp) is np.ndarray:
            x = inp
        else:
            raise TypeError("Input must be of type: Numpy ndarray")

        self.weight_activations = []
        self.node_activations = []

        self.node_activations.append(x/np.abs(x).max())
        for layer in self.layers:
            weight_activations = layer.weights * x.T
            self.weight_activations.append(weight_activations / np.abs(weight_activations).max())
            x = self.activationFunc(layer(x))
            self.node_activations.append(x)

        return x

    def __call__(self, inp: np.ndarray):
        return self.forward(inp)

    @property
    def state_dict(self) -> dict:
        # encode neural net state to a dictionary
        state: dict = {'weights': [], 'bias': []}
        for layer in self.layers:
            state['weights'].append(layer.weights)
            state['bias'].append(layer.bias)
        return state

    def load(self, state_dict: dict):
        # load encoded state to current neural network
        for i in range(len(self.layers)):
            self.layers[i].weights = np.array(state_dict['weights'][i])
            self.layers[i].bias = np.array(state_dict['bias'][i])


if __name__ == '__main__':
    arr1 = np.random.rand(3, 4)
    arr2 = np.random.rand(1, 3)

    print(np.matmul(arr2, arr1), '\n')
    print(np.dot(arr1, arr2))
