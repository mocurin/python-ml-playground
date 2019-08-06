import functools
import math
import numpy


class neural_network:
    def __init__(self, inp, outp):
        self._shape = (inp, outp)
        self._param = ([], [], [])

    def add_layer(self, string, *, matrix=None, bias=None, func=None):
        if string == 'linear':
            assert matrix and bias and func is None
            self._param[0].append(numpy.array(matrix))
            self._param[1].append(numpy.array(bias))
        elif string == 'function':
            assert func and matrix is None and bias is None
            self._param[2].append(func)

    def check_dimensions(self):
        current = self._shape[0]
        for i in zip(self._param[0], self._param[1]):
            if i[0].shape[1] != current or i[0].shape[0] != i[1].shape[0]:
                return False
            current = i[0].shape[0]
        return current == self._shape[1]

    def compute(self, input):
        # build 2D array: input, then grouped up parameters of nn (are made by transposing)
        # each position is counted with: transposedWeightMatrix * outputFromPrevLayer + biases
        # then squishification function is applied to the result by map()
        # reduce!
        return functools.reduce(
            lambda a, x: list(
                map(
                    x[2],
                    (x[0].dot(a) + x[1])
                )
            ), [input] + [list(i) for i in zip(*self._param)]
        )


def make_feedforward_neural_network(*, input_count, output_count):
    return neural_network(input_count, output_count)

#         O
#   ->O   O   O->
#   ->O   O
nn = make_feedforward_neural_network(input_count=2, output_count=1)
sigmoid = lambda x: 1.0 / (1 + math.exp(-x))
# First layer
nn.add_layer('linear',
             matrix=[[0.1, 0.2],
                     [0.3, 0.4],
                     [0.5, 0.6]],
             bias=[-0.5, -1, -1.5])
nn.add_layer('function', func=sigmoid)
# Second(output) layer
nn.add_layer('linear',
             matrix=[[0.1, 0.2, 0.3]],
             bias=[0.5])
nn.add_layer('function', func=sigmoid)
assert nn.check_dimensions()
input = [0.5, 0.5]
print(nn.compute(input))