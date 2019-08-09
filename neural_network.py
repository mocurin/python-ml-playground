import functools
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
