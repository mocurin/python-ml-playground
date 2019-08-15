import functools
import numpy


def linear_handler(parameters, input_data):
    return numpy.array(parameters['matrix']).dot(input_data) + parameters['bias']


def function_handler(parameters, input_data):
    return list(map(parameters['func'], input_data))


handlers = {'linear': linear_handler, 'function': function_handler}


class layer:
    def __init__(self, layer_type, parameters):
        self._parameters = parameters
        self._handler = handlers[layer_type]

    def __call__(self, input_data):
        return self._handler(self._parameters, input_data)


def make_feedforward_neural_network(*, input_count, output_count):
    return neural_network(input_count, output_count)


class neural_network:
    def __init__(self, input_count, output_count):
        self._shape = self._input, self._output = input_count, output_count
        self._layers = []

    def add_layer(self, layer_type, **parameters):
        self._layers.append(layer(layer_type, parameters))

    def check_dimensions(self):
        try:
            result = self([0] * self._input)
        except:
            return False
        return len(result) == self._output

    def __call__(self, input_data):
        return functools.reduce(lambda prev, current: current(prev), [input_data] + self._layers)
