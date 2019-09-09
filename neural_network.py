import functools
import numpy


def linear_handler(parameters, input_data):
    return numpy.array(parameters['matrix']).dot(input_data) + parameters['bias']


def function_handler(parameters, input_data):
    return list(map(parameters['func'], input_data))


_handlers = {'linear': linear_handler, 'function': function_handler}


def require_parameters(actual_parameters, *required_parameters):
    for param in required_parameters:
        assert param in actual_parameters, '"{}" is required'.format(param)


class layer:
    def __init__(self, *, handler, parameters):
        self._parameters = parameters
        self._handler = handler

    def __call__(self, input_data):
        return self._handler(self._parameters, input_data)


def make_feedforward_neural_network(*, input_count, output_count):
    return neural_network(input_count, output_count)


class neural_network:
    def __init__(self, input_count, output_count):
        self._shape = self._input, self._output = input_count, output_count
        self._layers = []

    def add_layer(self, layer_type, **layer_data):
        assert layer_type in _handlers, '"{}" is missing in handlers dictionary'.format(layer_type)
        if layer_type == 'function':
            require_parameters(layer_data, 'func')
        elif layer_type == 'linear':
            require_parameters(layer_data, 'matrix', 'bias')
        self._layers.append(layer(handler=_handlers[layer_type], parameters=layer_data))

    def add_function_layer(self, **layer_data):
        require_parameters(layer_data, 'func')
        self._layers.append(layer(handler=_handlers['function'], parameters=layer_data))

    def add_linear_layer(self, **layer_data):
        require_parameters(layer_data, 'matrix', 'bias')
        self._layers.append(layer(handler=_handlers['linear'], parameters=layer_data))

    def check_dimensions(self):
        try:
            result = self([0] * self._input)
        except:
            return False
        return len(result) == self._output

    def __call__(self, input_data):
        return functools.reduce(lambda prev, current: current(prev), [input_data] + self._layers)
