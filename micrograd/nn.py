import random
from micrograd.engine import Value

import copy

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class ConjoinLayer(Layer):
    def __init__(self, l1: Layer, l2: Layer, nout: int, **kwargs):
        self.neurons = [Neuron(len(l1.neurons) + len(l2.neurons), **kwargs) for _ in range(nout)]

class RegisterConjoinLayer(Layer):
    def __init__(self, nin1, nin2, nout, **kwargs):
        self.neurons = [Neuron(nin1 + nin2, **kwargs) for _ in range(nout)]

class MLP(Module):

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

class ArchitecturalModel(Layer):
    """
    The architecture is defined as a list of calls. 
    Each call has an input register (original data is contained in register ``I``)
    and an output register. Registers must be pre-defined. The content of register ``O``
    is returned.

    The architecture is a list::

        [(register_in, nneurons, register_out, kwargs), ...]

    Layers are called in the order they are in the architecture.
    Example for a split layer::

        [("I", 4, "r0", kwargs), ("I", 4, "r1", kwargs), (["r0", "r1"], 4, "O", kwargs)]

    The input registers mus be either a single string or a list of two strings.

    You are responsible for making sure everything works out.
    """
    def __init__(self, nin, registers, architecture):
        self.registers = {"I": None, "O": None}
        self.registers.update({r: None for r in registers})
        
        register_sizes = {"I": nin}
        for rin, nn, rout, kwa in architecture:
            if(rout not in register_sizes):
               register_sizes[rout] = nn 
            else:
                if(register_sizes[rout] != nn):
                    raise ValueError(f"register '{rout}' input mismatch (got {nn} expected {register_sizes[rout]})" )

        self.layers = []
        for rin, nn, rout, kwa in architecture:
            if(isinstance(rin, list)):
                if(len(rin) != 2):
                    raise ValueError("The input registers mus be either a single string or a list of two strings.")
                self.layers.append(RegisterConjoinLayer(register_sizes[rin[0]], register_sizes[rin[1]], register_sizes[rout], **kwa))
            self.layers.append(rin, Layer(register_sizes[rin], register_sizes[rout], **kwa), rout)

    def __call__(self, x):
        registers = copy.copy(self.registers)
        registers["I"] = x
        for rin, layer, rout in self.layers:
            if(isinstance(rin, list)):
                input = registers[rin[0]] + registers[rin[1]]
            else:
                input = registers[rin]
            registers[rout] = layer(input)
        return registers["O"]

    def parameters(self):
        return [p for _, layer, _ in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"ArchitecturalModel of [{', '.join(str(layer) for layer in self.layers)}]"


