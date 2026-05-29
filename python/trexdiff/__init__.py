import ctypes
import pathlib

_lib = ctypes.CDLL(str(pathlib.Path(__file__).parent / "libtrexdiff.so"))


class _Node(ctypes.Structure):
    pass

class _NodeArray(ctypes.Structure):
    pass


# forward-declared above so mutual pointers resolve
_NodeArray._fields_ = [
    ("graph", ctypes.POINTER(ctypes.POINTER(_Node))),
    ("size",  ctypes.c_size_t),
]

_Node._fields_ = [
    ("val",         ctypes.c_double),
    ("grad",        ctypes.c_double),
    ("visited",     ctypes.c_bool),
    ("topo_graph",  ctypes.POINTER(_NodeArray)),
    ("inputs",      ctypes.POINTER(ctypes.POINTER(_Node))),
    ("input_count", ctypes.c_short),
    ("op_type",     ctypes.c_int),
]


_P = ctypes.POINTER(_Node)
_lib.init.restype,        _lib.init.argtypes        = _P, [ctypes.c_double]
_lib.combine.restype,     _lib.combine.argtypes     = _P, [_P, _P, ctypes.c_int]
_lib.transform.restype,   _lib.transform.argtypes   = _P, [_P, ctypes.c_int]
_lib.forward.restype,     _lib.forward.argtypes     = None, [_P]
_lib.backward.restype,    _lib.backward.argtypes    = None, [_P, ctypes.c_double]
_lib.zerograd.restype,    _lib.zerograd.argtypes    = None, [_P]
_lib.free_node.restype,   _lib.free_node.argtypes   = None, [_P]
_lib.finite_diff.restype, _lib.finite_diff.argtypes = ctypes.c_double, [_P, _P]


class Node:
    _OP_ADD, _OP_SUB, _OP_MUL, _OP_DIV, _OP_RELU, _OP_SIG  = 1, 2, 3, 4, 5, 6

    def __init__(self, val):
        self._p = _lib.init(float(val))
        self._inputs = ()  # keep input Nodes alive

    def __del__(self):
        _lib.free_node(self._p)

    def _combine(self, o, op):
        n = Node.__new__(Node)
        n._p = _lib.combine(self._p, o._p, op)
        n._inputs = (self, o)
        return n

    def __add__(self, o):
        return self._combine(o, Node._OP_ADD)
    
    def __sub__(self, o):
        return self._combine(o, Node._OP_SUB)

    def __mul__(self, o):
        return self._combine(o, Node._OP_MUL)

    def __truediv__(self, o):
        return self._combine(o, Node._OP_DIV)

    def forward(self):
        _lib.forward(self._p)

    def backward(self, g=1.0):
        _lib.backward(self._p, g)

    def zerograd(self):
        _lib.zerograd(self._p)

    @property
    def val(self):
        return self._p.contents.val

    @val.setter
    def val(self, value):
        self._p.contents.val = float(value)

    @property
    def grad(self):
        return self._p.contents.grad

    def __repr__(self):
        return f"Node(val={self.val:.6f}, grad={self.grad:.6f})"


def transform(node, op_code):
    n = Node.__new__(Node)
    n._p = _lib.transform(node._p, op_code)
    n._inputs = (node,)
    return n


def relu(node):
    return transform(node, Node._OP_RELU)


def sigmoid(node):
    return transform(node, Node._OP_SIG)


def finite_diff(inp, target):
    return _lib.finite_diff(inp._p, target._p)


__all__ = ["Node", "finite_diff", "relu", "sigmoid"]
