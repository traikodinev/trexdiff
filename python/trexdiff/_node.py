import ctypes
from ._lib import lib as _lib
from ._tensor2d import tensor2d, _Tensor2D

class _Node(ctypes.Structure):
    pass


class _NodeArray(ctypes.Structure):
    pass


_NodeArray._fields_ = [
    ("graph", ctypes.POINTER(ctypes.POINTER(_Node))),
    ("size",  ctypes.c_size_t),
]

_Node._fields_ = [
    ("val",         ctypes.POINTER(_Tensor2D)),
    ("grad",        ctypes.POINTER(_Tensor2D)),
    ("partial",     ctypes.POINTER(_Tensor2D)),
    ("visited",     ctypes.c_bool),
    ("topo_graph",  ctypes.POINTER(_NodeArray)),
    ("inputs",      ctypes.POINTER(ctypes.POINTER(_Node))),
    ("input_count", ctypes.c_short),
    ("op_type",     ctypes.c_int),
]

_P = ctypes.POINTER(_Node)
_P_T2D = ctypes.POINTER(_Tensor2D)

_lib.init.restype,        _lib.init.argtypes        = _P,                 [_P_T2D]
_lib.combine.restype,     _lib.combine.argtypes     = _P,                 [_P, _P, ctypes.c_int]
_lib.transform.restype,   _lib.transform.argtypes   = _P,                 [_P, ctypes.c_int]
_lib.transpose.restype,   _lib.transpose.argtypes   = _P,                 [_P]
_lib.forward.restype,     _lib.forward.argtypes     = None,               [_P]
_lib.backward.restype,    _lib.backward.argtypes    = ctypes.c_int,       [_P, _P_T2D]
_lib.zerograd.restype,    _lib.zerograd.argtypes    = None,               [_P]
_lib.free_node.restype,   _lib.free_node.argtypes   = None,               [_P]
_lib.free_node_shallow.restype,   _lib.free_node_shallow.argtypes = None, [_P]
_lib.finite_diff.restype, _lib.finite_diff.argtypes = _P_T2D,             [_P, _P]


class Node:
    _OP_ADD, _OP_SUB, _OP_MUL, _OP_DIV, _OP_RELU, _OP_SIG, _OP_LN, _OP_TRANSPOSE = 1, 2, 3, 4, 5, 6, 7, 8
    _OP_BROADCAST_ADD, _OP_BROADCAST_SUB = 9, 10

    def __init__(self, tensor=None, size=None):
        if size is not None and tensor is not None:
            raise ValueError("Cannot specify both tensor and size")
        if size is None and tensor is None:
            raise ValueError("Must specify either tensor or size")
        
        if tensor is None and size is not None:
            tensor = tensor2d.zeros(*size)

        self._p = _lib.init(tensor._p)
        tensor._owned = False  # C owns val now; free_node will free it
        self.val = tensor  # keep alive so _p.contents.val stays valid
        self.grad = tensor2d._from_ptr(self._p.contents.grad, owned=False)  # freed by free_node
        self._inputs = ()  # keep input Nodes alive

    def __del__(self):
        _lib.free_node(self._p)

    def _combine(self, other, op):
        n = Node.__new__(Node)
        n._p = _lib.combine(self._p, other._p, op)
        n._inputs = (self, other)
        n.val = tensor2d._from_ptr(n._p.contents.val, owned=False)  # freed by free_node
        n.grad = tensor2d._from_ptr(n._p.contents.grad, owned=False)  # freed by free_node
        return n

    def __add__(self, other):
        if self.val.can_broadcast(other.val):
            return self._combine(other, Node._OP_BROADCAST_ADD)
        return self._combine(other, Node._OP_ADD)
    
    def __sub__(self, other):
        if self.val.can_broadcast(other.val):
            return self._combine(other, Node._OP_BROADCAST_SUB)
        return self._combine(other, Node._OP_SUB)
    
    
    def __mul__(self, other): return self._combine(other, Node._OP_MUL)
    def __matmul__(self, other): return self._combine(other, Node._OP_MUL)  # alias for matmul
    def __truediv__(self, other): return self._combine(other, Node._OP_DIV)

    @property
    def T(self):
        n = Node.__new__(Node)
        n._p = _lib.transpose(self._p)
        n._inputs = (self,)
        n.val = tensor2d._from_ptr(n._p.contents.val, owned=False)  # freed by free_node
        n.grad = tensor2d._from_ptr(n._p.contents.grad, owned=False)  # freed by free_node
        return n

    def forward(self):
        _lib.forward(self._p)

    def backward(self, g=None):
        if g is None:
            g = tensor2d.ones(1, 1)
        elif not isinstance(g, tensor2d):
            g = tensor2d([[float(g)]])
        _lib.backward(self._p, g._p)

    def zerograd(self):
        _lib.zerograd(self._p)

    def __repr__(self):
        return f"Node(val={self.val}, grad={self.grad})"


def _transform(node, op_code):
    n = Node.__new__(Node)
    n._p = _lib.transform(node._p, op_code)
    n._inputs = (node,)
    n.val = tensor2d._from_ptr(n._p.contents.val, owned=False)  # freed by free_node
    n.grad = tensor2d._from_ptr(n._p.contents.grad, owned=False)  # freed by free_node

    # print(n.val)
    return n


def relu(node):       return _transform(node, Node._OP_RELU)
def sigmoid(node):    return _transform(node, Node._OP_SIG)
def log(node):        return _transform(node, Node._OP_LN)


def finite_diff(inp, target):
    return _lib.finite_diff(inp._p, target._p)
