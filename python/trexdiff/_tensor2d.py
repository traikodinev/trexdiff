import ctypes
from ._lib import lib as _lib


class _Tensor2D(ctypes.Structure):
    _fields_ = [
        ("matrix", ctypes.POINTER(ctypes.c_double)),
        ("M",      ctypes.c_size_t),
        ("N",      ctypes.c_size_t),
        ("T",      ctypes.c_bool),
    ]


_TP = ctypes.POINTER(_Tensor2D)

_lib.tensor2d_from_array.restype,  _lib.tensor2d_from_array.argtypes  = _TP,  [ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(ctypes.c_double)]
_lib.tensor2d_zeros.restype,       _lib.tensor2d_zeros.argtypes        = _TP,  [ctypes.c_size_t, ctypes.c_size_t]
_lib.tensor2d_ones.restype,        _lib.tensor2d_ones.argtypes         = _TP,  [ctypes.c_size_t, ctypes.c_size_t]
_lib.tensor2d_free.restype,        _lib.tensor2d_free.argtypes         = None, [_TP]
_lib.tensor2d_matmul.restype,      _lib.tensor2d_matmul.argtypes       = _TP,  [_TP, _TP]
_lib.tensor2d_transpose.restype,   _lib.tensor2d_transpose.argtypes    = _TP,  [_TP]
_lib.tensor2d_add.restype,         _lib.tensor2d_add.argtypes          = _TP,  [_TP, _TP]
_lib.tensor2d_sub.restype,         _lib.tensor2d_sub.argtypes          = _TP,  [_TP, _TP]
_lib.tensor2d_scalar_mul.restype,  _lib.tensor2d_scalar_mul.argtypes   = _TP,  [_TP, ctypes.c_double]

_lib.tensor2d_add_inplace.restype, _lib.tensor2d_add_inplace.argtypes = None, [_TP, _TP, ctypes.c_double]

# tensor-scalar operations and inplace operations
_lib.tensor2d_scalar_add.restype,  _lib.tensor2d_scalar_add.argtypes   = _TP,  [_TP, ctypes.c_double]
_lib.tensor2d_scalar_mul.restype,  _lib.tensor2d_scalar_mul.argtypes   = _TP,  [_TP, ctypes.c_double]
_lib.tensor2d_sqrt.restype,        _lib.tensor2d_sqrt.argtypes         = _TP,  [_TP]
_lib.tensor2d_pow.restype,         _lib.tensor2d_pow.argtypes          = _TP,  [_TP, ctypes.c_double]
_lib.tensor2d_elwise_mul.restype, _lib.tensor2d_elwise_mul.argtypes = _TP, [_TP, _TP]
_lib.tensor2d_elwise_div.restype, _lib.tensor2d_elwise_div.argtypes = _TP, [_TP, _TP]

_lib.tensor2d_relu.restype,        _lib.tensor2d_relu.argtypes         = _TP,  [_TP]
_lib.tensor2d_sigmoid.restype,     _lib.tensor2d_sigmoid.argtypes      = _TP,  [_TP]

_lib.tensor2d_add_broadcast.restype, _lib.tensor2d_add_broadcast.argtypes = _TP, [_TP, _TP, ctypes.c_double]

_lib.tensor2d_cos.restype,         _lib.tensor2d_cos.argtypes          = _TP,  [_TP]
_lib.tensor2d_sin.restype,         _lib.tensor2d_sin.argtypes          = _TP,  [_TP]

class tensor2d:
    """A 2-D matrix backed by the C tensor2d library."""

    TRUNCATE_THRESHOLD = 10  # max rows/cols to print in __repr__

    def __init__(self, data):
        """Create from a 2-D nested list (or any sequence of sequences)."""
        rows = [list(row) for row in data]
        M, N = len(rows), len(rows[0])

        for row in rows:
            if len(row) != N:
                raise ValueError("All rows must have the same length")

        flat = [float(x) for row in rows for x in row]
        arr  = (ctypes.c_double * len(flat))(*flat)
        self._p = _lib.tensor2d_from_array(M, N, arr)
        self._owned = True

    @classmethod
    def zeros(cls, M, N):
        obj = cls.__new__(cls)
        obj._p = _lib.tensor2d_zeros(M, N)
        obj._owned = True
        return obj

    @classmethod
    def ones(cls, M, N):
        obj = cls.__new__(cls)
        obj._p = _lib.tensor2d_ones(M, N)
        obj._owned = True
        return obj

    @classmethod
    def _from_ptr(cls, ptr, owned=True):
        """Wrap a raw C pointer returned by a library call."""
        obj = cls.__new__(cls)
        obj._p = ptr
        obj._owned = owned
        return obj

    def __del__(self):
        if getattr(self, "_owned", True) and getattr(self, "_p", None):
            _lib.tensor2d_free(self._p)

    # shape is just (M,N)
    @property
    def shape(self):
        c = self._p.contents
        return (c.M, c.N)

    @property
    def T(self):
        return tensor2d._from_ptr(_lib.tensor2d_transpose(self._p))

    # operators
    def __matmul__(self, other):
        ptr = _lib.tensor2d_matmul(self._p, other._p)
        if not ptr:
            raise ValueError(f"matmul: incompatible shapes {self.shape} and {other.shape}")
        return tensor2d._from_ptr(ptr)

    def __iadd__(self, other):
        if isinstance(other, (int, float)):
            raise NotImplementedError("inplace tensor2d addition only supports other tensor2d objects")
        
        _lib.tensor2d_add_inplace(self._p, other._p, 1.0)
        return self
    
    def __isub__(self, other):
        if isinstance(other, (int, float)):
            raise NotImplementedError("inplace tensor2d subtraction only supports other tensor2d objects")

        _lib.tensor2d_add_inplace(self._p, other._p, -1.0)
        return self

    def can_broadcast(self, other):
        """Can you broadcast other to self?"""
        return self.shape[1] == other.shape[1] and other.shape[0] == 1

    def __add__(self, other):
        if isinstance(other, (int, float)):
            ptr = _lib.tensor2d_scalar_add(self._p, float(other))
            return tensor2d._from_ptr(ptr)

        if self.can_broadcast(other):
            ptr = _lib.tensor2d_add_broadcast(self._p, other._p, 1.0)
        else:
            ptr = _lib.tensor2d_add(self._p, other._p)
    
        if not ptr:
            raise ValueError(f"add: incompatible shapes {self.shape} and {other.shape}")
        return tensor2d._from_ptr(ptr)
    
    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self):
        ptr = _lib.tensor2d_scalar_mul(self._p, -1.0)
        return tensor2d._from_ptr(ptr)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            ptr = _lib.tensor2d_scalar_add(self._p, -float(other))
            return tensor2d._from_ptr(ptr)

        if self.can_broadcast(other):
            ptr = _lib.tensor2d_add_broadcast(self._p, other._p, 1.0)
        else:
            ptr = _lib.tensor2d_add(self._p, other._p)
            
        if not ptr:
            raise ValueError(f"sub: incompatible shapes {self.shape} and {other.shape}")
        return tensor2d._from_ptr(ptr)


    def __rsub__(self, other):
        return (-self).__add__(other)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            ptr = _lib.tensor2d_scalar_mul(self._p, float(other))
            return tensor2d._from_ptr(ptr)

        ptr = _lib.tensor2d_elwise_mul(self._p, other._p)
        if not ptr:
            raise ValueError(f"elementwise mul: incompatible shapes {self.shape} and {other.shape}")
        return tensor2d._from_ptr(ptr)


    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            ptr = _lib.tensor2d_scalar_mul(self._p, 1.0 / float(other))
            return tensor2d._from_ptr(ptr)
    
        ptr = _lib.tensor2d_elwise_div(self._p, other._p)
        if not ptr:
            raise ValueError(f"elementwise div: incompatible shapes {self.shape} and {other.shape}")
        return tensor2d._from_ptr(ptr)
    
    
    def __itruediv__(self, other):
        if isinstance(other, (int, float)):
            ptr = _lib.tensor2d_scalar_mul(self._p, 1.0 / float(other))
            return tensor2d._from_ptr(ptr)

        raise NotImplementedError("inplace tensor2d division only supports scalar divisors")
    

    def __pow__(self, other):
        if isinstance(other, (int, float)):
            # sqrt special case
            if other == 0.5:
                ptr = _lib.tensor2d_sqrt(self._p)
                return tensor2d._from_ptr(ptr)
            
            ptr = _lib.tensor2d_pow(self._p, float(other))
            return tensor2d._from_ptr(ptr)

        raise NotImplementedError("tensor2d only supports scalar exponents")
    

    def __rmul__(self, scalar):
        return self.__mul__(scalar)
    
    def __getitem__(self, key):
        M, N = self.shape
        if isinstance(key, tuple) and len(key) == 2:
            i, j = key
            if 0 <= i < M and 0 <= j < N:
                return self._p.contents.matrix[i * N + j]
            else:
                raise IndexError("tensor2d index out of range")
        else:
            raise TypeError("tensor2d indices must be a tuple of two integers")

    # tolist
    def tolist(self):
        """Return contents as a list of lists of floats."""
        M, N = self.shape
        m = self._p.contents.matrix
        return [[m[i * N + j] for j in range(N)] for i in range(M)]

    def __repr__(self):
        M, N = self.shape
        
        # print contents, row per line
        if M <= self.TRUNCATE_THRESHOLD and N <= self.TRUNCATE_THRESHOLD:
            rows = []
            for i in range(M):
                row = [f"{self._p.contents.matrix[i * N + j]:.2f}" for j in range(N)]
                rows.append("[" + ", ".join(row) + "]")
            return "tensor2d([\n  " + ",\n  ".join(rows) + "\n])"
        
        # print ... and first N rows/cols
        rows = []
        for i in range(min(M, self.TRUNCATE_THRESHOLD)):
            row = [f"{self._p.contents.matrix[i * N + j]:.2f}" for j in range(min(N, self.TRUNCATE_THRESHOLD))]
            if N > self.TRUNCATE_THRESHOLD:
                row.append("...")
            rows.append("[" + ", ".join(row) + "]")
        if M > self.TRUNCATE_THRESHOLD:
            rows.append("...")

        return "tensor2d([\n  " + ",\n  ".join(rows) + "\n])"
        
    
    def relu(self):
        ptr = _lib.tensor2d_relu(self._p)
        return tensor2d._from_ptr(ptr)

    def sigmoid(self):
        ptr = _lib.tensor2d_sigmoid(self._p)
        return tensor2d._from_ptr(ptr)


    def sin(self):
        ptr = _lib.tensor2d_sin(self._p)
        return tensor2d._from_ptr(ptr)
    
    def cos(self):
        ptr = _lib.tensor2d_cos(self._p)
        return tensor2d._from_ptr(ptr)
