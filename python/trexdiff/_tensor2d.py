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
        _lib.tensor2d_add_inplace(self._p, other._p, 1.0)
        return self
    
    def __isub__(self, other):
        _lib.tensor2d_add_inplace(self._p, other._p, -1.0)
        return self

    def __add__(self, other):
        ptr = _lib.tensor2d_add(self._p, other._p)
        if not ptr:
            raise ValueError(f"add: incompatible shapes {self.shape} and {other.shape}")
        return tensor2d._from_ptr(ptr)

    def __sub__(self, other):
        ptr = _lib.tensor2d_sub(self._p, other._p)
        if not ptr:
            raise ValueError(f"sub: incompatible shapes {self.shape} and {other.shape}")
        return tensor2d._from_ptr(ptr)

    def __mul__(self, scalar):
        return tensor2d._from_ptr(_lib.tensor2d_scalar_mul(self._p, float(scalar)))

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
        
