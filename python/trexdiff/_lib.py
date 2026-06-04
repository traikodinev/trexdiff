import ctypes
import pathlib

lib = ctypes.CDLL(str(pathlib.Path(__file__).parent / "libtrexdiff.so"))
