import trexdiff
import numpy as np

def test_matmul():
    """Test the matmul operator of tensor2d against NumPy's matmul."""
    N = 100
    for i in range(N):
        M, N, K = np.random.randint(1, 100, size=3)
        t1 = trexdiff.tensor2d(arr1 := np.random.rand(M, N))
        t2 = trexdiff.tensor2d(arr2 := np.random.rand(N, K))
        
        assert np.allclose((t1 @ t2).tolist(), arr1 @ arr2)
    else:
        print(f"All {N} tests passed!")
