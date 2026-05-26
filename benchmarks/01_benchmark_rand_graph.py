import time
import numpy as np
from trexdiff import Node


GRAPH_SEED = 11
N_LEAVES = 10      # leaf (input) nodes
N_OPS    = 50_000  # binary operations to chain on top

N_repeats = 500


def build_graph(rng, n_leaves, n_ops):
    """Build a random DAG: nodes in the pool are reused as inputs, so the
    result is a proper DAG (not just a binary tree) with shared sub-expressions.
    """
    leaves = [Node(rng.uniform(-1.0, 1.0)) for _ in range(n_leaves)]
    pool   = list(leaves)

    for _ in range(n_ops):
        i   = int(rng.integers(0, len(pool)))
        j   = int(rng.integers(0, len(pool)))
        op  = int(rng.integers(0, 3))          # 0=add, 1=sub, 2=mul
        if op == 0:
            node = pool[i] + pool[j]
        elif op == 1:
            node = pool[i] - pool[j]
        else:
            node = pool[i] * pool[j]
        pool.append(node)

    return pool[-1], leaves


rng            = np.random.default_rng(GRAPH_SEED)
output, leaves = build_graph(rng, N_LEAVES, N_OPS)

times = []
for _ in range(N_repeats):
    start = time.perf_counter()
    output.zerograd()
    output.forward()
    output.backward()
    times.append(time.perf_counter() - start)

times = np.array(times)
print(f"Large random DAG benchmark  (seed={GRAPH_SEED}, N_LEAVES={N_LEAVES}, N_OPS={N_OPS}, {N_repeats} repeats)")
print(f"{times.mean() * 1e3:.4f} ms +- {times.std() * 1e3:.4f} ms")
