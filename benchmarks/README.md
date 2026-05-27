# Benchmarks log

_NOTE: a raw log of benchmarks_

## `01_benchmark.py`

### `34160e6` - Before topological sorting optimization

```
Large random DAG benchmark  (seed=11, N_LEAVES=10, N_OPS=50000, 500 repeats)
0.0462 ms +- 0.0041 ms
```

### `24d4b28` - after topological sorting optimization

```
Large random DAG benchmark  (seed=11, N_LEAVES=10, N_OPS=50000, 500 repeats)
0.0240 ms +- 0.0707 ms
```

### `24a41af` - after adding gradient carryover with `partial`

(shouldn't increase, this remains `O(N)` but adds a zero-out pass, so `2 * O(N) = O(N)`)

```
Large random DAG benchmark  (seed=11, N_LEAVES=10, N_OPS=50000, 500 repeats)
0.0210 ms +- 0.0633 ms
```
