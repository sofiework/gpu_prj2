# gpu_prj2 — run: `n`

Published 2026-09-15T20:56:25-04:00 from PACE-ICE. `grade.py --perf-only` (no ncu).
Ran `grade.py --perf-only` plus explicit correctness checks at every graded size.

> **Check the GPU model in the Environment section below.** Only H100 numbers are
> gradeable; anything else is for relative comparison only.

### Environment / commit

```
variant : n
date    : 2026-09-15T20:55:18-04:00
job     : 5787166 on atl1-1-03-012-23-0.pace.gatech.edu
commit  : fa4f389 bitonic_sort: switch to NAIVE path for profiling comparison
md5     : 2975cc52e9495e7023826e49f4730bd0
NVIDIA H100 80GB HBM3, 81559 MiB, 595.71.05
Cuda compilation tools, release 12.6, V12.6.68
Build cuda_12.6.r12.6/compiler.34714021_0
```

### Correctness (2K / 10K / 100K / 1M / 10M)

```
size 2000 -> FUNCTIONAL SUCCESS
size 10000 -> FUNCTIONAL SUCCESS
size 100000 -> FUNCTIONAL SUCCESS
size 1000000 -> FUNCTIONAL SUCCESS
size 10000000 -> FUNCTIONAL SUCCESS
```

### grade.py --perf-only

```
Compiled! bitonic.cu
FUNCTIONAL SUCCESS
Array size         : 100000000
CPU Sort Time (ms) : 16026.594727
GPU Sort Time (ms) : 295.607056
GPU Sort Speed     : 338.286926 million elements per second
PERF PASSING
GPU Sort is  54x faster than CPU !!!
H2D Transfer Time (ms): 42.838879
Kernel Time (ms)      : 123.860672
D2H Transfer Time (ms): 128.907516

FUNCTIONAL SUCCESS
Array size         : 100000000
CPU Sort Time (ms) : 16105.275391
GPU Sort Time (ms) : 302.085266
GPU Sort Speed     : 331.032379 million elements per second
PERF PASSING
GPU Sort is  53x faster than CPU !!!
H2D Transfer Time (ms): 40.535358
Kernel Time (ms)      : 123.850304
D2H Transfer Time (ms): 137.699615

FUNCTIONAL SUCCESS
Array size         : 100000000
CPU Sort Time (ms) : 15939.096680
GPU Sort Time (ms) : 291.157104
GPU Sort Speed     : 343.457184 million elements per second
PERF PASSING
GPU Sort is  54x faster than CPU !!!
H2D Transfer Time (ms): 40.697536
Kernel Time (ms)      : 123.867554
D2H Transfer Time (ms): 126.592003

Kernel Time: 123.850304ms, Score: 6.459
Memory Transfer Time: 167.289539ms, Score: 0.717
Million elements per second: 343.478
Total Score: 12.18 pts
```

### Build

```
nvcc rc=0
```

