# gpu_prj2 — run: `v100_naive`

Published 2026-09-15T08:09:42-04:00 from PACE-ICE. `grade.py --perf-only` (no ncu).
Ran `grade.py --perf-only` plus explicit correctness checks at every graded size.

> **Check the GPU model in the Environment section below.** Only H100 numbers are
> gradeable; anything else is for relative comparison only.

### Environment / commit

```
variant : v100_naive
date    : 2026-09-15T08:07:52-04:00
job     : 5804895 on atl1-1-02-010-31-0.pace.gatech.edu
commit  : fa4f389 bitonic_sort: switch to NAIVE path for profiling comparison
md5     : 2975cc52e9495e7023826e49f4730bd0
Tesla V100-PCIE-16GB, 16384 MiB, 575.57.08
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
CPU Sort Time (ms) : 28562.091797
GPU Sort Time (ms) : 684.683228
GPU Sort Speed     : 146.052948 million elements per second
PERF PASSING
GPU Sort is  41x faster than CPU !!!
H2D Transfer Time (ms): 91.416321
Kernel Time (ms)      : 390.615845
D2H Transfer Time (ms): 202.651077

FUNCTIONAL SUCCESS
Array size         : 100000000
CPU Sort Time (ms) : 28630.214844
GPU Sort Time (ms) : 683.285522
GPU Sort Speed     : 146.351700 million elements per second
PERF PASSING
GPU Sort is  41x faster than CPU !!!
H2D Transfer Time (ms): 89.991615
Kernel Time (ms)      : 390.735870
D2H Transfer Time (ms): 202.558044

FUNCTIONAL SUCCESS
Array size         : 100000000
CPU Sort Time (ms) : 28440.152344
GPU Sort Time (ms) : 682.545715
GPU Sort Speed     : 146.510330 million elements per second
PERF PASSING
GPU Sort is  41x faster than CPU !!!
H2D Transfer Time (ms): 89.351997
Kernel Time (ms)      : 390.616760
D2H Transfer Time (ms): 202.576965

Kernel Time: 390.615845ms, Score: 2.048
Memory Transfer Time: 291.928962ms, Score: 0.411
Million elements per second: 146.511
Total Score: 7.46 pts
```

### Build

```
nvcc rc=0
```

