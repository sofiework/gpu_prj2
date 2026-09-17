# gpu_prj2 — run: `v100_opt`

Published 2026-09-15T08:10:10-04:00 from PACE-ICE. `grade.py --perf-only` (no ncu).
Ran `grade.py --perf-only` plus explicit correctness checks at every graded size.

> **Check the GPU model in the Environment section below.** Only H100 numbers are
> gradeable; anything else is for relative comparison only.

### Environment / commit

```
variant : v100_opt
date    : 2026-09-15T08:08:16-04:00
job     : 5804896 on atl1-1-02-010-32-0.pace.gatech.edu
commit  : b868d42 edit
md5     : 2499fada92fc8bde6ebac667d04cbdb4
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
CPU Sort Time (ms) : 28247.771484
GPU Sort Time (ms) : 495.807861
GPU Sort Speed     : 201.691040 million elements per second
PERF PASSING
GPU Sort is  56x faster than CPU !!!
H2D Transfer Time (ms): 88.228928
Kernel Time (ms)      : 203.278214
D2H Transfer Time (ms): 204.300705

FUNCTIONAL SUCCESS
Array size         : 100000000
CPU Sort Time (ms) : 28474.451172
GPU Sort Time (ms) : 487.913391
GPU Sort Speed     : 204.954407 million elements per second
PERF PASSING
GPU Sort is  58x faster than CPU !!!
H2D Transfer Time (ms): 88.667870
Kernel Time (ms)      : 195.537766
D2H Transfer Time (ms): 203.707748

FUNCTIONAL SUCCESS
Array size         : 100000000
CPU Sort Time (ms) : 28369.636719
GPU Sort Time (ms) : 495.557373
GPU Sort Speed     : 201.792984 million elements per second
PERF PASSING
GPU Sort is  57x faster than CPU !!!
H2D Transfer Time (ms): 87.540062
Kernel Time (ms)      : 204.392197
D2H Transfer Time (ms): 203.625122

Kernel Time: 195.537766ms, Score: 4.091
Memory Transfer Time: 291.165184ms, Score: 0.412
Million elements per second: 205.464
Total Score: 9.5 pts
```

### Build

```
nvcc rc=0
```

