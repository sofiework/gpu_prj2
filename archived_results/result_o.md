# gpu_prj2 — run: `o`

Published 2026-09-15T20:57:14-04:00 from PACE-ICE. `grade.py --perf-only` (no ncu).
Ran `grade.py --perf-only` plus explicit correctness checks at every graded size.

> **Check the GPU model in the Environment section below.** Only H100 numbers are
> gradeable; anything else is for relative comparison only.

### Environment / commit

```
variant : o
date    : 2026-09-15T20:55:19-04:00
job     : 5787167 on atl1-1-03-013-3-0.pace.gatech.edu
commit  : b868d42 edit
md5     : 2499fada92fc8bde6ebac667d04cbdb4
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
CPU Sort Time (ms) : 18443.876953
GPU Sort Time (ms) : 248.365891
GPU Sort Speed     : 402.631775 million elements per second
PERF PASSING
GPU Sort is  74x faster than CPU !!!
H2D Transfer Time (ms): 43.085857
Kernel Time (ms)      : 73.144226
D2H Transfer Time (ms): 132.135803

FUNCTIONAL SUCCESS
Array size         : 100000000
CPU Sort Time (ms) : 18411.544922
GPU Sort Time (ms) : 245.126083
GPU Sort Speed     : 407.953308 million elements per second
PERF PASSING
GPU Sort is  75x faster than CPU !!!
H2D Transfer Time (ms): 40.610271
Kernel Time (ms)      : 73.163712
D2H Transfer Time (ms): 131.352097

FUNCTIONAL SUCCESS
Array size         : 100000000
CPU Sort Time (ms) : 18281.636719
GPU Sort Time (ms) : 244.960037
GPU Sort Speed     : 408.229858 million elements per second
PERF PASSING
GPU Sort is  74x faster than CPU !!!
H2D Transfer Time (ms): 40.933186
Kernel Time (ms)      : 73.155357
D2H Transfer Time (ms): 130.871490

Kernel Time: 73.144226ms, Score: 10
Memory Transfer Time: 171.804676ms, Score: 0.698
Million elements per second: 408.248
Total Score: 15.7 pts
```

### Build

```
nvcc rc=0
```

