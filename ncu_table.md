# Nsight Compute profile — shared-memory + pinned-overlap + short, TILE = 16384, 10M elements

Source: `metrics_latest.csv` (67 launches: 1 kernel A, 55 global merge, 10 kernel B, 1 fill_padding).

A. Overall kernel cost & bottleneck classification

| Kernel Name | Number of Launches | Total Duration (ms) | Percentage of Total Kernel Time | Compute (SM) Throughput (% of SOL) | Memory Throughput (% of SOL) | Achieved Occupancy (%) | Theoretical Occupancy (%) |
|---|---|---|---|---|---|---|---|
| bitonic_merge_small_k | 1 | 2.137 | 29.8% | 75.09% | 25.07% | 93.95% | 100.0% |
| bitonic_merge | 55 | 2.008 | 28.0% | 60.31% | 30.05% | 80.15% | 100.0% |
| bitonic_merge_large_k | 10 | 2.983 | 41.6% | 74.50% | 26.96% | 93.79% | 100.0% |
| fill_padding | 1 | 0.034 | 0.5% | 23.65% | 9.58% | 9.64% | 100.0% |
| **Total** | **67** | **7.163** | **100.0%** | — | — | — | — |

B. Warp scheduling & stall reasons

| Kernel Name | Active Warps per Scheduler | Eligible Warps per Scheduler | No Eligible Cycles (%) | Warp Cycles per Issued Instruction | Stall: Barrier (%) | Stall: Short Scoreboard (%) | Stall: Long Scoreboard (%) | Stall: Other (%) |
|---|---|---|---|---|---|---|---|---|
| bitonic_merge_small_k | 15.03 | 6.21 | 22.4% | 19.4 | 5.8% | 1.0% | 0.1% | 93.1% |
| bitonic_merge | 12.66 | 2.16 | 35.2% | 19.5 | 0.0% | 12.4% | 31.3% | 56.4% |
| bitonic_merge_large_k | 15.01 | 6.36 | 22.5% | 19.4 | 5.8% | 0.9% | 0.6% | 92.8% |
| fill_padding | 1.36 | 0.15 | 86.8% | 10.3 | 0.0% | 31.4% | 0.0% | 68.6% |

C. Memory access pattern

| Kernel Name | Global Load Sectors (M) | Global Store Sectors (M) | Sectors per Request (load) | DRAM Read (GiB) | DRAM Written (GiB) | L1 Hit Rate (%) | L2 Hit Rate (%) | Shared Load Wavefronts (M) | Shared Store Wavefronts (M) | Shared Bank Conflicts (M) | Bank Conflicts per Wavefront |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bitonic_merge_small_k | 1.0 | 1.0 | 2.00 | 0.03 | 0.01 | 46.7% | 54.2% | 55.6 | 41.3 | 0.034 | 0.0004 |
| bitonic_merge | 57.7 | 27.7 | 2.00 | 1.80 | 0.17 | 27.1% | 33.9% | 0.0 | 0.0 | 0.000 | — |
| bitonic_merge_large_k | 10.5 | 10.5 | 2.00 | 0.33 | 0.13 | 46.6% | 51.2% | 78.8 | 47.5 | 0.300 | 0.0024 |
| fill_padding | 0.0 | 0.4 | 0.00 | 0.00 | 0.00 | 47.9% | 100.2% | 0.0 | 0.0 | 0.000 | — |
| **Total** | **69.2** | **39.7** | **2.00** | **2.16** | **0.31** | **30.3%** | **36.8%** | **134.4** | **88.8** | **0.335** | **0.0015** |

B2. Full stall breakdown (percentage of warp cycles per issued instruction)

| Kernel Name | not selected | math pipe throttle | long scoreboard | wait | short scoreboard | dispatch stall | barrier | mio throttle |
|---|---|---|---|---|---|---|---|---|
| bitonic_merge_small_k | 36.2% | 30.0% | 0.1% | 11.3% | 1.0% | 7.7% | 5.8% | 2.4% |
| bitonic_merge | 12.0% | 6.4% | 31.3% | 20.1% | 12.4% | 3.5% | 0.0% | 6.5% |
| bitonic_merge_large_k | 37.3% | 29.6% | 0.6% | 8.6% | 0.9% | 7.8% | 5.8% | 3.8% |
| fill_padding | 1.2% | 0.2% | 0.0% | 24.0% | 31.4% | 0.3% | 0.0% | 3.7% |


Analysis
I was expecting the stall to be dominated by barrier / long scoreboard / short scoreboard, but instead bitonic_merge_small_k and bitonic_merge_large_k are bottlenecked at math pipe throttle and queueing. 

So (optimize stride and dir calculation)


Analysis - vectorized memory access global memory [3]

128B cache line and 32B sector, 
with 8B/thread, 256B/warp,  its 8 sectors/2 cache line
with 16B/thread max, 512B/warp, 16 sectors/4 cache line


[3] vectorized memory access https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/
[4] static indexing small array on register https://developer.nvidia.com/blog/fast-dynamic-indexing-private-arrays-cuda/
[5] cuda runtime API https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html 

However vectorized 

┌────────────────────────┬──────────┬─────────┐
│                        │  non-vectorzied  │ vectorized │
├────────────────────────┼──────────┼─────────┤
│ regs/thread            │ 32       │ 40      │ 
├────────────────────────┼──────────┼─────────┤
│ blocks/SM by registers │ 2        │ 1       │
├────────────────────────┼──────────┼─────────┤
│ shared carveout        │ 102.4 KB │ 65.5 KB │
├────────────────────────┼──────────┼─────────┤
│ blocks/SM by shared    │ 3        │ 1       │
├────────────────────────┼──────────┼─────────┤
│ theoretical occupancy  │ 100%     │ 50%     │
└────────────────────────┴──────────┴─────────┘

At 1024 threads/block, 32 regs × 1024 = 32,768, so two blocks exactly fill the 65,536-register file; 40 regs needs 40,960, and two blocks would want 81,920, so only one fits.
## Latest profile — vectorized int4 global merge + `__launch_bounds__`-era build (`metrics_latest2.csv`, 10M elements, 67 launches)

---

A. Overall kernel cost & bottleneck classification

| Kernel Name | Number of Launches | Total Duration (ms) | Percentage of Total Kernel Time | Compute (SM) Throughput (% of SOL) | Memory Throughput (% of SOL) | Achieved Occupancy (%) | Theoretical Occupancy (%) |
|---|---|---|---|---|---|---|---|
| bitonic_merge_small_k | 1 | 1.621 | 40.7% | 78.20% | 33.06% | 93.78% | 100.0% |
| bitonic_merge_vec_int4 | 55 | 1.123 | 28.2% | 31.75% | 66.35% | 73.31% | 100.0% |
| bitonic_merge_large_k | 10 | 1.205 | 30.3% | 78.91% | 68.81% | 93.64% | 100.0% |
| fill_padding | 1 | 0.034 | 0.9% | 23.70% | 9.57% | 9.50% | 100.0% |
| **Total** | **67** | **3.984** | **100.0%** | — | — | — | — |

B. Warp scheduling & stall reasons

| Kernel Name | Active Warps per Scheduler | Eligible Warps per Scheduler | No Eligible Cycles (%) | Warp Cycles per Issued Instruction | Stall: Barrier (%) | Stall: Short Scoreboard (%) | Stall: Long Scoreboard (%) | Stall: Other (%) |
|---|---|---|---|---|---|---|---|---|
| bitonic_merge_small_k | 15.00 | 6.91 | 19.2% | 18.6 | 5.6% | 0.9% | 0.1% | 93.5% |
| bitonic_merge_vec_int4 | 11.93 | 0.94 | 67.0% | 36.2 | 0.0% | 4.4% | 67.0% | 28.6% |
| bitonic_merge_large_k | 14.97 | 6.29 | 17.0% | 18.0 | 8.0% | 8.7% | 1.4% | 81.9% |
| fill_padding | 1.37 | 0.15 | 86.5% | 10.1 | 0.0% | 31.8% | 0.0% | 68.2% |

C. Memory access pattern

| Kernel Name | Global Load Sectors (M) | Global Store Sectors (M) | Sectors per Request (load) | DRAM Read (GiB) | DRAM Written (GiB) | L1 Hit Rate (%) | L2 Hit Rate (%) | Shared Load Wavefronts (M) | Shared Store Wavefronts (M) | Shared Bank Conflicts (M) | Bank Conflicts per Wavefront |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bitonic_merge_small_k | 1.0 | 1.0 | 2.00 | 0.03 | 0.01 | 47.3% | 51.3% | 55.6 | 41.3 | 0.037 | 0.0004 |
| bitonic_merge_vec_int4 | 57.7 | 57.4 | 16.00 | 1.80 | 0.63 | 44.1% | 51.2% | 0.0 | 0.0 | 0.000 | — |
| bitonic_merge_large_k | 10.5 | 10.5 | 2.00 | 0.33 | 0.13 | 46.7% | 51.2% | 79.1 | 47.5 | 0.557 | 0.0044 |
| fill_padding | 0.0 | 0.4 | 0.00 | 0.00 | 0.00 | 47.9% | 100.1% | 0.0 | 0.0 | 0.000 | — |
| **Total** | **69.2** | **69.4** | **7.38** | **2.16** | **0.77** | **44.5%** | **51.2%** | **134.7** | **88.8** | **0.594** | **0.0027** |

B2. Full stall breakdown (percentage of warp cycles per issued instruction)

| Kernel Name | not selected | math pipe throttle | long scoreboard | wait | short scoreboard | dispatch stall | barrier | mio throttle |
|---|---|---|---|---|---|---|---|---|
| bitonic_merge_small_k | 40.7% | 28.4% | 0.1% | 11.1% | 0.9% | 6.7% | 5.6% | 0.8% |
| bitonic_merge_vec_int4 | 5.1% | 4.5% | 67.0% | 6.1% | 4.4% | 0.7% | 0.0% | 2.5% |
| bitonic_merge_large_k | 36.5% | 20.2% | 1.4% | 10.3% | 8.7% | 2.7% | 8.0% | 5.4% |
| fill_padding | 1.1% | 0.2% | 0.0% | 24.4% | 31.8% | 0.3% | 0.0% | 4.1% |

## Final profile — `cudaMemset` padding, no fill_padding kernel (`metrics_opt_latest.csv` from `opt_latest.ncu-rep`, 10M elements, 66 launches)

---

A. Overall kernel cost & bottleneck classification

| Kernel Name | Number of Launches | Total Duration (ms) | Percentage of Total Kernel Time | Compute (SM) Throughput (% of SOL) | Memory Throughput (% of SOL) | Achieved Occupancy (%) | Theoretical Occupancy (%) |
|---|---|---|---|---|---|---|---|
| bitonic_merge_small_k | 1 | 1.621 | 41.0% | 78.19% | 33.06% | 93.78% | 100.0% |
| bitonic_merge_vec_int4 | 55 | 1.125 | 28.5% | 31.88% | 66.16% | 73.22% | 100.0% |
| bitonic_merge_large_k | 10 | 1.205 | 30.5% | 78.89% | 68.80% | 93.64% | 100.0% |
| **Total** | **66** | **3.952** | **100.0%** | — | — | — | — |

B. Warp scheduling & stall reasons

| Kernel Name | Active Warps per Scheduler | Eligible Warps per Scheduler | No Eligible Cycles (%) | Warp Cycles per Issued Instruction | Stall: Barrier (%) | Stall: Short Scoreboard (%) | Stall: Long Scoreboard (%) | Stall: Other (%) |
|---|---|---|---|---|---|---|---|---|
| bitonic_merge_small_k | 15.00 | 6.91 | 19.2% | 18.6 | 5.6% | 0.9% | 0.1% | 93.5% |
| bitonic_merge_vec_int4 | 11.95 | 0.94 | 66.9% | 36.1 | 0.0% | 4.3% | 67.4% | 28.3% |
| bitonic_merge_large_k | 14.98 | 6.30 | 16.9% | 18.0 | 8.0% | 8.7% | 1.4% | 81.9% |

C. Memory access pattern

| Kernel Name | Global Load Sectors (M) | Global Store Sectors (M) | Sectors per Request (load) | DRAM Read (GiB) | DRAM Written (GiB) | L1 Hit Rate (%) | L2 Hit Rate (%) | Shared Load Wavefronts (M) | Shared Store Wavefronts (M) | Shared Bank Conflicts (M) | Bank Conflicts per Wavefront |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bitonic_merge_small_k | 1.0 | 1.0 | 2.00 | 0.03 | 0.01 | 47.2% | 51.2% | 55.6 | 41.3 | 0.037 | 0.0004 |
| bitonic_merge_vec_int4 | 57.7 | 57.5 | 16.00 | 1.80 | 0.63 | 44.1% | 51.2% | 0.0 | 0.0 | 0.000 | — |
| bitonic_merge_large_k | 10.5 | 10.5 | 2.00 | 0.33 | 0.13 | 46.7% | 51.2% | 79.1 | 47.5 | 0.559 | 0.0044 |
| **Total** | **69.2** | **69.1** | **7.38** | **2.16** | **0.77** | **44.5%** | **51.2%** | **134.7** | **88.8** | **0.596** | **0.0027** |

B2. Full stall breakdown (percentage of warp cycles per issued instruction)

| Kernel Name | not selected | math pipe throttle | long scoreboard | wait | short scoreboard | dispatch stall | barrier | mio throttle |
|---|---|---|---|---|---|---|---|---|
| bitonic_merge_small_k | 40.7% | 28.4% | 0.1% | 11.1% | 0.9% | 6.7% | 5.6% | 0.8% |
| bitonic_merge_vec_int4 | 5.1% | 4.5% | 67.4% | 6.1% | 4.3% | 0.7% | 0.0% | 2.5% |
| bitonic_merge_large_k | 36.5% | 20.1% | 1.4% | 10.3% | 8.7% | 2.7% | 8.0% | 5.4% |
