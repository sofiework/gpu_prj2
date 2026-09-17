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