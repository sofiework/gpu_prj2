## Optimizations

### Baseline: naive implementation
Global memory access (no shared memory)
With outer subarr len loop and inner stride loop, complexity is O((logn)^2), for an array length of 10E8, it pads to 2^27, 
    num_launches = 27 * (27 + 1) / 2 = 378

    in each launch, launches
    num_threads = pad_size / 2 = 2^26 threads 
    memory_traffic = num_threads × 2 elements × 4Byte = 512 MB read and 512 MB written per launch

so total memory traffic is 512mb * 378 = ~400 GB of total global memory traffic (read plus write) for a 512mb array (2^27 * 4bytes) is obviously not ideal. With H100 memory bandwidth 3.35 TB/s, the memory traffic time itself is 120ms > target 80ms.

As a result, the dominate optimization is reusing data on shared memory.


#### NCU profile of baseline approach

### Major optimization: tiling through shared memory
By tiling, tile size T of threads load data to shared memory once and reuse 

#### Host Launch Algorithm
##### Pad array size, allocate and move data to device global memory
Pad array size 100M to nearest power of two, pad_size = 2^27.

##### Decide launch grid and tile size
H100 shared memory is 48KB (software default), 48KB / 4Byte = 12K, tile size is the nearest power of two <= 12K is 8192. Giving per thread block limitation 1024 threads, set threads_per_block = 1024, and blocks_per_grid = pad_size 2^27 / 1024.

##### Launch kernelA for k <= 8192
In k ranges [2, 4, ... N], when k <= 8192, the shared memory can hold all k loops and stride loops with fixed tile size 8192.

##### Launch kernelB for k > 8192

##### Clear device memory


#### Device Kernel A Algorithm
##### Phasing loop loading
Each thread need to load 8192 / 1024 = 8 elements to shared memory, so the phasing loop need 8 loops for 1024 threads to load entire tile data. Each thread load stride is 1024 for memory coalescing.

Sync threads after loading tile data.

##### Loop k and stride
Loop k in [2^1, ... 2^13], inner loop stride in [k/2, k/4... 1], sequentially compare and swap arr[i] with arr[i + stride].

Each thread does 4096 / 1024 = 4 pairs, with stride 1024 (eg: thread[t] works on arr[t], arr[t + 1024], arr[t + 2048] arr[t + 3072], and corresponding arr[i + stride]).

Sync threads after each stride done.

##### Write back to global memory


#### Device Kernel B Algorithm
Each kernel takes in current k for direction, fix tile size = 8192, loop over all strides in [4096, 1] inside kernel. Launch grid: <<<pad_size / TILE, 1024>>>

##### Phasing loop loading
Loop phase in [0, TILE / 1024 - 1]. 
Sync threads.

##### Loop stride and compare
For stride in [4096, 1], each thread compare num_pairs = 4096 / 1024, phase loop: derive i, compare and swap.

Sync threads after each stride done.

##### Write back to global memory



### Major optimization: D2H Transfer time

==change to h100 benchmark!!!!
FUNCTIONAL SUCCESS
Array size         : 100000000
CPU Sort Time (ms) : 16661.224609
GPU Sort Time (ms) : 548.342651
GPU Sort Speed     : 182.367722 million elements per second
PERF PASSING
GPU Sort is  30x faster than CPU !!!
H2D Transfer Time (ms): 34.100929
Kernel Time (ms)      : 376.160919
D2H Transfer Time (ms): 138.080826

After kernel shared memory optimization, I got a result with H2D, kernel time looking ok, but D2H takes too much time.

The reason is naive malloc host memory allocate virtual memory only, and allocate physical pages lazily on first touch, so for every page size device copy to CPU, there's one OS page fault overhead to allocate physical page. 

The solution is replacing malloc with cudaHostAlloc, it pins the required physical memory ahead, and also the DMA engine of GPU can directly read and write in this pinned memory.

cudaHostAlloc reference documentation:
https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gb65da58f444e7230d3322b6126bb4902




### Minor optimizations
##### 1. Move data on register rather than read from global

'''
// before - double load each from global memory

if ((arr[i] < arr[i + stride]) != dir) {
    // swap
    DTYPE tmp = arr[i];
    arr[i] = arr[i + stride];
    arr[i + stride] = tmp;
}

// after - once load from global memory
DTYPE a = arr[i];
DTYPE b = arr[i + stride];
if ((a < b) != dir) {
    arr[i] = b;
    arr[i + stride] = a;
} 

'''


### Bugs

##### 1. Out of bound when pad_size < TILE=8192 with optimized kernels
Initially in optimized 

