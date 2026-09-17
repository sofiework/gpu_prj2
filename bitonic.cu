/**
 * 
 * The student is required to add content to this file.  This file is
 * your implementation of the project and will be submitted for grading.
 * 
 */

#include "main.h"
#include "student.h"
#include <limits>

/**********************************************************************************
 * 
 * Implement your GPU device kernel(s) here (e.g., the bitonic sort kernel).
 * 
 **********************************************************************************/
static DTYPE *arrD;
int pad_size = 0;

__global__ void bitonic_merge(DTYPE *arr, int stride, int subarr_len, int num_pairs) {
    // at subarr_len, stride
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    // boundary check
    if (tid >= num_pairs) {
        return;
    }
    // which stride group * stride * 2 + offset
    int i = (tid / stride) * stride * 2 + (tid % stride);

    // dir = even subarr -> ascending
    bool dir = ((i / subarr_len) % 2) == 0;
    DTYPE a = arr[i];
    DTYPE b = arr[i + stride];
    if ((a < b) != dir) {
        arr[i] = b;
        arr[i + stride] = a;
    } 
}

__global__ void bitonic_merge_vec_int4(DTYPE *arr, int stride, int subarr_len, int num_pairs) {
    // at subarr_len, stride
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    // boundary check
    if (tid >= num_pairs) {
        return;
    }

    int start = tid * 8; // per 8 element group
    // which stride group * stride * 2 + offset
    int i = (start / stride) * stride * 2 + (start % stride);

    // dir = even subarr -> ascending
    bool dir = ((i / subarr_len) % 2) == 0;

    // vectorized load - 16Byte
    int4 a_vec = *reinterpret_cast<int4*>(arr + i); // 8 elements * 2B = 16B
    int4 b_vec = *reinterpret_cast<int4*>(arr + i + stride);

    DTYPE* a_arr = reinterpret_cast<DTYPE*>(&a_vec);
    DTYPE* b_arr = reinterpret_cast<DTYPE*>(&b_vec);

    #pragma unroll
    for (int j = 0; j < 8; j++) {
        DTYPE a = a_arr[j];
        DTYPE b = b_arr[j];

        if ((a < b) != dir) {
            a_arr[j] = b; 
            b_arr[j] = a;
        } 
    }

    // vectorized write back
    *reinterpret_cast<int4*>(arr + i) = a_vec;
    *reinterpret_cast<int4*>(arr + i + stride) = b_vec;
}


// shared memory optimization

// load fix tile size on shared memory, and loop k, stride
__global__ void __launch_bounds__(1024, 2) bitonic_merge_small_k(DTYPE *arr) {
    // tid
    int tid = threadIdx.x;

    // allocate on shared mem
    __shared__ DTYPE tile[TILE];

    // phase load and sync
    int phases = TILE / blockDim.x; // 8192 / 1024

    for (int ph = 0; ph < phases; ph++) {
        int offset = ph * blockDim.x + tid; // ph * 1024 + tid
        tile[offset] = arr[blockIdx.x * TILE + offset];
    }

    // sync
    __syncthreads();

    // loop k, stride
    for (int k = 2; k <= TILE; k <<= 1) {
        for (int stride = k/2; stride >= 1; stride >>= 1) {

            // phase compare arr[i], arr[i + stride]
            int num_pairs = TILE / 2 / blockDim.x; // 4096 / 1024
            // math pipe congestion optimization !!! k and stride are power of 2
            const int mask = stride - 1;

            for (int ph = 0; ph < num_pairs; ph++) {
                
                int t = ph * blockDim.x + tid; // [0..4096]
                // int i = (t / stride) * 2 * stride + (t % stride); // group * 2 * stride + offset
                int i = (t << 1) - (t & mask);
                int i_stride = i + stride;

                // swap
                DTYPE a = tile[i]; 
                DTYPE b = tile[i_stride];

                // dir = (global_idx / k) % 2 == 0
                bool dir = ((TILE * blockIdx.x + i) / k) % 2 == 0;
                if ((a < b) != dir) {
                tile[i] = b;
                tile[i_stride] = a;
                }
            }

            // sync after each stride
            __syncthreads();
        }
    }

    // sync after k loops done
    __syncthreads();

    // write back - same phase loop as load
    for (int ph = 0; ph < phases; ph++) {
        int offset = ph * blockDim.x + tid; // ph * 1024 + tid
        arr[blockIdx.x * TILE + offset] = tile[offset];
    }

}


// loop all small strides once for large k
__global__ void __launch_bounds__(1024, 2) bitonic_merge_large_k(DTYPE *arr, int k) {
    // tid
    int tid = threadIdx.x;

    // dir = (global_idx / k) % 2 == 0, uniform at same k
    bool dir = ((TILE * blockIdx.x) / k) % 2 == 0;

    // allocate on shared mem
    __shared__ DTYPE tile[TILE];

    // phase load and sync
    int phases = TILE / blockDim.x;
    for (int ph = 0; ph < phases; ph++) {
        int offset = ph * blockDim.x + tid;
        tile[offset] = arr[TILE * blockIdx.x + offset];
    }

    // sync after read
    __syncthreads();

    // loop strides
    for (int stride = TILE / 2; stride >= 1; stride >>= 1) {

        // phase compare arr[i], arr[i + stride]
        int num_pairs = TILE / 2 / blockDim.x; // 4096 / 1024
        // math pipe congestion optimization !!! k and stride are power of 2
        const int mask = stride - 1; // keep low bits

        for (int ph = 0; ph < num_pairs; ph++) {

            int t = ph * blockDim.x + tid; // [0..4096]
            // int i = (t / stride) * 2 * stride + (t % stride); // group * 2 * stride + offset
            // bit operation optimization
            int i = (t << 1) - (t & mask);
            int i_stride = i + stride;

            // swap
            DTYPE a = tile[i]; 
            DTYPE b = tile[i_stride];

            if ((a < b) != dir) {
                tile[i] = b;
                tile[i_stride] = a;
            }
        }

        // sync after each stride
        __syncthreads();
    }

    // sync and write back
    __syncthreads();
    for (int ph = 0; ph < phases; ph++) {
        int offset = ph * blockDim.x + tid;
        arr[TILE * blockIdx.x + offset] = tile[offset];
    }

}


/**********************************************************************************
 * 
 * Implement your utility functions here
 * 
 **********************************************************************************/

// __global__ void fill_padding(DTYPE *arrD, int size, int pad_size, DTYPE sentinel) {
//     int tid = blockDim.x * blockIdx.x + threadIdx.x;
//     if (tid + size < pad_size) {
//         arrD[tid + size] = sentinel;
//     }
// }

int padding_size(int size) {
    // pad size to power of 2
    int pad_size = 1;
    while (pad_size < size) {
        pad_size <<= 1;
    }
    return pad_size;

}



/**********************************************************************************
 * 
 * Implement the three main program functions
 * 
 **********************************************************************************/


/**
 * This function transfers data from Host to Device
 */


void host_to_dev()
{   
    // calculate padded size
    pad_size = padding_size(size);
    
    // malloc on device
    size_t total_pad_size = sizeof(DTYPE) * pad_size;
    size_t total_unpad_size = sizeof(DTYPE) * size;
    cudaMalloc((void **)&arrD, total_pad_size);

    // fill padding if padded
    if (pad_size > size) {
        // DTYPE sentinel = std::numeric_limits<DTYPE>::max();
        // int block = 128;
        // int grid = (pad_size - size + block - 1) / block;
        // fill_padding<<<grid, block>>>(arrD, size, pad_size, sentinel);

        cudaMemset(arrD + size, 0x7F, (pad_size - size) * sizeof(DTYPE)); // int value use lower 8 bit
    }
    
    // copy device <- host
    cudaMemcpy(arrD, arrCpu, total_unpad_size, cudaMemcpyHostToDevice);

}

/**
 * This function performs the bitonic sort and merge by calling the
 * kernels you have defined in the section above
 */
void bitonic_sort()
{   
    // // NAIVE SOLUTION
    // int num_pair = pad_size / 2;

    // // subarr_len k in [2, 4, 8,... N]
    // for (int k = 2; k <= pad_size; k <<= 1) {

    //     // stride in [k/2, k/4 ... 1]
    //     for (int stride = k/2; stride >= 1; stride >>= 1) {

    //         // N/2 threads at each depth
    //         // parallel compare arr[i] and arr[i + stride]
    //         int block = 512;
    //         int grid = (pad_size/2 + block - 1) / block; // each stage does N/k * k/2 compares
    //         bitonic_merge<<<grid, block>>>(arrD, stride, k, num_pair);
    //     }

    // }
    // printf("Running NAIVE approach\n");

    // OPTIMIZATION
    // printf("Running shared memory OPTIMIZATION approach\n");
    // base case - pad_size < TILE
    if (pad_size < TILE) {
        int num_pair = pad_size / 2;

        // subarr_len k in [2, 4, 8,... N]
        for (int k = 2; k <= pad_size; k <<= 1) {

            // stride in [k/2, k/4 ... 1]
            for (int stride = k/2; stride >= 1; stride >>= 1) {

                // N/2 threads at each depth
                // parallel compare arr[i] and arr[i + stride]
                int block = 512;
                int grid = (pad_size/2 + block - 1) / block; // each stage does N/k * k/2 compares
                bitonic_merge<<<grid, block>>>(arrD, stride, k, num_pair);
            }
        }
    }

    else {
        // launch kernel A: all stages k in [2, 8192] in one launch
        int grid = (pad_size + TILE - 1) / TILE;
        bitonic_merge_small_k<<<grid, 1024>>>(arrD);

        // launch kernel B: k in (8192, pad_size]
        for (int k = TILE * 2; k <= pad_size; k <<= 1) {
            
            // // NAIVE
            // // large stride [k/2, 8192] that doesn't fit in shared
            // int block = 512;
            // int grid = (pad_size/2 + block - 1) / block;

            // for (int stride = k/2; stride >= TILE; stride >>= 1) {
            //     bitonic_merge<<<grid, block>>>(arrD, stride, k, pad_size/2);
            // }
            // // NAIVE


            // VECTORIZED
            // large stride [k/2, 8192] that doesn't fit in shared
            int block = 512;
            int grid = ((pad_size/2) / 8 + block - 1) / block;

            for (int stride = k/2; stride >= TILE; stride >>= 1) {
                // vectorized
                bitonic_merge_vec_int4<<<grid, block>>>(arrD, stride, k, (pad_size/2) / 8);
            }
            // VECTORIZED

            // small stride in [4096, 1] that fit in shared
            bitonic_merge_large_k<<<pad_size / TILE, 1024>>>(arrD, k);
        }
        
    }

    size_t total_size = sizeof(DTYPE) * size; // non-pad size
    // allocate in CPU memory
    // arrSortedGpu = (DTYPE *)malloc(total_size);
    cudaHostAlloc((void **)&arrSortedGpu, total_size, cudaHostAllocDefault);
    
}

/**
 * This functiuon transfers the sorted data from Device to Host
 */
DTYPE *dev_to_host() {

    size_t total_size = sizeof(DTYPE) * size; // non-pad size
    // copy host <- device
    cudaMemcpy(arrSortedGpu, arrD, total_size, cudaMemcpyDeviceToHost);
    return arrSortedGpu;
}

/**
 * This function frees memory and anything else the student requires 
 * before exiting the program
 */
void cleanup(){
    
    // You may modify/remove these as needed to make your implementation work
    // properly. The defaults provided here allow the skeleton code to compile.   
    
    // // NAIVE
    // free(arrCpu);
    // free(arrSortedGpu);

    // cudaFree(arrD);

    // OPTIMIZATION
    free(arrCpu);
    cudaFreeHost(arrSortedGpu);
    cudaFree(arrD);
    
}
