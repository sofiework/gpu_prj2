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


/**********************************************************************************
 * 
 * Implement your utility functions here
 * 
 **********************************************************************************/

__global__ void fill_padding(DTYPE *arrD, int size, int pad_size, DTYPE sentinel) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid + size < pad_size) {
        arrD[tid + size] = sentinel;
    }
}

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
        int block = 128;
        int grid = (pad_size - size + block - 1) / block;
        DTYPE sentinel = std::numeric_limits<DTYPE>::max();
        fill_padding<<<grid, block>>>(arrD, size, pad_size, sentinel);
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

/**
 * This functiuon transfers the sorted data from Device to Host
 */
DTYPE *dev_to_host() {

    size_t total_size = sizeof(DTYPE) * size; // non-pad size

    // allocate in CPU memory
    arrSortedGpu = (DTYPE *)malloc(total_size);

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
    free(arrCpu);
    free(arrSortedGpu);

    cudaFree(arrD);
}
