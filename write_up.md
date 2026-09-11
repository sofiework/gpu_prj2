## Optimization

#### Minor
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
