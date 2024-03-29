/*
   * A simple example code for warp reduce
   * Especially, this code illustrates '__shfl_xor_sync'
   * This function can be used to get a maximum number within a block
*/

#include <stdio.h>
#include <iostream>
#include <stdlib.h>

__global__ void get_max(float* arr, float* max_val, int block_size) {
    float tmp_max = arr[threadIdx.x];

    for (int mask = block_size/2; mask > 0; mask /= 2) {
        // __shfl_xor_sync(mask_threads, var_to_be_shuffled, lane_mask)
        //      mask_threads: mask of active threads for exchanging data
        //      source idx of var to be shuffled = own_thread_id ^ lane_mask
        float _tmp = __shfl_xor_sync(0xFFFFFFFF, tmp_max, mask);
        tmp_max = (_tmp > tmp_max) ? _tmp : tmp_max;
    }

    max_val[threadIdx.x] = tmp_max;
}

int main() {
    int arr_size = 16;
    int seed = 10000;
    float x[arr_size];
    float val[arr_size];
    srand(seed);
    printf("---arr x---\n");
    for (int i = 0; i < arr_size; i++) {
        x[i] = (rand() / static_cast<float>(RAND_MAX)) * 10.0;
        printf("%.2f ", x[i]);
    }
    printf("\n\n");

    float* x_dev;
    float* val_dev;
    cudaMalloc((void**)&x_dev, arr_size*sizeof(float));
    cudaMalloc((void**)&val_dev, arr_size*sizeof(float));
    cudaMemcpy(x_dev, x, arr_size*sizeof(float), cudaMemcpyHostToDevice);
    
    get_max <<<1, arr_size>>> (x_dev, val_dev, arr_size);

    cudaDeviceSynchronize();
    cudaMemcpy(val, val_dev, arr_size*sizeof(float), cudaMemcpyDeviceToHost);

    printf("---arr y---\n");
    for (int i = 0; i < arr_size; i++) {
        printf("%.2f ", val[i]);
    }
    printf("\n");

    cudaFree(x_dev);
    cudaFree(val_dev);

}
