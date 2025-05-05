#include <stdio.h>

__global__ void hello_from_gpu()
{
    printf("Hello World from the GPU\n");
}

int main(void)
{
    hello_from_gpu<<<2,4>>>();  // grid_size: 2 (2 blocks in a grid), block_size: 4 (4 thread in a block)
    cudaDeviceSynchronize();

    return 0;
}