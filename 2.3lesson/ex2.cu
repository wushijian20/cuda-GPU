#include <stdio.h>

__global__ void hello_from_gpu()
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int id =  threadIdx.x + blockIdx.x * blockDim.x;
    printf("Hello World from block %d and thread %d, global id %d\n", bid, tid, id);

}

int main(void)
{
    hello_from_gpu<<<2,4>>>();  // grid_size: 2 (2 blocks in a grid), block_size: 4 (4 thread in a block)
    cudaDeviceSynchronize();

    return 0;
}