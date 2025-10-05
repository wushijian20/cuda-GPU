#include <stdio.h>

__global__ void hello_from_gpu(void) 
{
    printf("Hello World from GPU!\n");
}

int main(void)
{
    // Launch kernel with 4 threads
    hello_from_gpu<<<2, 4>>>();
    
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    
    return 0;
}