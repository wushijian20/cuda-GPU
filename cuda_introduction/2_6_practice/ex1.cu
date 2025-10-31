 #include <stdio.h>

 __global__ void hello_from_gpu(void)
 {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;

    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    printf("Hello World from GPU! (block %d, thread %d, id %d)\n", bid, tid, id);
 }

 int main(void)
 {
    printf("Hello World from CPU!\n");
    hello_from_gpu<<<2,4>>>();
    cudaDeviceSynchronize();

    return 0;
 }
