#include <stdio.h>
#include "../tools/common.cuh"


__global__ void addFormGPU(float *a, float *b, float *c, int elemCount)
{
    const int bid = blockIdx.x; // block index
    const int tid = threadIdx.x; // thread index within the block
    const int id = tid + bid * blockDim.x; // global thread index

    c[id] = a[id] + b[id]; // perform the addition
}


void initialData(float *ip, int elemCount)
{
    for (int i = 0; i < elemCount; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f; // generate a random float number between 0.0 and 25.5
    }
    return;
}

int main(void)
{
    setGPU(); 

    int iElemCount = 512;  // set the number of elements in the vectors
    size_t stBytesCount = iElemCount * sizeof(float); // calculate the size of the vectors in bytes


    // allocate the host memory
    // declares three pointers to float values
    // each pointer will later refer to the start of an array(vector) in memory
    //     fpHost_A → [ a0, a1, a2, a3, ... ]
    //     fpHost_B → [ b0, b1, b2, b3, ... ]
    //     fpHost_C → [ c0, c1, c2, c3, ... ]

    float *fpHost_A, *fpHost_B, *fpHost_C; // host vectors

    // malloc() stands for "memory allocation". It is a C standard library function
    // (declared in <stdio.h>) used to dynamically allocate memory at runtime.
    // When you write a program, you sometimes don't know how much memory you will
    // need until the program runs. malloc() allows you to ask the operating system
    // to give your program a specific number of bytes from the heap momory.

    // void* malloc(size_t size); 
    // in the above function declaration, size_t is an unsigned integer type
    // size is the number of bytes to allocate
    // malloc() returns a pointer of type void* which can be cast to the desired type
    // if the allocation fails, it returns a null pointer

    // int *p;
    // p = (int*)malloc(10 * sizeof(int)); // allocate space for 10 integers
    // sizeof(int) is usually 4 bytes, so this allocates 40 bytes
    // malloc(40) asks for 40 bytes from the heap.
    // malloc() returns a pointer to the first byte of this memory block (type void*), 
    // which is then cast to an int* and assigned to p
    // now p points to an array of 10 integers: p → [ ?, ?, ?, ?, ?, ?, ?, ?, ?, ? ]

    // malloc(stBytesCount) allocates a block of memory of size stBytesCount bytes on the heap
    // It returns a generic pointer (void*) to that memory. Then we cast that pointer to a float*
    // Now fpHoster_A points to the start of an allocated array large enough to hold however many
    // floats stBytesCount represents. The same logic applies to fpHost_B and fpHost_C.
    fpHost_A = (float*)malloc(stBytesCount); // host input vector
    fpHost_B = (float*)malloc(stBytesCount); // host input vector
    fpHost_C = (float*)malloc(stBytesCount); //host result vector



    // memset() means "memory set". It's a function from <string.h> used to 
    // fill a block of memory with a specific value.
    // void  *memset(void *ptr, int value, size_t num);
    // ptr: pointer to the start of the memory block you want to fill
    // value: the value to set (converted to an unsigned char)
    // num: number of bytes to be set to the value

    if (fpHost_A != NULL && fpHost_B != NULL && fpHost_C != NULL)
    {
        // Initialize the host input vectors
        memset(fpHost_A, 0, stBytesCount);
        memset(fpHost_B, 0, stBytesCount);
        memset(fpHost_C, 0, stBytesCount);
    }
    else
    { 
        printf("Fail to allocate host memory!\n");
        exit(-1);
    }


    // & is the address-of operator in c/c++. It returns the memory address of its operand.
    // *fpDev_A means "the value stored in GPU memory at the address held by fpDev_A"
    float *fpDev_A, *fpDev_B, *fpDev_C; // device vectors
    cudaMalloc((float**)&fpDev_A, stBytesCount); // device input vector A
    cudaMalloc((float**)&fpDev_B, stBytesCount); // device input vector B   
    cudaMalloc((float**)&fpDev_C, stBytesCount); // device result vector C

    if(fpDev_A != NULL && fpDev_B != NULL && fpDev_C != NULL)
    {
        // Initialize the device input vectors
        cudaMemset(fpDev_A, 0, stBytesCount);
        cudaMemset(fpDev_B, 0, stBytesCount);
        cudaMemset(fpDev_C, 0, stBytesCount);
    }
    else
    { 
        printf("Fail to allocate device memory!\n");
        free(fpHost_A);
        free(fpHost_B);
        free(fpHost_C);
        exit(-1);
    }

    srand(2025); // set the seed for rand()
    initialData(fpHost_A, iElemCount); // initialize host vector A
    initialData(fpHost_B, iElemCount); // initialize host vector B

    cudaMemcpy(fpDev_A, fpHost_A, stBytesCount, cudaMemcpyHostToDevice); // copy data from host to device
    cudaMemcpy(fpDev_B, fpHost_B, stBytesCount, cudaMemcpyHostToDevice); // copy data from host to device       
    cudaMemcpy(fpDev_C, fpHost_C, stBytesCount, cudaMemcpyHostToDevice); // copy data from host to device

    dim3 block(32); // set the number of threads per block
    dim3 grid(iElemCount / 32); // set the number of blocks in the grid

    addFormGPU<<<grid, block>>>(fpDev_A, fpDev_B, fpDev_C, iElemCount); // launch the kernel
    cudaDeviceSynchronize(); // synchronize the device

    cudaMemcpy(fpHost_C, fpDev_C, stBytesCount, cudaMemcpyDeviceToHost); // copy result from device to host

    for (int i = 0; i < 10; i++)
    {
        printf("idx=%2d\tmaxtrixA: %.2f\tmatrix_B:%.2f\tresult=%.2f\n", i+1, fpHost_A[i], fpHost_B[i], fpHost_C[i]);
    }

    // free the device memory
    free(fpHost_A);
    free(fpHost_B); 
    free(fpHost_C);
    cudaFree(fpDev_A);
    cudaFree(fpDev_B);
    cudaFree(fpDev_C);

    cudaDeviceReset();
    return 0;
}