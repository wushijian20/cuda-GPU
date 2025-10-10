


int main(void)
{
    setGPU(); 

    int iElemCount = 512;  // set the number of elements in the vectors
    size_t stBytesCount = IElemCount * sizeof(float); // calculate the size of the vectors in bytes

    
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

    if (fpHost_A != NULL && fpHost_B != NULL && fpHost_C != NULL)
    {
        memset
    }


}