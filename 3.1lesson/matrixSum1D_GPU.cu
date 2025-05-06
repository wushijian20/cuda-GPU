#include <stdio.h>
#include "../tools/common.cuh"

void initialData(float *addr, int elemCount)
{
    for (int i = 0; i < elemCount; i++)
    {
        addr[i] = (float)(rand() & 0xFF) / 10.f;
    }
    return;
}


int main(void)
{
    // set GPU device
    setGPU();
    
    // 分配主机和设备内存，并初始化
    int iElemCount = 512;
    size_t stBytesCount = iElemCount * sizeof(float);

    // 分配主机内存， 并初始化
    float *fpHost_A, *fpHost_B, *fpHost_C;
    fpHost_A = (float *)malloc(stBytesCount);
    fpHost_B = (float *)malloc(stBytesCount);
    fpHost_C = (float *)malloc(stBytesCount);

    if (fpHost_A != NULL && fpHost_B != NULL && fpHost_c != NULL)
    {
        memset(fpHost_A, 0, stBytesCount); // 主机内存初始化为0
        memset(fpHost_B, 0, stBytesCount);
        memset(fpHost_C, 0, stBytesCount);
    }
    else
    {
        printf("Fail to allocate host memory!\n");
        exit(-1);
    }

    // 分配设备内存， 并初始化
    float *fpDevice_A, *fpDevice_B, *fpDevice_C;
    cudaMalloc((float **)&fpDevice_A, stBytesCount);
    cudaMalloc((float **)&fpDevice_B, stBytesCount);
    cudaMalloc((float **)&fpDevice_C, stBytesCount);

    if (fpDevice_A != NULL && fpDevice_B != NULL && fpDevice !=NULL)
    {
        cudaMemset(fpDevice_A, 0, stBytesCount);
        cudaMemset(fpDevice_B, 0, stBytesCount);
        cudaMemset(fpDevice_C, 0, stBytesCount);
    }
    else
    {
        print("fail to allocate memory\n");
        free(fpHost_A);
        free(fpHost_B);
        free(fpHost_C);
        exit(-1);
    }

    // 初始化主机中数据
    srand(666); //设置随机种子
    initialData(fpHost_A, iElemCount);
    initialData(fpHost_B, iElemCount);











}




