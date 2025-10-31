#pragma once
#include <stdlib.h>
#include <stdio.h>

cudaError_t ErrorCheck(cudaError_t error_code, const char* filename, int lineNumber)
{
    if(error_code != cudaSuccess)
    {
        printf("CUDA error: \r\ncode=%d, name=%s, description=%s\r\nfile=%s, line%d\r\n",
            error_code, cudaGetErrorName(error_code), 
            cudaGetErrorString(error_code), filename, lineNumber);
        return error_code;
    }
    return error_code;
}

void setGPU()
{
       int iDeviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&iDeviceCount);

    if(error != cudaSuccess || iDeviceCount == 0)
    {
        printf("No CUDA GPU found!\n");
        exit(-1);
    }
    else
    {
        printf("The number of Cuda GPU(s): %d\n", iDeviceCount);

    }

    int iDev = 0;
    error = cudaSetDevice(iDev);
    if(error != cudaSuccess)
    {
        printf("fail to set GPU %d as the current device\n", iDev);
        exit(-1);
    }
    else
    {
        printf("set GPU %d for the current device\n", iDev);
    }


}