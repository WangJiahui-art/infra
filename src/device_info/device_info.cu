/*
    Query the detailed information of the CUDA device.
*/

#include <cuda_runtime.h>
#include <cstdio>

int main()
{
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("Detected %d CUDA device(s)\n\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        printf("========== Device %d: %s ==========\n", dev, prop.name);
        printf("CUDA Capability Major/Minor: %d.%d\n", prop.major, prop.minor);
        printf("Total Global Memory: %.0f MB\n", prop.totalGlobalMem / 1024.0 / 1024.0);
        printf("Shared Memory per Block: %zu bytes\n", prop.sharedMemPerBlock);
        printf("Shared Memory per Multiprocessor: %zu bytes\n", prop.sharedMemPerMultiprocessor);
        printf("Registers per Block: %d\n", prop.regsPerBlock);
        printf("Registers per Multiprocessor: %d\n", prop.regsPerMultiprocessor);
        printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("Max Threads per Multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("Warp Size: %d\n", prop.warpSize);
        printf("Max Blocks per Multiprocessor: %d\n", prop.maxBlocksPerMultiProcessor);
        printf("Clock Rate: %.0f MHz\n", prop.clockRate / 1000.0);
        printf("Memory Clock Rate: %.0f MHz\n", prop.memoryClockRate / 1000.0);
        printf("Memory Bus Width: %d bits\n", prop.memoryBusWidth);
        printf("L2 Cache Size: %d bytes\n", prop.l2CacheSize);
        printf("Multiprocessor Count: %d\n", prop.multiProcessorCount);
        printf("Concurrent Kernels: %d\n", prop.concurrentKernels);
        printf("ECC Enabled: %d\n", prop.ECCEnabled);
        printf("Unified Addressing: %d\n", prop.unifiedAddressing);
        printf("Managed Memory: %d\n", prop.managedMemory);
        printf("Compute Mode: %d\n", prop.computeMode);
        printf("PCI Bus ID: %d, Device ID: %d\n", prop.pciBusID, prop.pciDeviceID);
        printf("Kernel Exec Timeout Enabled: %d\n", prop.kernelExecTimeoutEnabled);
        printf("\n");
    }
    return 0;
}