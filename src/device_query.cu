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

/*
========== Device 0: NVIDIA GeForce RTX 2060 ==========
* CUDA Capability Major/Minor: 7.5
* Total Global Memory: 6144 MB
* Shared Memory per Block: 49152 bytes
* Shared Memory per Multiprocessor: 65536 bytes
* Registers per Block: 65536
* Registers per Multiprocessor: 65536
* Max Threads per Block: 1024
* Max Threads per Multiprocessor: 1024
* Warp Size: 32
* Max Blocks per Multiprocessor: 16
* Clock Rate: 1350 MHz
* Memory Clock Rate: 5501 MHz
* Memory Bus Width: 192 bits
* L2 Cache Size: 3145728 bytes
* Multiprocessor Count: 30
* Concurrent Kernels: 1
* ECC Enabled: 0
* Unified Addressing: 1
* Managed Memory: 1
* Compute Mode: 0
* PCI Bus ID: 1, Device ID: 0
* Kernel Exec Timeout Enabled: 1
*/