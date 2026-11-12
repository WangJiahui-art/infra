#include <cstdio>
#include <cuda_runtime.h>
/*
采用 折半规约的方式,来缓解Bank Conflict.

*/

__global__ void reduce(float *input, float *output, int n)
{
    // 计算索引
    unsigned int tid = threadIdx.x;
    unsigned int gid = threadIdx.x + blockDim.x * blockIdx.x;

    // HBN -> shared memory
    extern __shared__ float smem[];
    smem[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();

    for (unsigned int step = blockDim.x / 2; step > 0u; step >>= 2)
    {
        if (tid < step)
        {
            smem[tid] += smem[tid + step];
        }
        __syncthreads(); // ! 必须放在if外面, 防止出现死锁.
    }

    // 写回HBM
    if (tid == 0)
    {
        output[blockIdx.x] = smem[0];
    }
}