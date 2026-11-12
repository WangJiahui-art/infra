#include <cstdio>
#include <cuda_runtime.h>

/*
    解决v3 版本存在的 最后一个warp进行规约时的__syncthread()浪费所有线程的问题
*/


// warp内规约
__device__ void warp_reduce(volatile float* smem, unsigned int tid){
    smem[tid] += smem[tid + 32];
    smem[tid] += smem[tid + 16];
    smem[tid] += smem[tid + 8];
    smem[tid] += smem[tid + 4];
    smem[tid] += smem[tid + 2];
    smem[tid] += smem[tid + 1];
}

__global__ void reduce(float *input, float *output, int n)
{

    // 线程计算
    unsigned int tid = threadIdx.x;
    unsigned int gid = threadIdx.x + blockDim.x * blockIdx.x * 2;

    // 数据加载 每个线程加载两个数据
    extern __shared__ float smem[];
    float val = 0.0f;
    if (gid < n)
    {
        val += input[gid];
    }
    if ((gid + blockDim.x) < n)
    {
        val += input[gid + blockDim.x];
    }
    smem[tid] = val;
    __syncthreads();

    // 规约计算, 只执行大于一个warp的部分
    for (unsigned int step = blockDim.x / 2; step > 32u; step >>= 1)
    {
        if (tid < step)
        {
            smem[tid] += smem[tid + step];
        }
        __syncthreads();
    }

    // 执行warp内规约
    if (tid < 32){
        warp_reduce(smem, tid);
    }

    // 写回HBM
    if (tid == 0)
    {
        output[blockIdx.x] = smem[0];
    }
}