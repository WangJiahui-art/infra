# include <cstdio>
# include <cuda_runtime.h>

/*
    解决v2 版本存在的 idle 问题
    用的总线程数是v2 版本的一半
*/

__global__ void reduce(float* input, float* output, int n){
    
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
    if ((gid + blockDim.x) < n){
        val += input[gid + blockDim.x];
    }
    smem[tid] = val;
    __syncthreads();

    //规约计算
    for (unsigned int step = blockDim.x / 2; step > 0u; step >>= 1){
        if(tid < step){
            smem[tid] += smem[tid + step];
        }
        __syncthreads();
    }

    //写回HBM
    if (tid == 0){
        output[blockIdx.x] = smem[0];
    }
}

/*
但是这个引入了一个新的问题, 就是在每个块内进行规约的时候,只有blockDim.x / (2^k) 个线程参与计算了.
仍然浪费了大量的线程.
*/