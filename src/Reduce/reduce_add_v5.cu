# include <cstdio>
# include <cuda_runtime.h>



template <unsigned int BLOCK_SIZE>
__device__ void warp_reduce(volatile float* smem, unsigned int tid){
    if (BLOCK_SIZE >= 64) {
        smem[tid] += smem[tid + 32];
    }
    smem[tid] += smem[tid + 16];
    smem[tid] += smem[tid + 8];
    smem[tid] += smem[tid + 4];
    smem[tid] += smem[tid + 2];
    smem[tid] += smem[tid + 1];
}



template <unsigned int BLOCK_SIZE>
__global__ void reduce(float* input, float* output, int n){
    // 计算索引
    unsigned int tid = threadIdx.x;
    unsigned int gid = threadIdx.x + blockIdx.x * BLOCK_SIZE * 2;

    // 加载数据
    extern __shared__ float smem[];
    float val = 0.0f;
    // 加载一个数
    if (gid < n){
        val += input[gid];
    }
    // 加载相隔BLOCK_SIZE的数.
    if ((gid + BLOCK_SIZE) < n){
        val += input[gid + BLOCK_SIZE];
    }

    // 加载到shared memory
    smem[tid] = val;
    __syncthreads();

    // 编译器按模板展开代码
    if (BLOCK_SIZE >= 512) {
        if (tid < 256) {
            smem[tid] += smem[tid + 256];
        }
        __syncthreads();
    }
    if (BLOCK_SIZE >= 256) {
        if (tid < 128) {
            smem[tid] += smem[tid + 128];
        }
        __syncthreads();
    }
    if (BLOCK_SIZE >= 128) {
        if (tid < 64) {
            smem[tid] += smem[tid + 64];
        }
        __syncthreads();
    }

    // 剩余 32个线程,64个数字 采用warp内规约
    if (tid < 32){
        warp_reduce<BLOCK_SIZE>(smem, tid);
    }

    // 写回HBM
    if (tid == 0){
        output[blockIdx.x] = smem[0];
    }
}
