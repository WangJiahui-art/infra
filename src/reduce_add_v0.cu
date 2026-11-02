# include <cstdio>
# include <cuda_runtime.h>


__global__ void reduce(float* input, float* output, int n){
    //线程索引
    int tid = threadIdx.x;      // 线程在block内的索引
    int gid = threadIdx.x + blockDim.x * blockIdx.x;    // 线程在grid中的索引

    // 数据读取 把HBM中的数据加载到共享内存中
    extern __shared__ float smem[];
    smem[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();    // 等待block内线程同步

    // reduce
    for (int step = 1; step < blockDim.x; step *= 2){
        if (tid % (2 * step) == 0){
            smem[tid] += smem[tid + step];
        }
        __syncthreads(); // 等待block内线程同步
    }

    // 结果写入HBM
    if (tid ==0){
        output[blockIdx.x] = smem[0];
    }
}