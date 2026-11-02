#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>


__device__ float warp_reduce_sum(float val)
{
    /*实现warp级的规约*/
    for (int i = 16; i > 0; i >>= 1)
    {
        val += __shfl_down_sync(0xFFFFFFFF, val, i);
    }
    return val;
}


__global__ void block_reduce_sum(float* input, float* output, int N)
{
/*
    执行 block粒度的求和
    采用规约方法
*/
    // 数据加载, 每个线程加载一个数字
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = idx < N ? input[idx] : 0.0f;

    // 进行warp级别的规约
    val = warp_reduce_sum(val);

    // 每个warp的lane0将结果写入共享内存中,用于后续的计算.
    __shared__ float warp_sum[32];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if(lane_id == 0)
    {
        warp_sum[warp_id] = val;
    }
    __syncthreads();

    // 将刚刚的存储每个warp的和的数组加载到第一个warp中.
    int num_warps = (blockDim.x + 31) / 32;
    val = threadIdx.x < num_warps ? warp_sum[threadIdx.x] : 0.0f;
    if (warp_id == 0)
    {
        val = warp_reduce_sum(val);
    }

    // 将每个block的求和结果写入输出数组中.
    if(threadIdx.x == 0)
    {
        output[blockIdx.x] = val;
    }
}

int main()
{
    // 申请cuda内存空间
    int N = 300;
    size_t size_i = sizeof(float) * N;
    size_t size_o = sizeof(float) * ((N + 255) / 256);
    
    float *h_input = (float *)malloc(size_i);
    float *h_output = (float *)malloc(size_o);
    float *d_input, *d_output;
    cudaMalloc(&d_input, size_i);
    cudaMalloc(&d_output, size_o);

    // 初始化输入数组
    for (int i = 0; i < N; i ++){
        h_input[i] = i;
    }

    //拷贝到cuda上
    cudaMemcpy(d_input, h_input, size_i, cudaMemcpyHostToDevice);

    block_reduce_sum<<<2, 256>>>(d_input, d_output, N);

    cudaMemcpy(h_output, d_output, size_o, cudaMemcpyDeviceToHost);

    printf("%f, %f", h_output[0], h_output[1]);
}