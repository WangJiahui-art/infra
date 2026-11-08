#include <cstdio>
#include <cuda_runtime.h>
#include <cmath>

/**
 * 针对朴素版本存在的单线程串行处理慢的问题.
 * 采用 : 一个block并行处理一行, 每个线程结果写入共享内存, 通过规约求 '全局最大值' 和 '分母'.
 */

__global__ void Softmax(float *input, float *output, int M, int N)
{
    extern __shared__ float smem[]; // 申请共享内存

    float *start = input + blockIdx.x * N; // 每个block要读取的数据起点指针
    float *y = output + blockIdx.x * N;    // 每个block要写入的数据起点指针

    int tid = threadIdx.x; // 线程映射

    float max_num = -INFINITY; // 每个线程存储局部最大值

    /* 获取行最大值 : 跨步循环 + 规约 */
    for (int i = tid; i < N; i += blockDim.x)
    {
        max_num = fmaxf(start[i], max_num);
    }
    smem[tid] = max_num; // 写入共享内存
    __syncthreads();     // 线程同步

    for (int s = blockDim.x / 2; s > 0; s >>= 1) // 规约
    {
        if (tid < s)
        {
            smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        }
        __syncthreads();
    }
    max_num = smem[0]; // 存储全局最大值
    __syncthreads();

    /* 计算分母 : 跨步循环 + 规约 */
    float t_sum = 0.0f; // 存储线程的局部和
    for (int s = tid; s < N; s += blockDim.x)
    {
        t_sum += expf(start[s] - max_num);
    }

    smem[tid] = t_sum; // 局部和 写入共享内存
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) // 规约
    {
        if (tid < s)
        {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    float sector = 1.0f / smem[0]; // 把分母加载到寄存器

    /* 计算每一个输出值并写入 : 跨步循环 */
    for (int i = tid; i < N; i += blockDim.x)
    {
        y[i] = sector * expf(start[i] - max_num);
    }
}

int main()
{

    // 定义矩阵的维度
    int M = 4096;
    int N = 4096;

    // 计算内存空间
    size_t size = sizeof(float) * M * N;

    // 申请host内存
    float *input = (float *)malloc(size);
    float *output = (float *)malloc(size);

    // 初始化input
    std::fill(input, input + M * N, 1.0f);

    // 申请设备内存
    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    // 数据host -> device
    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);

    // 传参 调用 //! 调用的时候要分配共享内存
    Softmax<<<M, 256, 256 * sizeof(float)>>>(d_input, d_output, M, N);

    // d_output -> output
    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
    {
        printf("%f ", output[30 * N + i]);
    }
    printf("\n");

    /* 内存释放 */
    free(input);
    free(output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}