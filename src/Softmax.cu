#include <cstdio>
#include <cuda_runtime.h>
#include <algorithm>

__global__ void Softmax(float *input, float *output, int M, int N)
{
    /**
     * 实现矩阵按行Softmax激活.
     *
     * @param input : 源矩阵
     * @param output : 目标矩阵
     * @param M : 源矩阵的行数
     * @param N : 源矩阵的列数
     *
     * 每个线程执行一行.
     *
     */

    // 线程映射到矩阵的行
    int tid = threadIdx.x;
    int row = blockDim.x * blockIdx.x + tid;
    if (row > M)
    {
        return;
    }

    float *x = input + N * row;
    float *y = output + N * row;

    // 行扫描 记录行最大值
    float max_num = -INFINITY;
    for (int i = 0; i < N; i++)
    {
        max_num = fmaxf(max_num, x[i]);
    }

    // 计算分母
    float sum = 0.0f;
    for (int i = 0; i < N; i++)
    {
        sum += expf(x[i] - max_num);
    }

    // 计算输出矩阵的行
    float sector = 1.0f / sum;
    for (int i = 0; i < N; i++)
    {
        y[i] = expf(x[i] - max_num) * sector;
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

    // 传参 调用
    Softmax<<<4096, 256>>>(d_input, d_output, M, N);

    // d_output -> output
    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
    {
        printf("%f ", output[0 * N + M]);
    }
    printf("\n");
    return 0;
}