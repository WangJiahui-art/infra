#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

/**
 * 实现online softmax
 * 每个线程处理一行, 第一次扫描同时维护 当前最大值 and 当前和 .
 */

__global__ void online_softmax(float *input, float *output, int M, int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x; // 线程要处理的行

    if (row > M)
        return;

    float *x = input + row * N;
    float *y = output + row * N;

    /* 一次扫描, 更新当前最大值和当前和 */
    float max_num = -INFINITY;
    float sum = 0.0f;
    for (int i = 0; i < N; i++) // 逐元素遍历
    {
        float temp = max(max_num, x[i]);
        sum = expf(max_num - temp) * sum + expf(x[i] - temp);
        max_num = temp;
    }

    /* 归一化 */
    for (int i = 0; i < N; i++)
    {
        y[i] = expf(x[i] - max_num) / sum;
    }
}

int main()
{
    /* 矩阵规模 */
    int M = 4096;
    int N = 4096;

    size_t size = M * N * sizeof(float); // 申请内存大小

    /* 申请Host内存 */
    float *h_X = (float *)malloc(size);
    float *h_Y = (float *)malloc(size);

    /* 申请Device内存 */
    float *d_X, *d_Y;
    cudaMalloc(&d_X, size);
    cudaMalloc(&d_Y, size);

    std::fill(h_X, h_X + M * N, 1.0f); // 初始化h_X;

    cudaMemcpy(d_X, h_X, size, cudaMemcpyDeviceToHost); // 数据拷贝到Device

    online_softmax<<<256, 16>>>(d_X, d_Y, M, N); // 启动内核

    // 结果拷贝到Host
    cudaMemcpy(h_Y, d_Y, size, cudaMemcpyDeviceToHost);

    // 显示一行
    for (int i = 0; i < N; i++) // 打印h_X[20]
    {
        printf("%f ", h_Y[20 * N + i]);
    }
    printf("\n");

    /* 释放内存 */
    free(h_X);
    free(h_Y);
    cudaFree(d_X);
    cudaFree(d_Y);

    return 0;
}
