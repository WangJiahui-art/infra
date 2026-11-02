#include <cstdio>         // 标准的输入输出库
#include <cstdlib>        // 标准工具库
#include <cuda_runtime.h> // CUDA应用编程接口

#define CUDA_CHECK(call)                                                            \
    do                                                                              \
    {                                                                               \
        cudaError_t err = call;                                                     \
        if (err != cudaSuccess)                                                     \
        {                                                                           \
            fprintf(stderr, "CUDA is error in %s:%d, %s", __FILE__, __LINE__, err); \
            exit(EXIT_FAILURE);                                                     \
        }                                                                           \
    }while(0)

__global__ void vectorAdd(const float *A, const float *B, float *C, int N)
{
    /*
        向量加法函数
    */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        C[idx] = A[idx] + B[idx];
    }
}

int main()
{
    // 定义元素数量
    int N = 1 << 10;
    size_t size = N * sizeof(float); // 计算需要的内存空间

    // 分配主机内存
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // 初始化A, B数组
    for (int i = 0; i < N; i++)
    {
        h_A[i] = float(i);
        h_B[i] = float(i) * 2;
    }

    // 分配cuda的HBM内存
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // 数据传递
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 配置并启动kernal
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    vectorAdd<<<grid_size, block_size>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 回传结果
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 结果打印
    for (int i = 0; i < 20; i++)
    {
        printf("%f  ", h_C[i]);
    }

    // 释放内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}