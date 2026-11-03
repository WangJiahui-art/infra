# include <cstdio>
# include <cuda_runtime.h>

__global__ void GEMM(const float* A,const float* B, float* C, int M, int N, int K) {
    // 计算索引
    unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;

    /*
        首先, 数组在内存中是连续的一维的. 
        A 的 size 是 M * K, 所以逻辑的 A[i][j] 对应的是内存的 A[i * k + j].
        B 的 size 是 K * N, 所以逻辑的 B[i][j] 对应的是内存的 A[i * N + j].
        于是:
            逻辑上的 C[row][col] = sigma(i) {A[row][i] * B[i][col]}
        等价于:
            内存上的 C[row * N + col] = sigma(i) {A[row * k + i] * B[i * N + col]}
    */

    // 每个线程计算一个对应的行和列
    if (row < M && col < N) { 
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }

}

int main() {
    int M = 100, K = 50, N = 200;
    // 申请 host 内存
    int sizeA = sizeof(float) * M * K;
    int sizeB = sizeof(float) * K * N;
    int sizeC = sizeof(float) * M * N;

    float* h_A = (float *) malloc(sizeA);
    float* h_B = (float *) malloc(sizeB);
    float* h_C = (float *) malloc(sizeC);

    //初始化
    for (int i = 0; i < M * K; i++){
        h_A[i] = 1.0f;
    }
    for (int i = 0; i < K * N; i++){
        h_B[i] = 0.5f;
    }

    // 申请HBM内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    // 数据拷贝
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    // 执行运算
    dim3 block_size(16, 16);
    dim3 grid_size((M + block_size.x - 1) / block_size.x, (N + block_size.y - 1) / block_size.y);
    GEMM<<<block_size, grid_size>>>(d_A, d_B, d_C, M, N, K);

    //数据拷贝
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    printf("%f, %f\n", h_C[5], h_C[6]);

    // 清理缓存
    free(h_A);
    free(h_B);
    free(h_C);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}