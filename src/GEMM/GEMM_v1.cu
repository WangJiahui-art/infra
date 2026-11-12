#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

template <uint32_t BM, uint32_t BN, uint32_t BK, uint32_t BLOCK_SIZE>
__global__ void GEMM(const float *A, const float *B, float *C, uint32_t M, uint32_t K, uint32_t N)
{
    // todo 1 申请内存空间
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // todo 2 计算索引和tile起始点
    uint32_t row_start = blockIdx.y * BM;
    uint32_t col_start = blockIdx.x * BN;
    uint32_t tid = threadIdx.x;

    // todo 3 加载tileA时， 重新映射thread
    constexpr uint32_t A_load_blockDim_x = BK;
    constexpr uint32_t A_load_blockDim_y = BLOCK_SIZE / BK;
    uint32_t A_load_threadIdx_x = tid % A_load_blockDim_x;
    uint32_t A_load_threadIdx_y = tid / A_load_blockDim_x;

    // todo 4 加载tileB时， 重映射thread
    constexpr uint32_t B_load_blockDim_x = 32;
    constexpr uint32_t B_load_blockDim_y = BLOCK_SIZE / B_load_blockDim_x;
    uint32_t B_load_threadIdx_x = tid % B_load_blockDim_x;
    uint32_t B_load_threadIdx_y = tid / B_load_blockDim_x;

    // 计算tileC时， 重映射thread
    constexpr uint32_t C_blockDim_x = 16;
    constexpr uint32_t C_blockDim_y = BLOCK_SIZE / C_blockDim_x;
    uint32_t C_threadIdx_x = tid % C_blockDim_x;
    uint32_t C_threadIdx_y = tid / C_blockDim_x;

    // 每个线程负责Tm * Tn 个输出。
    constexpr uint32_t Tm = BM / C_blockDim_y;
    constexpr uint32_t Tn = BN / C_blockDim_x;
    float C_T[Tm][Tn] = {0.0f};

    // K loop
    for (uint32_t k = 0u; k < K; k += BK)
    {
        /*
            A(BM, K) @ B(K * BN)的计算
        */
#pragma unroll
        for (uint32_t i = A_load_threadIdx_y; i < BM; i += A_load_blockDim_y)
        {
            /*
                协作加载tileA, 跨步循环覆盖BM行
            */
            // 线程要加载的数据在A矩阵中的绝对位置
            uint32_t r = row_start + i;
            uint32_t c = k + A_load_threadIdx_x;

            // 数据加载到As
            As[i][A_load_threadIdx_x] = (r < M && c < K) ? A[r * K + c] : 0.0f;
        }

#pragma unroll
        for (uint32_t i = B_load_threadIdx_x; i < BN; i += B_load_blockDim_x)
        {
            /*
                协作加载tileB,跨步循环覆盖BN列
            */
            // 线程要加载的数据在B矩阵中的位置
            uint32_t r = k + B_load_threadIdx_y;
            uint32_t c = i + col_start;

            // 数据加载到Bs
            Bs[B_load_threadIdx_y][i] = (r < K && c < N) ? B[r * N + c] : 0.0f;
        }

        __syncthreads();

// 外积 As * Bs
#pragma unroll
        for (uint32_t p = 0; p < BK; p++)
        {
            for (uint32_t i = 0; i < Tm; i++)
            {
                int row = C_threadIdx_y + i * C_blockDim_y; // 线程计算的行

                for (uint32_t j = 0; j < Tn; j++)
                {
                    int col = C_threadIdx_x + j * C_blockDim_x;
                    C_T[i][j] += As[row][p] * Bs[p][col];
                }
            }
        }

        __syncthreads();
    }

    // 写回
    for (int i = 0; i < Tm; i++)
    {
        // 计算要写的row和col
        int r = row_start + C_threadIdx_y + i * C_blockDim_y;
        for (int j = 0; j < Tn; j++)
        {
            int c = col_start + C_threadIdx_x + j * C_blockDim_x;
            if (r < M && c < N)
            {
                C[r * N + c] = C_T[i][j];
            }
        }
    }
}