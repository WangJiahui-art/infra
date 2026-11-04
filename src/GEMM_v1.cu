# include <cstdio>
# include <cuda_runtime.h>

template <unsigned int BM, unsigned int BN, unsigned int BK, unsigned int BLOCK_SIZE>
__global__ void GEMM(const float* A, const float* B, float* C, unsigned int M, unsigned int K, unsigned int N)
{
    // todo 1 申请内存空间
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // todo 2 计算索引和tile起始点
    unsigned int row_start = blockIdx.y * BM;
    unsigned int col_start = blockIdx.x * BN;
    unsigned int tid = threadIdx.x;

    // todo 3 加载tileA时， 重新映射thread
    constexpr unsigned int A_load_blockDim_x = BK;
    constexpr unsigned int A_load_blockDim_y = BLOCK_SIZE / BK;
    unsigned int A_load_threadIdx_x = tid % A_load_blockDim_x;
    unsigned int A_load_threadIdx_y = tid / A_load_blockDim_x;

    // todo 4 加载tileB时， 重映射thread
    constexpr unsigned int B_load_blockDim_x = 32;
    constexpr unsigned int B_load_blockDim_y = BLOCK_SIZE / B_load_blockDim_x;
    unsigned int B_load_threadIdx_x = tid % B_load_blockDim_x;
    unsigned int B_load_threadIdx_y = tid / B_load_blockDim_x;

    // 计算tileC时， 重映射thread
    constexpr unsigned int C_blockDim_x = 16;
    constexpr unsigned int C_blockDim_y = BLOCK_SIZE / C_blockDim_x;
    unsigned int C_threadIdx_x = tid % C_blockDim_x;
    unsigned int C_threadIdx_y = tid / C_blockDim_x;

    // 每个线程负责Tm * Tn 个输出。
    constexpr unsigned int Tm = BM / C_blockDim_y;
    constexpr unsigned int Tn = BN / C_blockDim_x;
    float C_T[Tm][Tn] = {0.0f};

    // K loop
    for (int k = 0u; k < K;k += BK)
    {
        /*
            完成 tileA = (BM, K) @ tileB(K * BN)的计算
        */
       for (unsigned int i = )
       {
            /*
                完成
            */
       }
    }
}