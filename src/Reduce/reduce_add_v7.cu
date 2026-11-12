# include <cstdio>
# include <cuda_runtime.h>


__global__ void reduce(float* input, float* output, int n) {
    /* 向量化加载 + grid_stride_loop */
    
    // 索引计算
    unsigned int tid = threadIdx.x;
    unsigned int lid = tid % 32;
    unsigned int wid = tid / 32;
    unsigned int gid = tid + blockIdx.x * blockDim.x;

    // 向量化加载
    const float4 *input4 = reinterpret_cast<float4 *>(input);
    unsigned int n4 = n / 4;
    float val = 0.0f;
    for (unsigned int idx = gid; idx < n4; idx += gridDim.x * blockDim.x)
    {
        float4 data = input4[idx];
        val += data.x + data.w + data.y + data.z;
    }

    // 处理尾部数据
    
    if (gid < (n - n4 * 4)) {
        // 用前面几个线程来处理
        val += input[n4 * 4 + gid];
    }

    // 进行warp内规约
    for (unsigned int step = 16; step > 0; step >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, step);
    }

    // 将每一个warp的lane0 的结果放到共享内存中.
    __shared__ float warp_result[32];
    if (lid == 0) {
        warp_result[wid] = val;
    }
    __syncthreads();

    // 二次加载
    unsigned int num = (blockDim.x + 31) / 32; // 计算有实际意义的下标范围
    if (wid == 0)
    {
        val = (lid < num) ? warp_result[lid] : 0.0f;
        // shuffle
        for (unsigned int delta = 16; delta > 0; delta >>= 1)
        {
            val += __shfl_down_sync(0xffffffff, val, delta);
        }
    }



    // 每一个block的求和结果在thread0上.

    // 写回HBM
    if (tid == 0)
    {
        output[blockIdx.x] = val;
    }
}