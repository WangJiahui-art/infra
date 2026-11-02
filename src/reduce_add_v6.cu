/*
通过warp内shuffle,快速的求出单个warp的和, 然后在做二次聚合.
*/


__device__ float warp_reduce_sum(float val){
    /* 实现warp内shuffle */
    for (unsigned int delta = 16; delta > 0u; delta >>= 1){
        val += __shfl_down_sync(0xffffffff, val, delta);
    }
    return val;
}

__global__ void reduce(float* input, float* output, int n){
    // 索引
    unsigned int tid = threadIdx.x;
    unsigned int gid = threadIdx.x + blockDim.x * blockIdx.x * 2;
    unsigned int wid = tid / 32;
    unsigned int lane_id = tid % 32;
    
    // 数据读取,每线程读2个
    float val = 0.0f;
    if (gid < n) {
        val += input[gid];
    }
    if ((gid + blockDim.x) < n) {
        val += input[gid + blockDim.x];
    }

    // warp内规约
    val = warp_reduce_sum(val);

    // 将lane0的数据读到shared memory
    __shared__ float warp_sum[32];
    if (lane_id == 0)
    {
        warp_sum[wid] = val;
    }
    __syncthreads();

    // 在warp0中对warp_sum做规约
    unsigned int num = (blockDim.x + 31) / 32;
    if (wid == 0)
    {
        val = (lane_id < num) ? warp_sum[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
    }

    // 将warp0的lane0的结果写到HBM
    if (tid == 0) {
        output[blockIdx.x] = val;
    }
}