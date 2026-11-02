# include <cstdio>
# include <cuda_runtime.h>

/*
在v0版本中,只有偶数索引的 线程参与运算,奇数线程都在等待,导致warp divergence.

那么第一个改进思路如下:
在一个block中, 前1/2线程参与计算,后1/2闲置. 而后 1/4参与计算,3/4闲置,如此递归.
strided_index方式
假定blockDim.x = 256;
通过idx = threadIdx.x * step * 2 的方式,将前128个threadIdx.x映射到0 , 2, 4...
然后递归映射
*/

__global__ void reduce(float* input, float* output, int n){
    //计算索引
    unsigned int tid = threadIdx.x;
    unsigned int gid = threadIdx.x + blockDim.x * blockIdx.x;

    //从HBM读取数据到 shared memory
    extern __shared__ float smem[];
    smem[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2){
        unsigned int index = tid * stride * 2;
        if(index + stride < blockDim.x){    //! 这里要保证index + stride 不越界.因为如果blockDim.x不是2的幂次,就会产生脏读.
            smem[index] += smem[index + stride];
        }
        __syncthreads();
    }

    // 写入HBM
    if (tid == 0u){
        output[blockIdx.x] = smem[0];
    }
}