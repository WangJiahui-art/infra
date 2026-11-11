/**
 * File : Attention.cu
 * 功能 : 实现Decode阶段的一个Q的Attention运算.
 * S = softmax((Q @ K^T) / sqrt(d))
 * 单个warp, 单个batch, 单个head
 * head_dim == 64
 *
 */

#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void Attention(const float __restrict__ *Q, const float __restrict__ *K, const float __restrict__ *V, int seq_len)