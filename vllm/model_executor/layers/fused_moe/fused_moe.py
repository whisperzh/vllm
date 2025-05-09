# SPDX-License-Identifier: Apache-2.0
"""Fused MoE kernel."""
import functools
import importlib.util
import json
import os
from math import prod
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import defaultdict

import torch
import triton
import triton.language as tl

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8)
from vllm.platforms import current_platform
from vllm.utils import direct_register_custom_op, round_up
from vllm.model_executor.layers.fused_moe.eplb import rebalance_experts

from .rocm_aiter_fused_moe import (is_rocm_aiter_moe_enabled,
                                   rocm_aiter_fused_experts,
                                   rocm_aiter_topk_softmax)

logger = init_logger(__name__)

has_deep_gemm = importlib.util.find_spec("deep_gemm") is not None


@triton.jit
def write_zeros_to_output(c_ptr, stride_cm, stride_cn, pid_n, N, offs_token,
                          token_mask, BLOCK_SIZE_M, BLOCK_SIZE_N,
                          compute_type):
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[
        None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def fused_moe_kernel_gptq_awq(
        # Pointers to matrices
        a_ptr,
        b_ptr,
        c_ptr,
        b_scale_ptr,
        b_zp_ptr,
        topk_weights_ptr,
        sorted_token_ids_ptr,
        expert_ids_ptr,
        num_tokens_post_padded_ptr,
        # Matrix dimensions
        N: tl.constexpr,
        K: tl.constexpr,
        EM,
        num_valid_tokens,
        # The stride variables represent how much to increase the ptr by when
        # moving by 1 element in a particular dimension. E.g. `stride_am` is
        # how much to increase `a_ptr` by to get the element one row down
        # (A has M rows).
        stride_am,
        stride_ak,
        stride_be,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        stride_bse,
        stride_bsk,
        stride_bsn,
        stride_bze,
        stride_bzk,
        stride_bzn,
        block_k_diviable: tl.constexpr,
        group_size: tl.constexpr,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        MUL_ROUTED_WEIGHT: tl.constexpr,
        top_k: tl.constexpr,
        compute_type: tl.constexpr,
        has_zp: tl.constexpr,
        use_int4_w4a16: tl.constexpr,
        use_int8_w8a16: tl.constexpr):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(
        tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        # -----------------------------------------------------------
        # Write back zeros to the output when the expert is not
        # in the current expert parallel rank.
        write_zeros_to_output(c_ptr, stride_cm, stride_cn, pid_n, N,
                              offs_token, token_mask, BLOCK_SIZE_M,
                              BLOCK_SIZE_N, compute_type)
        return

    offs_bn = (pid_n * BLOCK_SIZE_N +
               tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am +
                      offs_k[None, :] * stride_ak)

    if use_int4_w4a16:
        b_ptrs = b_ptr + off_experts * stride_be + \
            (offs_k[:, None] // 2) * stride_bk + offs_bn[None, :] * \
                stride_bn
        b_shifter = (offs_k[:, None] % 2) * 4
    elif use_int8_w8a16:
        b_ptrs = b_ptr + off_experts * stride_be + \
            offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    if not has_zp and use_int4_w4a16:
        b_zp_num = 8
    if not has_zp and use_int8_w8a16:
        b_zp_num = 128
    elif has_zp and use_int4_w4a16:
        b_zp_shifter = (offs_bn[None, :] % 2) * 4

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.

        if not block_k_diviable:
            k_mask = offs_k[:, None] < K - k * BLOCK_SIZE_K
            k_other = 0.0
        else:
            k_mask = None
            k_other = None

        a = tl.load(a_ptrs,
                    mask=token_mask[:, None] &
                    (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                    other=0.0)
        b = tl.load(b_ptrs)
        if use_int4_w4a16:
            b = (b >> b_shifter) & 0xF

        b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + \
            offs_bn[None, :] * stride_bsn + \
            ((offs_k[:, None] + BLOCK_SIZE_K * k) // group_size) * \
                stride_bsk
        b_scale = tl.load(b_scale_ptrs, mask=k_mask, other=k_other)
        b_scale = b_scale.to(tl.float32)

        if has_zp and use_int4_w4a16:
            offs_k_true = (offs_k[:, None] + BLOCK_SIZE_K * k) // group_size
            b_zp_ptrs = b_zp_ptr + off_experts * stride_bze + \
                (offs_bn[None, :] // 2) * stride_bzn + \
                offs_k_true * stride_bzk
            b_zp = tl.load(b_zp_ptrs, mask=k_mask, other=k_other)
            b_zp = ((b_zp >> b_zp_shifter) & 0xF)
            b_zp = b_zp.to(tl.float32)
        elif has_zp and use_int8_w8a16:
            offs_k_true = (offs_k[:, None] + BLOCK_SIZE_K * k) // group_size
            b_zp_ptrs = b_zp_ptr + off_experts * stride_bze + \
                offs_bn[None, :] * stride_bzn + \
                offs_k_true * stride_bzk
            b_zp = tl.load(b_zp_ptrs, mask=k_mask, other=k_other)
            b_zp = b_zp.to(tl.float32)

        # We accumulate along the K dimension.
        if has_zp:
            b = ((b.to(tl.float32) - b_zp) * b_scale).to(compute_type)
        else:
            b = ((b.to(tl.float32) - b_zp_num) * b_scale).to(compute_type)
        accumulator = tl.dot(a, b, acc=accumulator)

        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        if use_int4_w4a16:
            b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk
        else:
            b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token,
                             mask=token_mask,
                             other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[
        None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def fused_moe_kernel(
        # Pointers to matrices
        a_ptr,
        b_ptr,
        c_ptr,
        a_scale_ptr,
        b_scale_ptr,
        topk_weights_ptr,
        sorted_token_ids_ptr,
        expert_ids_ptr,
        num_tokens_post_padded_ptr,
        # Matrix dimensions
        N,
        K,
        EM,
        num_valid_tokens,
        # The stride variables represent how much to increase the ptr by when
        # moving by 1 element in a particular dimension. E.g. `stride_am` is
        # how much to increase `a_ptr` by to get the element one row down
        # (A has M rows).
        stride_am,
        stride_ak,
        stride_be,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        stride_asm,
        stride_ask,
        stride_bse,
        stride_bsk,
        stride_bsn,
        # Block size for block-wise quantization
        group_n: tl.constexpr,
        group_k: tl.constexpr,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        MUL_ROUTED_WEIGHT: tl.constexpr,
        top_k: tl.constexpr,
        compute_type: tl.constexpr,
        use_fp8_w8a8: tl.constexpr,
        use_int8_w8a16: tl.constexpr):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(
        tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        # -----------------------------------------------------------
        # Write back zeros to the output when the expert is not
        # in the current expert parallel rank.
        write_zeros_to_output(c_ptr, stride_cm, stride_cn, pid_n, N,
                              offs_token, token_mask, BLOCK_SIZE_M,
                              BLOCK_SIZE_N, compute_type)
        return

    offs_bn = (pid_n * BLOCK_SIZE_N +
               tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am +
                      offs_k[None, :] * stride_ak)

    b_ptrs = b_ptr + off_experts * stride_be + (offs_k[:, None] * stride_bk +
                                                offs_bn[None, :] * stride_bn)
    if use_int8_w8a16:
        b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + offs_bn[
            None, :] * stride_bsn
        b_scale = tl.load(b_scale_ptrs)

    if use_fp8_w8a8:
        if group_k > 0 and group_n > 0:
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            offs_bsn = offs_bn // group_n
            b_scale_ptrs = (b_scale_ptr + off_experts * stride_bse +
                            offs_bsn * stride_bsn)
        else:
            a_scale = tl.load(a_scale_ptr)
            b_scale = tl.load(b_scale_ptr + off_experts)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        a = tl.load(a_ptrs,
                    mask=token_mask[:, None] &
                    (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                    other=0.0)
        b = tl.load(b_ptrs,
                    mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                    other=0.0)
        # We accumulate along the K dimension.
        if use_int8_w8a16:
            accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
        elif use_fp8_w8a8:
            if group_k > 0 and group_n > 0:
                k_start = k * BLOCK_SIZE_K
                offs_ks = k_start // group_k
                a_scale = tl.load(a_scale_ptrs + offs_ks * stride_ask,
                                  mask=token_mask,
                                  other=0.0)
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)

                accumulator += tl.dot(a, b) * a_scale[:,
                                                      None] * b_scale[None, :]
            else:
                accumulator = tl.dot(a, b, acc=accumulator)
        else:
            accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token,
                             mask=token_mask,
                             other=0)
        accumulator = accumulator * moe_weight[:, None]
    if use_int8_w8a16:
        accumulator = (accumulator * b_scale).to(compute_type)
    elif use_fp8_w8a8:
        if group_k > 0 and group_n > 0:
            accumulator = accumulator.to(compute_type)
        else:
            accumulator = (accumulator * a_scale * b_scale).to(compute_type)
    else:
        accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[
        None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def ceil_div(a, b):
    return (a + b - 1) // b


@triton.jit
def moe_align_block_size_stage1(
    topk_ids_ptr,
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
    numel: tl.constexpr,
    tokens_per_thread: tl.constexpr,
):
    pid = tl.program_id(0)

    start_idx = pid * tokens_per_thread

    off_c = (pid + 1) * num_experts

    for i in range(tokens_per_thread):
        if start_idx + i < numel:
            idx = tl.load(topk_ids_ptr + start_idx + i)
            token_cnt = tl.load(tokens_cnts_ptr + off_c + idx)
            tl.store(tokens_cnts_ptr + off_c + idx, token_cnt + 1)


@triton.jit
def moe_align_block_size_stage2(
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
):
    pid = tl.program_id(0)

    last_cnt = 0
    for i in range(1, num_experts + 1):
        token_cnt = tl.load(tokens_cnts_ptr + i * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + i * num_experts + pid, last_cnt)


@triton.jit
def moe_align_block_size_stage3(
    total_tokens_post_pad_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
):
    last_cumsum = 0
    off_cnt = num_experts * num_experts
    for i in range(1, num_experts + 1):
        token_cnt = tl.load(tokens_cnts_ptr + off_cnt + i - 1)
        last_cumsum = last_cumsum + tl.cdiv(token_cnt, block_size) * block_size
        tl.store(cumsum_ptr + i, last_cumsum)
    tl.store(total_tokens_post_pad_ptr, last_cumsum)


@triton.jit
def moe_align_block_size_stage4(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
    numel: tl.constexpr,
    tokens_per_thread: tl.constexpr,
):
    pid = tl.program_id(0)
    start_idx = tl.load(cumsum_ptr + pid)
    end_idx = tl.load(cumsum_ptr + pid + 1)

    for i in range(start_idx, end_idx, block_size):
        tl.store(expert_ids_ptr + i // block_size, pid)

    start_idx = pid * tokens_per_thread
    off_t = pid * num_experts

    for i in range(start_idx, tl.minimum(start_idx + tokens_per_thread,
                                         numel)):
        expert_id = tl.load(topk_ids_ptr + i)
        token_cnt = tl.load(tokens_cnts_ptr + off_t + expert_id)
        rank_post_pad = token_cnt + tl.load(cumsum_ptr + expert_id)
        tl.store(sorted_token_ids_ptr + rank_post_pad, i)
        tl.store(tokens_cnts_ptr + off_t + expert_id, token_cnt + 1)


# @triton.jit
# def EXP_load_balaning_expert_ids(expert_ids_ptr,
#                              output_expert_ids_ptr,
#                             threshold,
#                             num_experts: tl.constexpr):
   



# Triton implementation based on:
# https://github.com/sgl-project/sglang/commit/ba5112ff691d791a9e38c6c71f59324a5fcb49d0
def moe_align_block_size_triton(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
) -> None:
    numel = topk_ids.numel()
    grid = (num_experts, )
    tokens_cnts = torch.zeros((num_experts + 1, num_experts),
                              dtype=torch.int32,
                              device=topk_ids.device)
    cumsum = torch.zeros((num_experts + 1, ),
                         dtype=torch.int32,
                         device=topk_ids.device)
    tokens_per_thread = ceil_div(numel, num_experts)

    moe_align_block_size_stage1[grid](
        topk_ids,
        tokens_cnts,
        num_experts,
        numel,
        tokens_per_thread,
    )
    moe_align_block_size_stage2[grid](
        tokens_cnts,
        num_experts,
    )
    moe_align_block_size_stage3[(1, )](
        num_tokens_post_pad,
        tokens_cnts,
        cumsum,
        num_experts,
        block_size,
    )
    moe_align_block_size_stage4[grid](
        topk_ids,
        sorted_token_ids,
        expert_ids,
        tokens_cnts,
        cumsum,
        num_experts,
        block_size,
        numel,
        tokens_per_thread,
    )


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: Optional[torch.Tensor] = None,
    pad_sorted_ids: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns the token distribution across experts to be compatible with block
    size for matrix multiplication.

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the
        top-k expert indices for each token.
    - block_size: The block size used in block matrix multiplication.
    - num_experts: The total number of experts.
    - expert_map: A tensor of shape [num_experts] that maps the expert index
        from the global space to the local index space of the current
        expert parallel shard. If the expert is not in the current expert
        parallel shard, the mapping is set to -1.
    - pad_sorted_ids: A flag indicating whether the sorted_token_ids length
      should be padded to a multiple of block_size,

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according
        to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding,
        ensuring divisibility by block_size.

    This function pads the number of tokens that each expert needs to process
    so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions
    align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]],
    block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts,
        with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids
        [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in
        the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible
        by block_size for proper block matrix operations.
    """
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    sorted_ids = torch.empty((max_num_tokens_padded, ),
                             dtype=torch.int32,
                             device=topk_ids.device)
    sorted_ids.fill_(topk_ids.numel())
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    # Expert ids must be zeroed out to prevent index out of bounds error while
    # mapping global expert ids to local expert ids in expert parallelism.
    expert_ids = torch.zeros((max_num_m_blocks, ),
                             dtype=torch.int32,
                             device=topk_ids.device)
    num_tokens_post_pad = torch.empty((1),
                                      dtype=torch.int32,
                                      device=topk_ids.device)
    if num_experts >= 224:
        if envs.VLLM_ENABLE_MOE_ALIGN_BLOCK_SIZE_TRITON or num_experts != 256:
            moe_align_block_size_triton(
                topk_ids,
                num_experts,
                block_size,
                sorted_ids,
                expert_ids,
                num_tokens_post_pad,
            )
        else:
            # Currently requires num_experts=256
            ops.sgl_moe_align_block_size(
                topk_ids,
                num_experts,
                block_size,
                sorted_ids,
                expert_ids,
                num_tokens_post_pad,
            )
    else:
        ops.moe_align_block_size(topk_ids, num_experts, block_size, sorted_ids,
                                 expert_ids, num_tokens_post_pad)
    if expert_map is not None:
        expert_ids = expert_map[expert_ids]

    return sorted_ids, expert_ids, num_tokens_post_pad


def _valid_deep_gemm(hidden_states: torch.Tensor, w1: torch.Tensor,
                     w2: torch.Tensor,
                     expert_map: Optional[torch.Tensor]) -> bool:
    """
    Check if the given problem size is supported by the DeepGemm grouped
    gemm kernel.  All of M, N, K and the quantization block_shape must be
    aligned by `dg.get_m_alignment_for_contiguous_layout()`.
    """
    if not has_deep_gemm:
        return False

    # Lazy import to avoid CUDA initialization problems.
    import deep_gemm as dg

    # Expert maps not supported yet.
    if expert_map is not None:
        return False

    align = dg.get_m_alignment_for_contiguous_layout()
    M = hidden_states.shape[0]
    _, K, N = w2.shape

    # For now, disable DeepGemm for small N until better permute/unpermute
    # ops are available.
    if N <= 512:
        return False

    if align > M or N % align != 0 or K % align != 0:
        return False

    return (hidden_states.is_contiguous() and w1.is_contiguous()
            and w2.is_contiguous())


def _fp8_quantize(
    A: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    block_shape: Optional[List[int]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform fp8 quantization on the inputs.  If a block_shape
    is provided, the output will be blocked.
    """
    if block_shape is None:
        A, A_scale = ops.scaled_fp8_quant(A, A_scale)
    else:
        assert len(block_shape) == 2
        _, block_k = block_shape[0], block_shape[1]
        A, A_scale = per_token_group_quant_fp8(A, block_k)
        assert triton.cdiv(A.shape[-1], block_k) == A_scale.shape[-1]
    return A, A_scale


def invoke_fused_moe_kernel(A: torch.Tensor,
                            B: torch.Tensor,
                            C: torch.Tensor,
                            A_scale: Optional[torch.Tensor],
                            B_scale: Optional[torch.Tensor],
                            B_zp: Optional[torch.Tensor],
                            topk_weights: torch.Tensor,
                            topk_ids: torch.Tensor,
                            sorted_token_ids: torch.Tensor,
                            expert_ids: torch.Tensor,
                            num_tokens_post_padded: torch.Tensor,
                            mul_routed_weight: bool,
                            top_k: int,
                            config: Dict[str, Any],
                            compute_type: tl.dtype,
                            use_fp8_w8a8: bool,
                            use_int8_w8a16: bool,
                            use_int4_w4a16: bool,
                            block_shape: Optional[List[int]] = None) -> None:
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1

    if use_fp8_w8a8:
        assert B_scale is not None
        assert (block_shape is None or triton.cdiv(B.shape[-2], block_shape[0])
                == B_scale.shape[-2])
        assert (block_shape is None or triton.cdiv(B.shape[-1], block_shape[1])
                == B_scale.shape[-1])

    elif use_int8_w8a16 or use_int4_w4a16:
        assert B_scale is not None
        assert block_shape is None or block_shape[0] == 0
    else:
        assert A_scale is None
        assert B_scale is None

    EM = sorted_token_ids.shape[0]
    if A.shape[0] < config["BLOCK_SIZE_M"]:
        # optimize for small batch_size.
        # We assume that top_ids of each token is unique, so
        # so num_valid_experts <= batch_size <= BLOCK_SIZE_M,
        # and we can skip some invalid blocks.
        EM = min(sorted_token_ids.shape[0],
                 A.shape[0] * top_k * config['BLOCK_SIZE_M'])
    grid = lambda META: (triton.cdiv(EM, META['BLOCK_SIZE_M']) * triton.cdiv(
        B.shape[1], META['BLOCK_SIZE_N']), )

    if (use_int8_w8a16 or use_int4_w4a16) and \
            block_shape is not None and block_shape[1] > 0:
        assert B_scale is not None and B_scale.ndim == 3
        assert B_zp is None or B_zp.ndim == 3

        use_moe_wna16_cuda = should_moe_wna16_use_cuda(
            num_valid_tokens=topk_ids.numel(),
            group_size=block_shape[1],
            num_experts=B.shape[0],
            bit=4 if use_int4_w4a16 else 8)
        config = config.copy()
        config.update(
            get_moe_wna16_block_config(config=config,
                                       use_moe_wna16_cuda=use_moe_wna16_cuda,
                                       num_valid_tokens=topk_ids.numel(),
                                       size_k=A.shape[1],
                                       size_n=B.shape[1],
                                       num_experts=B.shape[1],
                                       group_size=block_shape[1],
                                       real_top_k=topk_ids.shape[1],
                                       block_size_m=config["BLOCK_SIZE_M"]))

        if use_moe_wna16_cuda:
            bit = 4 if use_int4_w4a16 else 8
            ops.moe_wna16_gemm(A, C, B, B_scale, B_zp,
                               topk_weights if mul_routed_weight else None,
                               sorted_token_ids, expert_ids,
                               num_tokens_post_padded, top_k,
                               config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"],
                               config["BLOCK_SIZE_K"], bit)
            return

        fused_moe_kernel_gptq_awq[grid](
            A,
            B,
            C,
            B_scale,
            B_zp,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            B.shape[1],
            A.shape[1],
            EM,
            topk_ids.numel(),
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(2),
            B.stride(1),
            C.stride(1),
            C.stride(2),
            B_scale.stride(0),
            B_scale.stride(2),
            B_scale.stride(1),
            B_zp.stride(0) if B_zp is not None else 0,
            B_zp.stride(2) if B_zp is not None else 0,
            B_zp.stride(1) if B_zp is not None else 0,
            block_k_diviable=A.shape[1] % config["BLOCK_SIZE_K"] == 0,
            group_size=block_shape[1],
            MUL_ROUTED_WEIGHT=mul_routed_weight,
            top_k=top_k,
            compute_type=compute_type,
            has_zp=B_zp is not None,
            use_int4_w4a16=use_int4_w4a16,
            use_int8_w8a16=use_int8_w8a16,
            **config,
        )
    else:
        config = config.copy()
        BLOCK_SIZE_K = config.pop("BLOCK_SIZE_K")
        if block_shape is not None:
            BLOCK_SIZE_K = min(BLOCK_SIZE_K, min(block_shape[0],
                                                 block_shape[1]))
        fused_moe_kernel[grid](
            A,
            B,
            C,
            A_scale,
            B_scale,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            B.shape[1],
            B.shape[2],
            EM,
            topk_ids.numel(),
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(2),
            B.stride(1),
            C.stride(1),
            C.stride(2),
            A_scale.stride(0)
            if A_scale is not None and A_scale.ndim == 2 else 0,
            A_scale.stride(1)
            if A_scale is not None and A_scale.ndim == 2 else 0,
            B_scale.stride(0)
            if B_scale is not None and B_scale.ndim >= 2 else 0,
            B_scale.stride(2)
            if B_scale is not None and B_scale.ndim == 3 else 0,
            B_scale.stride(1)
            if B_scale is not None and B_scale.ndim >= 2 else 0,
            0 if block_shape is None else block_shape[0],
            0 if block_shape is None else block_shape[1],
            MUL_ROUTED_WEIGHT=mul_routed_weight,
            top_k=top_k,
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            **config,
        )


# Adapted from: https://github.com/sgl-project/sglang/pull/2628
def get_config_file_name(E: int,
                         N: int,
                         dtype: Optional[str],
                         block_shape: Optional[List[int]] = None) -> str:
    device_name = current_platform.get_device_name().replace(" ", "_")
    dtype_selector = "" if not dtype else f",dtype={dtype}"
    block_shape_selector = ("" if not block_shape or not all(block_shape) else
                            f",block_shape={block_shape}").replace(" ", "")
    return f"E={E},N={N},device_name={device_name}{dtype_selector}{block_shape_selector}.json"  # noqa: E501


# Adapted from: https://github.com/sgl-project/sglang/pull/2628
@functools.lru_cache
def get_moe_configs(
    E: int,
    N: int,
    dtype: Optional[str],
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
) -> Optional[Dict[int, Any]]:
    """
    Return optimized configurations for the fused MoE kernel.

    The return value will be a dictionary that maps an irregular grid of
    batch sizes to configurations of the fused_moe kernel. To evaluate the
    kernel on a given batch size bs, the closest batch size in the grid should
    be picked and the associated configuration chosen to invoke the kernel.
    """

    # First look up if an optimized configuration is available in the configs
    # directory
    block_shape = [block_n, block_k] if block_n and block_k else None
    json_file_name = get_config_file_name(E, N, dtype, block_shape)

    config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs", json_file_name)
    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            logger.info("Using configuration from %s for MoE layer.",
                        config_file_path)
            # If a configuration has been found, return it
            return {int(key): val for key, val in json.load(f).items()}

    # If no optimized configuration is available, we will use the default
    # configuration
    logger.warning(
        ("Using default MoE config. Performance might be sub-optimal! "
         "Config file not found at %s"), config_file_path)
    return None


def get_moe_wna16_block_config(config: Dict[str,
                                            int], use_moe_wna16_cuda: bool,
                               num_valid_tokens: int, size_k: int, size_n: int,
                               num_experts: int, group_size: int,
                               real_top_k: int, block_size_m: int):
    if "BLOCK_SIZE_N" in config and "BLOCK_SIZE_K" in config:
        # optimal block config is set
        return {}
    if not use_moe_wna16_cuda:
        # triton moe wna16 kernel
        if num_valid_tokens // real_top_k == 1:
            # if bs=1, use a smaller BLOCK_SIZE_N
            return {"BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64}
        else:
            return {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}
    else:
        # cuda moe wna16 kernel
        # set default block_size 128, and increase them when num_blocks
        # is too large.
        block_size_n = 128
        block_size_k = 128
        if block_size_k <= group_size:
            block_size_k = group_size

        num_n_blocks = size_k // block_size_k
        num_k_blocks = size_n // block_size_k
        num_m_blocks = (num_valid_tokens + block_size_m - 1) / block_size_m + \
            num_experts
        if num_valid_tokens // real_top_k <= block_size_m:
            num_m_blocks = min(num_m_blocks, num_valid_tokens)
        num_blocks = num_m_blocks * num_n_blocks * num_k_blocks

        if size_k % 256 == 0 and num_blocks >= 256 and \
                block_size_k < 256:
            block_size_k = 256
            num_blocks = num_blocks // (256 // block_size_k)

        if num_m_blocks <= 16 and size_k % (block_size_k * 2) == 0 and \
                size_k % (block_size_k * 2) == 0 and block_size_k <= 512 and \
                num_blocks >= 512:
            block_size_k = block_size_k * 2
            num_blocks = num_blocks // 2

        if num_blocks > 1024:
            block_size_n = 256
            num_n_blocks = num_n_blocks // 2
            num_blocks = num_blocks // 2

        if size_n <= 1024 and num_blocks >= 1024:
            # The kernel performance got much better with BLOCK_SIZE_N=1024
            # when num_blocks is large, event when N is small.
            # Not sure why, maybe it force the CUDA SM process only one block
            # at the same time.
            block_size_n = 1024

        return {"BLOCK_SIZE_N": block_size_n, "BLOCK_SIZE_K": block_size_k}


def should_moe_wna16_use_cuda(num_valid_tokens: int, group_size: int,
                              num_experts: int, bit: int):
    return bit == 4 and group_size in [32, 64, 128] and \
        num_valid_tokens / num_experts <= 6


def get_default_config(
    M: int,
    E: int,
    N: int,
    K: int,
    topk: int,
    dtype: Optional[str],
    is_marlin: bool,
    block_shape: Optional[List[int]] = None,
) -> Dict[str, int]:
    if dtype == "fp8_w8a8" and block_shape is not None:
        # Block-wise quant: BLOCK_SIZE_N must be divisible by block_shape[0]
        # BLOCK_SIZE_K must be divisible by block_shape[1]
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": block_shape[0],
            "BLOCK_SIZE_K": block_shape[1],
            "GROUP_SIZE_M": 32,
            "num_warps": 4,
            "num_stages": 3,
        }
    elif dtype in ["int4_w4a16", "int8_w8a16"] and block_shape is not None:
        # moe wna16 kernels
        # only set BLOCK_SIZE_M
        # BLOCK_SIZE_N and BLOCK_SIZE_K would be set later
        bit = 4 if dtype == "int4_w4a16" else 8
        use_moe_wna16_cuda = should_moe_wna16_use_cuda(M * topk,
                                                       block_shape[1], E, bit)
        if use_moe_wna16_cuda:
            config = {"BLOCK_SIZE_M": min(16, M)}
        elif M <= 20:
            config = {"BLOCK_SIZE_M": 16, "GROUP_SIZE_M": 1}
        elif M <= 40:
            config = {"BLOCK_SIZE_M": 32, "GROUP_SIZE_M": 1}
        else:
            config = {"BLOCK_SIZE_M": 64, "GROUP_SIZE_M": 1}
    else:
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
        }
        # A heuristic: fused marlin works faster with this config for small M
        if M <= E or (is_marlin and M <= 32):
            config = {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1,
            }
    return config


def try_get_optimal_moe_config(
    w1_shape: Tuple[int, ...],
    w2_shape: Tuple[int, ...],
    top_k: int,
    dtype: Optional[str],
    M: int,
    is_marlin: bool = False,
    block_shape: Optional[List[int]] = None,
):
    from vllm.model_executor.layers.fused_moe import get_config
    override_config = get_config()
    if override_config:
        config = override_config
    else:
        # First try to load optimal config from the file
        E, _, N = w2_shape
        if dtype == "int4_w4a16":
            N = N * 2
        block_n = block_shape[0] if block_shape else 0
        block_k = block_shape[1] if block_shape else 0
        configs = get_moe_configs(E, N, dtype, block_n, block_k)

        if configs:
            # If an optimal configuration map has been found, look up the
            # optimal config
            config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
        else:
            # Else use the default config
            config = get_default_config(M, E, N, w1_shape[2], top_k, dtype,
                                        is_marlin, block_shape)
    return config


def vllm_topk_softmax(topk_weights: torch.Tensor, topk_indices: torch.Tensor,
                      token_expert_indices: torch.Tensor,
                      gating_output: torch.Tensor,
                      renormalize: bool) -> tuple[torch.Tensor, ...]:
    ops.topk_softmax(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
    )
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_indices


def dispatch_topk_func() -> Callable[..., tuple[torch.Tensor, ...]]:
    if is_rocm_aiter_moe_enabled():
        return rocm_aiter_topk_softmax
    return vllm_topk_softmax


def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert hidden_states.shape[0] == gating_output.shape[0], (
        "Number of tokens mismatch")

    M, _ = hidden_states.shape

    topk_weights = torch.empty(M,
                               topk,
                               dtype=torch.float32,
                               device=hidden_states.device)
    topk_ids = torch.empty(M,
                           topk,
                           dtype=torch.int32,
                           device=hidden_states.device)
    token_expert_indicies = torch.empty(M,
                                        topk,
                                        dtype=torch.int32,
                                        device=hidden_states.device)

    gating_output_float = gating_output.float()  # TODO(woosuk): Optimize this.

    topk_func = dispatch_topk_func()
    topk_weights, topk_ids = topk_func(topk_weights, topk_ids,
                                       token_expert_indicies,
                                       gating_output_float, renormalize)

    del token_expert_indicies  # Not used. Will be used in the future.
    return topk_weights, topk_ids


# This is used by the Deepseek-V2 and Deepseek-V3 model
@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
def grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
    e_score_correction_bias: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:

    assert hidden_states.shape[0] == gating_output.shape[0], (
        "Number of tokens mismatch")

    if scoring_func == "softmax":
        scores = torch.softmax(gating_output, dim=-1)
    elif scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    num_token = scores.shape[0]
    if e_score_correction_bias is not None:
        # Store original scores before applying correction bias. We use biased
        # scores for expert selection but original scores for routing weights
        original_scores = scores
        scores = scores + e_score_correction_bias.unsqueeze(0)
        group_scores = (scores.view(num_token, num_expert_group,
                                    -1).topk(2, dim=-1)[0].sum(dim=-1))
    else:
        group_scores = scores.view(num_token, num_expert_group,
                                   -1).max(dim=-1).values  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1,
                           sorted=False)[1]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = group_mask.unsqueeze(-1).expand(
        num_token, num_expert_group,
        scores.shape[-1] // num_expert_group).reshape(num_token, -1)  # [n, e]
    tmp_scores = scores.masked_fill(~score_mask.bool(),
                                    float("-inf"))  # [n, e]

    if e_score_correction_bias is not None:
        topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)[1]
        # Use original unbiased scores for the routing weights
        topk_weights = original_scores.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(tmp_scores,
                                            k=topk,
                                            dim=-1,
                                            sorted=False)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def get_config_dtype_str(
        dtype: torch.dtype,
        use_int4_w4a16: Optional[bool] = False,
        use_int8_w8a16: Optional[bool] = False,
        use_fp8_w8a8: Optional[bool] = False) -> Optional[str]:
    if use_fp8_w8a8:
        return "fp8_w8a8"
    elif use_int8_w8a16:
        return "int8_w8a16"
    elif use_int4_w4a16:
        return "int4_w4a16"
    elif dtype == torch.float:
        # avoiding cases where kernel fails when float32 MoE
        # use fp16/bfloat16 configs
        return "float32"
    return None


def inplace_fused_experts(hidden_states: torch.Tensor,
                          w1: torch.Tensor,
                          w2: torch.Tensor,
                          topk_weights: torch.Tensor,
                          topk_ids: torch.Tensor,
                          activation: str = "silu",
                          use_fp8_w8a8: bool = False,
                          use_int8_w8a16: bool = False,
                          use_int4_w4a16: bool = False,
                          global_num_experts: int = -1,
                          expert_map: Optional[torch.Tensor] = None,
                          w1_scale: Optional[torch.Tensor] = None,
                          w2_scale: Optional[torch.Tensor] = None,
                          w1_zp: Optional[torch.Tensor] = None,
                          w2_zp: Optional[torch.Tensor] = None,
                          a1_scale: Optional[torch.Tensor] = None,
                          a2_scale: Optional[torch.Tensor] = None,
                          block_shape: Optional[List[int]] = None) -> None:
    fused_experts_impl(hidden_states, w1, w2, topk_weights, topk_ids, True,
                       activation, use_fp8_w8a8, use_int8_w8a16,
                       use_int4_w4a16, global_num_experts, expert_map,
                       w1_scale, w2_scale, w1_zp, w2_zp, a1_scale, a2_scale,
                       block_shape)


def inplace_fused_experts_fake(
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str = "silu",
        use_fp8_w8a8: bool = False,
        use_int8_w8a16: bool = False,
        use_int4_w4a16: bool = False,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        w1_scale: Optional[torch.Tensor] = None,
        w2_scale: Optional[torch.Tensor] = None,
        w1_zp: Optional[torch.Tensor] = None,
        w2_zp: Optional[torch.Tensor] = None,
        a1_scale: Optional[torch.Tensor] = None,
        a2_scale: Optional[torch.Tensor] = None,
        block_shape: Optional[List[int]] = None) -> None:
    pass


direct_register_custom_op(
    op_name="inplace_fused_experts",
    op_func=inplace_fused_experts,
    mutates_args=["hidden_states"],
    fake_impl=inplace_fused_experts_fake,
)


def outplace_fused_experts(
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str = "silu",
        use_fp8_w8a8: bool = False,
        use_int8_w8a16: bool = False,
        use_int4_w4a16: bool = False,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        w1_scale: Optional[torch.Tensor] = None,
        w2_scale: Optional[torch.Tensor] = None,
        w1_zp: Optional[torch.Tensor] = None,
        w2_zp: Optional[torch.Tensor] = None,
        a1_scale: Optional[torch.Tensor] = None,
        a2_scale: Optional[torch.Tensor] = None,
        block_shape: Optional[List[int]] = None) -> torch.Tensor:
    return fused_experts_impl(hidden_states, w1, w2, topk_weights, topk_ids,
                              False, activation, use_fp8_w8a8, use_int8_w8a16,
                              use_int4_w4a16, global_num_experts, expert_map,
                              w1_scale, w2_scale, w1_zp, w2_zp, a1_scale,
                              a2_scale, block_shape)


def outplace_fused_experts_fake(
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str = "silu",
        use_fp8_w8a8: bool = False,
        use_int8_w8a16: bool = False,
        use_int4_w4a16: bool = False,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        w1_scale: Optional[torch.Tensor] = None,
        w2_scale: Optional[torch.Tensor] = None,
        w1_zp: Optional[torch.Tensor] = None,
        w2_zp: Optional[torch.Tensor] = None,
        a1_scale: Optional[torch.Tensor] = None,
        a2_scale: Optional[torch.Tensor] = None,
        block_shape: Optional[List[int]] = None) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="outplace_fused_experts",
    op_func=outplace_fused_experts,
    mutates_args=[],
    fake_impl=outplace_fused_experts_fake,
)


def torch_vllm_inplace_fused_experts(**kwargs) -> torch.Tensor:
    torch.ops.vllm.inplace_fused_experts(**kwargs)
    hidden_states = kwargs['hidden_states']
    return hidden_states


def torch_vllm_outplace_fused_experts(**kwargs) -> torch.Tensor:
    return torch.ops.vllm.outplace_fused_experts(**kwargs)


def dispatch_fused_experts_func(inplace: bool) -> Callable[..., torch.Tensor]:
    if is_rocm_aiter_moe_enabled():
        return rocm_aiter_fused_experts
    if inplace:
        return torch_vllm_inplace_fused_experts
    return torch_vllm_outplace_fused_experts


def fused_experts(hidden_states: torch.Tensor,
                  w1: torch.Tensor,
                  w2: torch.Tensor,
                  topk_weights: torch.Tensor,
                  topk_ids: torch.Tensor,
                  inplace: bool = False,
                  activation: str = "silu",
                  use_fp8_w8a8: bool = False,
                  use_int8_w8a16: bool = False,
                  use_int4_w4a16: bool = False,
                  global_num_experts: int = -1,
                  expert_map: Optional[torch.Tensor] = None,
                  w1_scale: Optional[torch.Tensor] = None,
                  w2_scale: Optional[torch.Tensor] = None,
                  w1_zp: Optional[torch.Tensor] = None,
                  w2_zp: Optional[torch.Tensor] = None,
                  a1_scale: Optional[torch.Tensor] = None,
                  a2_scale: Optional[torch.Tensor] = None,
                  block_shape: Optional[List[int]] = None,
                  allow_deep_gemm: bool = False) -> torch.Tensor:
    if (allow_deep_gemm and use_fp8_w8a8
            and _valid_deep_gemm(hidden_states, w1, w2, expert_map)):
        return deep_gemm_moe_fp8(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
        )
    else:
        return dispatch_fused_experts_func(inplace)(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_zp=w1_zp,
            w2_zp=w2_zp,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            block_shape=block_shape)


def _fp8_perm(m: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    A permutation routine that works on fp8 types.
    """
    if torch.is_floating_point(m) and torch.finfo(m.dtype).bits == 8:
        return m.view(dtype=torch.uint8)[idx, ...].view(dtype=m.dtype)
    else:
        return m[idx, ...]


def _moe_permute(
    curr_hidden_states: torch.Tensor,
    a1q_scale: Optional[torch.Tensor],
    curr_topk_ids: torch.Tensor,
    global_num_experts: int,
    expert_map: Optional[torch.Tensor],
    top_k_num: int,
    block_m: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor,
           torch.Tensor]:
    """
    Determine the sorted_token_ids, expert_ids for the given problem size.
    Permute the hidden states and scales according to `sorted_token_ids`.
    """
    tokens_in_chunk, _ = curr_hidden_states.shape

    sorted_token_ids, expert_ids, num_tokens_post_padded = (
        moe_align_block_size(curr_topk_ids,
                             block_m,
                             global_num_experts,
                             expert_map,
                             pad_sorted_ids=True))

    inv_perm: Optional[torch.Tensor] = None

    num_tokens = top_k_num * tokens_in_chunk
    sorted_token_ids = sorted_token_ids.clamp(max=num_tokens - 1)
    expert_ids = torch.repeat_interleave(expert_ids, block_m, dim=0)
    inv_perm = torch.argsort(sorted_token_ids)[:num_tokens]

    # Permute according to sorted token ids.
    curr_hidden_states = _fp8_perm(curr_hidden_states,
                                   sorted_token_ids // top_k_num)

    if a1q_scale is not None:
        a1q_scale = a1q_scale[sorted_token_ids // top_k_num]

    return (curr_hidden_states, a1q_scale, sorted_token_ids, expert_ids,
            inv_perm)


def _moe_unpermute_and_reduce(
    out: torch.Tensor,
    curr_hidden: torch.Tensor,
    inv_perm: Optional[torch.Tensor],
    topk: int,
    K: int,
    topk_weight: torch.Tensor,
) -> None:
    """
    Unpermute the final result and apply topk_weights, then perform the final
    reduction on the hidden states.
    """
    M = topk_weight.shape[0]
    curr_hidden = curr_hidden[inv_perm, ...]
    curr_hidden = curr_hidden.view(-1, topk, K)
    curr_hidden.mul_(topk_weight.view(M, -1, 1))
    ops.moe_sum(curr_hidden, out)


def _resize_cache(x: torch.Tensor, v: Tuple[int, ...]) -> torch.Tensor:
    """
    Shrink the given tensor and apply the given view to it.  This is
    used to resize the intermediate fused_moe caches.
    """
    assert prod(v) <= x.numel()
    return x.flatten()[:prod(v)].view(*v)


def fused_experts_impl(hidden_states: torch.Tensor,
                       w1: torch.Tensor,
                       w2: torch.Tensor,
                       topk_weights: torch.Tensor,
                       topk_ids: torch.Tensor,
                       inplace: bool = False,
                       activation: str = "silu",
                       use_fp8_w8a8: bool = False,
                       use_int8_w8a16: bool = False,
                       use_int4_w4a16: bool = False,
                       global_num_experts: int = -1,
                       expert_map: Optional[torch.Tensor] = None,
                       w1_scale: Optional[torch.Tensor] = None,
                       w2_scale: Optional[torch.Tensor] = None,
                       w1_zp: Optional[torch.Tensor] = None,
                       w2_zp: Optional[torch.Tensor] = None,
                       a1_scale: Optional[torch.Tensor] = None,
                       a2_scale: Optional[torch.Tensor] = None,
                       block_shape: Optional[List[int]] = None):

    EXP_FEATURE_ON = False
    EXP_FEATURE_CONFIG={
        "expert_block_threshold":10
    }
    # Check constraints.
    if use_int4_w4a16:
        assert hidden_states.shape[1] // 2 == w1.shape[
            2], "Hidden size mismatch"
    else:
        assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"

    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
    assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
    assert hidden_states.dtype in [
        torch.float32, torch.float16, torch.bfloat16
    ]
    if EXP_FEATURE_ON:
        # EXP_expert_duplicate(topk_ids,32,30)
        # ww = torch.empty(w1.shape[0]+1,
        #                  w1.shape[1],
        #                 w1.shape[2],
        #                 dtype=w1.dtype,
        #                 device=hidden_states.device)
        # ww = torch.cat([w1, w1[-1].unsqueeze(0)], dim=0)
        # ww = w1[-1].unsqueeze(0)
        new_topk_ids,new_exp_map =EXP_expert_duplicate(topk_ids.flatten(),1024,30)
        new_topk_ids.to(topk_ids.dtype)
        new_topk_ids = new_topk_ids.view(topk_ids.shape)
        
        slices1,slices2 = w1[new_exp_map],w2[new_exp_map]   # shape [len(map), H, W]
        # Append to the back
        w1 = torch.cat([w1, slices1], dim=0) 
        w2 = torch.cat([w2, slices2], dim=0) 

    num_tokens, _ = hidden_states.shape
    E, N, _ = w1.shape
    K = w2.shape[1]
    if global_num_experts == -1:
        global_num_experts = E
    top_k_num = topk_ids.shape[1]
    # We execute the fused_moe kernel in chunks to circumvent this issue:
    # https://github.com/vllm-project/vllm/issues/5938
    CHUNK_SIZE = envs.VLLM_FUSED_MOE_CHUNK_SIZE
    M = min(num_tokens, CHUNK_SIZE)
    config_dtype = get_config_dtype_str(use_fp8_w8a8=use_fp8_w8a8,
                                        use_int8_w8a16=use_int8_w8a16,
                                        use_int4_w4a16=use_int4_w4a16,
                                        dtype=hidden_states.dtype)

    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        w1.shape,
        w2.shape,
        top_k_num,
        config_dtype,
        block_shape=block_shape,
    )

    config = get_config_func(M)

    # We can reuse the memory between these because by the time we need
    # cache3, we're done with cache1
    cache13 = torch.empty(M * top_k_num * max(N, K),
                          device=hidden_states.device,
                          dtype=hidden_states.dtype)
    intermediate_cache1 = cache13[:M * top_k_num * N].view(M, top_k_num, N)
    intermediate_cache3 = cache13[:M * top_k_num * K].view(M, top_k_num, K)

    # This needs separate memory since it's used concurrently with cache1
    intermediate_cache2 = torch.empty((M * top_k_num, N // 2),
                                      device=hidden_states.device,
                                      dtype=hidden_states.dtype)

    if hidden_states.dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    elif hidden_states.dtype == torch.float16:
        compute_type = tl.float16
    elif hidden_states.dtype == torch.float32:
        compute_type = tl.float32
    else:
        raise ValueError(f"Unsupported compute_type: {hidden_states.dtype}")

    if inplace:
        out_hidden_states = hidden_states
    else:
        out_hidden_states = torch.empty_like(hidden_states)

    for chunk in range((num_tokens // CHUNK_SIZE) + 1):
        begin_chunk_idx, end_chunk_idx = (chunk * CHUNK_SIZE,
                                          min((chunk + 1) * CHUNK_SIZE,
                                              num_tokens))
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, _ = curr_hidden_states.shape

        if tokens_in_chunk == 0:
            break

        if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
            # Adjust the intermediate cache size and config for the last
            # chunk. Note that in most cases we only have one chunk
            # so the cache size and config are already set correctly and
            # do not need to be adjusted.
            intermediate_cache1 = intermediate_cache1[:tokens_in_chunk]
            intermediate_cache2 = intermediate_cache2[:tokens_in_chunk *
                                                      topk_ids.shape[1]]
            intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]
            config = get_config_func(tokens_in_chunk)

        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

        a1q_scale: Optional[torch.Tensor] = None

        if use_fp8_w8a8:
            qcurr_hidden_states, a1q_scale = _fp8_quantize(
                curr_hidden_states, a1_scale, block_shape)
        else:
            qcurr_hidden_states = curr_hidden_states
            a1q_scale = a1_scale

        sorted_token_ids, expert_ids, num_tokens_post_padded = (
            moe_align_block_size(curr_topk_ids, config['BLOCK_SIZE_M'],
                                 global_num_experts, expert_map))

        logger.info(f"expert_ids.shape: {expert_ids.shape}")
        invoke_fused_moe_kernel(qcurr_hidden_states,
                                w1,
                                intermediate_cache1,
                                a1q_scale,
                                w1_scale,
                                w1_zp,
                                curr_topk_weights,
                                curr_topk_ids,
                                sorted_token_ids,
                                expert_ids,
                                num_tokens_post_padded,
                                False,
                                top_k_num,
                                config,
                                compute_type=compute_type,
                                use_fp8_w8a8=use_fp8_w8a8,
                                use_int8_w8a16=use_int8_w8a16,
                                use_int4_w4a16=use_int4_w4a16,
                                block_shape=block_shape)

        if activation == "silu":
            torch.ops._C.silu_and_mul(intermediate_cache2,
                                      intermediate_cache1.view(-1, N))
        elif activation == "gelu":
            torch.ops._C.gelu_and_mul(intermediate_cache2,
                                      intermediate_cache1.view(-1, N))
        else:
            raise ValueError(f"Unsupported FusedMoe activation: {activation}")

        a2q_scale: Optional[torch.Tensor] = None

        if use_fp8_w8a8:
            qintermediate_cache2, a2q_scale = _fp8_quantize(
                intermediate_cache2, a2_scale, block_shape)
        else:
            qintermediate_cache2 = intermediate_cache2
            a2q_scale = a2_scale

        invoke_fused_moe_kernel(qintermediate_cache2,
                                w2,
                                intermediate_cache3,
                                a2q_scale,
                                w2_scale,
                                w2_zp,
                                curr_topk_weights,
                                curr_topk_ids,
                                sorted_token_ids,
                                expert_ids,
                                num_tokens_post_padded,
                                True,
                                1,
                                config,
                                compute_type=compute_type,
                                use_fp8_w8a8=use_fp8_w8a8,
                                use_int8_w8a16=use_int8_w8a16,
                                use_int4_w4a16=use_int4_w4a16,
                                block_shape=block_shape)

        ops.moe_sum(intermediate_cache3.view(*intermediate_cache3.shape),
                    out_hidden_states[begin_chunk_idx:end_chunk_idx])
    return out_hidden_states

def EXP_expert_duplicate(expert_ids,threshold,max_expert_id):
    device = expert_ids.device
    expert_ids=expert_ids.flatten()
    # Step 1: Count tokens per expert (graph-safe)
    expert_counts = torch.bincount(expert_ids)

    # Step 2: Identify experts needing duplication (graph-safe)
    experts_to_split = torch.nonzero(expert_counts > threshold).flatten()
    
    # Step 3: Calculate duplicates needed (graph-safe)
    duplicate_counts = (expert_counts[experts_to_split] - 1) // threshold
    num_new_experts = torch.sum(duplicate_counts).item()
    
    # Step 4: Create offset mapping (graph-safe)
    duplicate_starts = torch.zeros_like(expert_counts)
    if experts_to_split.numel() > 0:
        offsets = torch.cat([
            torch.tensor([0], device=device),
            torch.cumsum(duplicate_counts, dim=0)
        ])
        duplicate_starts = torch.scatter(
            torch.zeros_like(expert_counts),
            0,
            experts_to_split,
            max_expert_id + 1 + offsets
        )
        
    sorted_ids, sort_idx = torch.sort(expert_ids)
    diff = torch.cat([
            torch.ones(1, dtype=torch.bool, device=device),
            sorted_ids[1:] != sorted_ids[:-1]
        ])

    
    segment_ids = torch.cumsum(diff, dim=0) - 1
    counts = torch.bincount(segment_ids)


    
    # Step 4: Calculate segment start positions
    segment_starts = torch.cat([
        torch.zeros(1, dtype=torch.long, device=device),
        torch.cumsum(counts, dim=0)[:-1]
    ])
    # print(f"segment_starts: {segment_starts}")
    
    # Step 5: Compute counters within each segment
    counters = torch.arange(len(expert_ids), device=device) - \
              segment_starts.repeat_interleave(counts)

    
    counters = counters[torch.argsort(sort_idx)]

        
    # Step 6: Create final assignments (graph-safe)
    reassign_mask = counters >= threshold
    
    reassign_pos = (counters - threshold) // threshold
    
    new_expert_ids = torch.where(
        reassign_mask,
        duplicate_starts[expert_ids] + reassign_pos,
        expert_ids
    )
    map_mask = torch.where(new_expert_ids!=expert_ids,new_expert_ids,-1)!=-1
    new_expert_map = torch.empty(new_expert_ids.max().item() + 1, device=new_expert_ids.device,
                    dtype=torch.int32)

    # Step 2: Fill C
    new_expert_map[new_expert_ids[map_mask]] = expert_ids[map_mask]
    new_expert_map = new_expert_map[max_expert_id:]
    return new_expert_ids, new_expert_map

def fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    inplace: bool = False,
    activation: str = "silu",
    use_grouped_topk: bool = False,
    num_expert_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    custom_routing_function: Optional[Callable] = None,
    use_fp8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    This function computes a Mixture of Experts (MoE) layer using two sets of
    weights, w1 and w2, and top-k gating mechanism.

    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.
    - w1 (torch.Tensor): The first set of expert weights.
    - w2 (torch.Tensor): The second set of expert weights.
    - gating_output (torch.Tensor): The output of the gating operation
        (before softmax).
    - topk (int): The number of top-k experts to select.
    - renormalize (bool): If True, renormalize the top-k weights to sum to 1.
    - inplace (bool): If True, perform the operation in-place.
        Defaults to False.
    - activation (str): The activation function to apply after the first
        MoE layer.
    - num_expert_group: Optional[int]: additional parameter for grouped_topk
    - topk_group: Optional[int]: additional parameter for grouped_topk
    - use_grouped_topk: If True, use grouped_topk instead of fused_topk
        note: Deepseekv2 model uses grouped_topk
    - use_fp8_w8a8 (bool): If True, use fp8 arithmetic to compute the inner
        products for w1 and w2. Defaults to False.
    - use_int8_w8a16 (bool): If True, use matmul of int8 weight and bf16/fp16
        activation to compute the inner products for w1 and w2.
        Defaults to False.
    - use_int4_w4a16 (bool): If True, use matmul of int4 weight and bf16/fp16
        activation to compute the inner products for w1 and w2.
        Defaults to False.
    - global_num_experts (int): The total number of experts in the global
        expert space.
    - expert_map (Optional[torch.Tensor]):  A tensor mapping expert indices 
        from the global expert space to the local expert space of the expert 
        parallel shard.
    - w1_scale (Optional[torch.Tensor]): Optional scale to be used for
        w1.
    - w2_scale (Optional[torch.Tensor]): Optional scale to be used for
        w2.
    - a1_scale (Optional[torch.Tensor]): Optional scale to be used for
        a1.
    - a2_scale (Optional[torch.Tensor]): Optional scale to be used for
        a2.
    - block_shape: (Optional[List[int]]): Optional block size for block-wise
        quantization.

    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer.
    """

    if use_grouped_topk:
        assert num_expert_group is not None and topk_group is not None
        topk_weights, topk_ids = grouped_topk(hidden_states, gating_output,
                                              topk, renormalize,
                                              num_expert_group, topk_group)
    elif custom_routing_function is None:
        topk_weights, topk_ids = fused_topk(hidden_states, gating_output, topk,
                                            renormalize)
    else:
        topk_weights, topk_ids = custom_routing_function(
            hidden_states, gating_output, topk, renormalize)
        
        
    ## EXP_ feature

    
    
    ## EXP_ feature

    return fused_experts(hidden_states,
                         w1,
                         w2,
                         topk_weights,
                         topk_ids,
                         inplace=inplace,
                         activation=activation,
                         use_fp8_w8a8=use_fp8_w8a8,
                         use_int8_w8a16=use_int8_w8a16,
                         use_int4_w4a16=use_int4_w4a16,
                         global_num_experts=global_num_experts,
                         expert_map=expert_map,
                         w1_scale=w1_scale,
                         w2_scale=w2_scale,
                         w1_zp=w1_zp,
                         w2_zp=w2_zp,
                         a1_scale=a1_scale,
                         a2_scale=a2_scale,
                         block_shape=block_shape)


def deep_gemm_moe_fp8(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    This function computes a a8w8-quantized Mixture of Experts (MoE) layer
    using two sets of quantized weights, w1_q and w2_q, and top-k gating
    mechanism. The matrix multiplications are implemented with DeepGemm
    grouped gemm.

    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.
        Shape: [M, K]
    - w1 (torch.Tensor): The first set of fp8 quantized expert weights.
        Shape: [num_experts, K, 2N] (the weights are passed transposed)
    - w2 (torch.Tensor): The second set of fp8 quantized expert weights.
        Shape: [num_experts, N, K] (the weights are passed transposed)
    - w1_scale (torch.Tensor): The fp32 scale to dequantize w1_q.
        Shape: [num_experts] or [num_experts, 2N]
    - w2_scale (torch.Tensor): The fp32 scale to dequantize w2_q.
        Shape: [num_experts] or [num_experts, K]
    - topk_weights (torch.Tensor): The weights of each token->expert mapping.
    - topk_ids (torch.Tensor): The token->expert mapping for topk_weights.
    - inplace (bool): If True, perform the operation in-place.
        Defaults to False.
    - activation (str): The activation function to apply after the first
        MoE layer.
    - global_num_experts (int): The total number of experts in the global
        expert space.
    - expert_map (Optional[torch.Tensor]):  A tensor mapping expert indices
        from the global expert space to the local expert space of the expert
        parallel shard.
    - a1_scale (Optional[torch.Tensor]): The optional fp32 scale to quantize a.
        Shape: scalar or [M]
    - a2_scale (Optional[torch.Tensor]): The optional fp32 scale to
        quantize the intermediate result between the gemms.
        Shape: scalar or [M]

    Returns:
    - torch.Tensor: The bfloat16 output tensor after applying the MoE layer.
    """
    # Lazy import to avoid CUDA initialization problems.
    import deep_gemm as dg

    assert expert_map is None, "Expert maps not supported yet"

    assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"

    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
    assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
    assert hidden_states.dtype in [
        torch.float32, torch.float16, torch.bfloat16
    ]
    assert w1.dtype == torch.float8_e4m3fn
    assert w2.dtype == torch.float8_e4m3fn
    assert w1.shape[0] == w2.shape[0], "Expert number mismatch"
    assert w1.shape[0] == w1_scale.shape[0], "w1 scales expert number mismatch"
    assert w1.shape[0] == w2_scale.shape[0], "w2 scales expert number mismatch"
    assert a1_scale is None or a1_scale.dim(
    ) == 0 or a1_scale.shape[0] == 1 or a1_scale.shape[
        0] == hidden_states.shape[0], "Input scale shape mismatch"
    assert a2_scale is None or a1_scale is None or a2_scale.shape == a1_scale.shape, "Intermediate scale shape mismatch"  # noqa: E501

    num_tokens, _ = hidden_states.shape
    E, N, _ = w1.shape
    K = w2.shape[1]
    if global_num_experts == -1:
        global_num_experts = E
    top_k_num = topk_ids.shape[1]
    # We execute the fused_moe kernel in chunks to circumvent this issue:
    # https://github.com/vllm-project/vllm/issues/5938
    CHUNK_SIZE = envs.VLLM_FUSED_MOE_CHUNK_SIZE

    assert _valid_deep_gemm(hidden_states, w1, w2, expert_map)

    if inplace:
        out_hidden_states = hidden_states
    else:
        out_hidden_states = torch.empty_like(hidden_states)

    block_m = dg.get_m_alignment_for_contiguous_layout()
    block_shape = [block_m, block_m]

    assert w1_scale is not None
    assert w2_scale is not None

    # We attempt to transpose and align offline in Fp8MoEMethod, in which
    # case these calls will be nops.  Otherwise, they'll be performed every
    # time the layer is executed.
    w1_scale = dg.get_col_major_tma_aligned_tensor(w1_scale).contiguous()
    w2_scale = dg.get_col_major_tma_aligned_tensor(w2_scale).contiguous()

    M_sum = topk_ids.numel() + global_num_experts * (block_m - 1)
    M_sum = round_up(M_sum, block_m)

    num_chunks = (num_tokens // CHUNK_SIZE) + 1

    # We can reuse the memory between cache1 and cache3 because by the time
    # we need cache3, we're done with cache1
    cache13 = torch.empty(M_sum * max(N, K),
                          device=hidden_states.device,
                          dtype=hidden_states.dtype)

    intermediate_cache1 = cache13[:M_sum * N].view(M_sum, N)
    intermediate_cache2 = torch.empty((M_sum, N // 2),
                                      device=hidden_states.device,
                                      dtype=hidden_states.dtype)
    intermediate_cache3 = cache13[:M_sum * K].view(M_sum, K)

    for chunk in range(num_chunks):
        begin_chunk_idx, end_chunk_idx = (chunk * CHUNK_SIZE,
                                          min((chunk + 1) * CHUNK_SIZE,
                                              num_tokens))
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, _ = curr_hidden_states.shape

        if tokens_in_chunk == 0:
            break

        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

        a1q_scale: Optional[torch.Tensor] = None

        qcurr_hidden_states, a1q_scale = _fp8_quantize(curr_hidden_states,
                                                       a1_scale, block_shape)

        (qcurr_hidden_states, a1q_scale, sorted_token_ids, expert_ids,
         inv_perm) = _moe_permute(qcurr_hidden_states, a1q_scale,
                                  curr_topk_ids, global_num_experts,
                                  expert_map, top_k_num, block_m)

        # Adjust the intermediate cache size and config for the last chunk.
        # Note that in most cases we only have one chunk so the cache size
        # and config are already set correctly and do not need to be adjusted.
        if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
            curr_M = sorted_token_ids.numel()
            intermediate_cache1 = _resize_cache(intermediate_cache1,
                                                (curr_M, N))
            intermediate_cache2 = _resize_cache(intermediate_cache2,
                                                (curr_M, N // 2))
            intermediate_cache3 = _resize_cache(intermediate_cache3,
                                                (curr_M, K))

        dg.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
            (qcurr_hidden_states, a1q_scale), (w1, w1_scale),
            intermediate_cache1, expert_ids)

        if activation == "silu":
            torch.ops._C.silu_and_mul(intermediate_cache2,
                                      intermediate_cache1.view(-1, N))
        elif activation == "gelu":
            torch.ops._C.gelu_and_mul(intermediate_cache2,
                                      intermediate_cache1.view(-1, N))
        else:
            raise ValueError(f"Unsupported FusedMoe activation: {activation}")

        a2q_scale: Optional[torch.Tensor] = None

        qintermediate_cache2, a2q_scale = _fp8_quantize(
            intermediate_cache2, a2_scale, block_shape)

        dg.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
            (qintermediate_cache2, a2q_scale), (w2, w2_scale),
            intermediate_cache3, expert_ids)

        _moe_unpermute_and_reduce(
            out_hidden_states[begin_chunk_idx:end_chunk_idx],
            intermediate_cache3.view(*intermediate_cache3.shape), inv_perm,
            top_k_num, K, curr_topk_weights)

    return out_hidden_states


#TODO make the grouped gemm kernel consistent with scaled gemm kernel
def cutlass_moe_fp8(
    a: torch.Tensor,
    w1_q: torch.Tensor,
    w2_q: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    ab_strides1: torch.Tensor,
    c_strides1: torch.Tensor,
    ab_strides2: torch.Tensor,
    c_strides2: torch.Tensor,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.half,
) -> torch.Tensor:
    """
    This function computes a a8w8-quantized Mixture of Experts (MoE) layer
    using two sets of quantized weights, w1_q and w2_q, and top-k gating
    mechanism. The matrix multiplications are implemented with CUTLASS
    grouped gemm.

    Parameters:
    - a (torch.Tensor): The input tensor to the MoE layer.
        Shape: [M, K]
    - w1_q (torch.Tensor): The first set of fp8-quantized expert weights.
        Shape: [num_experts, K, 2N] (the weights are passed transposed)
    - w2_q (torch.Tensor): The second set of fp8-quantized expert weights.
        Shape: [num_experts, N, K] (the weights are passed transposed)
    - w1_scale (torch.Tensor): The fp32 scale to dequantize w1_q.
        Shape: [num_experts] or [num_experts, 2N]
    - w2_scale (torch.Tensor): The fp32 scale to dequantize w2_q.
        Shape: [num_experts] or [num_experts, K]
    - gating_output (torch.Tensor): The output of the gating operation
        (before softmax).
    - topk_weights (torch.Tensor): The weights of each token->expert mapping.
    - ab_strides1 (torch.Tensor): The input and weights strides of the first
        grouped gemm.
    - c_strides1 (torch.Tensor): The output strides of the first grouped gemm.
    - ab_strides2 (torch.Tensor): The input and weights strides of the second
        grouped gemm.
    - c_strides2 (torch.Tensor): The output strides of the second grouped gemm.
    - a1_scale (Optional[torch.Tensor]): The optional fp32 scale to quantize a.
        Shape: scalar or [M]
    - a2_scale (Optional[torch.Tensor]): The optional fp32 scale to
        quantize the intermediate result between the gemms.
        Shape: scalar or [M]
    - out_dtype (torch.Tensor): The output tensor type.

    Returns:
    - torch.Tensor: The fp16 output tensor after applying the MoE layer.
    """

    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert w1_q.dtype == torch.float8_e4m3fn
    assert w2_q.dtype == torch.float8_e4m3fn
    assert a.shape[1] == w1_q.shape[1], "Hidden size mismatch w1"
    assert w1_q.shape[2] == w2_q.shape[1] * 2, "Hidden size mismatch w2"
    assert w1_q.shape[0] == w2_q.shape[0], "Expert number mismatch"
    assert a1_scale is None or a1_scale.dim(
    ) == 0 or a1_scale.shape[0] == 1 or a1_scale.shape[0] == a.shape[
        0], "Input scale shape mismatch"
    assert w1_scale.dim() == 1 or w1_scale.shape[1] == 1 or w1_scale.shape[
        1] == w1_q.shape[2], "W1 scale shape mismatch"
    assert w2_scale.dim() == 1 or w2_scale.shape[1] == 1 or w2_scale.shape[
        1] == w2_q.shape[2], "W2 scale shape mismatch"
    assert w1_q.shape[0] == w2_q.shape[0], "Weights expert number mismatch"
    assert w1_q.shape[0] == w1_scale.shape[
        0], "w1 scales expert number mismatch"
    assert w1_q.shape[0] == w2_scale.shape[
        0], "w2 scales expert number mismatch"
    assert a2_scale is None or a1_scale is None or a2_scale.shape == a1_scale.shape, "Intermediate scale shape mismatch"  # noqa: E501
    assert ab_strides1.shape[0] == w1_q.shape[
        0], "AB Strides 1 expert number mismatch"
    assert c_strides1.shape[0] == w1_q.shape[
        0], "C Strides 1 expert number mismatch"
    assert ab_strides2.shape[0] == w2_q.shape[
        0], "AB Strides 2 expert number  mismatch"
    assert c_strides2.shape[0] == w2_q.shape[
        0], "C Strides 2 expert number mismatch"
    assert out_dtype in [torch.half, torch.bfloat16], "Invalid output dtype"

    num_experts = w1_q.size(0)
    m = a.size(0)
    k = w1_q.size(1)
    n = w2_q.size(1)

    topk = topk_ids.size(1)
    per_act_token = a1_scale.numel() != 1 if a1_scale is not None else (
        a2_scale.numel() != 1 if a2_scale is not None else False)

    a_q, a1_scale = ops.scaled_fp8_quant(
        a, a1_scale, use_per_token_if_dynamic=per_act_token)
    device = a_q.device

    expert_offsets = torch.empty((num_experts + 1),
                                 dtype=torch.int32,
                                 device=device)
    problem_sizes1 = torch.empty((num_experts, 3),
                                 dtype=torch.int32,
                                 device=device)
    problem_sizes2 = torch.empty((num_experts, 3),
                                 dtype=torch.int32,
                                 device=device)

    a_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)
    c_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)

    ops.get_cutlass_moe_mm_data(topk_ids, expert_offsets, problem_sizes1,
                                problem_sizes2, a_map, c_map, num_experts, n,
                                k)

    rep_a_q = a_q.view(dtype=torch.uint8)[a_map].view(dtype=a_q.dtype)
    rep_a1_scales = a1_scale[a_map] if per_act_token else a1_scale

    c1 = torch.empty((m * topk, n * 2), device=device, dtype=out_dtype)
    c2 = torch.empty((m * topk, k), device=device, dtype=out_dtype)

    ops.cutlass_moe_mm(c1, rep_a_q, w1_q, rep_a1_scales, w1_scale,
                       expert_offsets[:-1], problem_sizes1, ab_strides1,
                       ab_strides1, c_strides1)

    intermediate = torch.empty((m * topk, n), device=device, dtype=out_dtype)
    torch.ops._C.silu_and_mul(intermediate, c1)

    intemediate_q, a2_scale = ops.scaled_fp8_quant(
        intermediate, a2_scale, use_per_token_if_dynamic=per_act_token)

    ops.cutlass_moe_mm(c2, intemediate_q, w2_q, a2_scale, w2_scale,
                       expert_offsets[:-1], problem_sizes2, ab_strides2,
                       ab_strides2, c_strides2)

    return (c2[c_map].view(m, topk, k) *
            topk_weights.view(m, topk, 1).to(out_dtype)).sum(dim=1)
