# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Overlay of vllm/model_executor/layers/fused_moe/moe_align_block_size.py at 8a728663 (T14,
# journal/16-nondeterminism.md). MOEDET (QWEN4EXP_MOE_DET=1): the moe_align_block_size kernel places each expert's
# tokens into its blocks in whatever order its atomics run, and the Marlin MoE GEMM's rounding can depend on which
# block of its expert a token lands in (one bf16 step; never on the slot inside a block). With MOEDET, each expert's
# token list is re-sorted ascending after the kernel, so a token's block depends only on the routing. Skipped when the
# batch has no more tokens than block_size (no expert can then fill more than one block). One Triton program per
# expert rewrites its segment, so the change is one kernel launch per MoE layer. With it unset, the code below is the
# image's. Changes are marked MOEDET.
import os

import torch

from vllm import _custom_ops as ops
from vllm.triton_utils import tl, triton  # MOEDET: tl
from vllm.utils.math_utils import round_up

_MOE_DET = os.environ.get("QWEN4EXP_MOE_DET") == "1"  # MOEDET


def _canonical_sorted_ids(  # MOEDET
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    block_size: int,
    numel: int,
) -> torch.Tensor:
    """Reference for the kernel below (used by the checks in journal/16): each expert's segment of sorted_ids in
    ascending order. A segment is a run of blocks with the same expert id;
    the padding value numel sorts after every token. Positions past num_tokens_post_pad keep a key above every
    segment, so they stay at the end. Fixed shapes and no host sync, so it can be captured in a CUDA graph."""
    pos = torch.arange(sorted_ids.numel(), device=sorted_ids.device)
    change = torch.zeros_like(expert_ids, dtype=torch.int64)
    change[1:] = expert_ids[1:] != expert_ids[:-1]
    seg = torch.cumsum(change, 0)[(pos // block_size).clamp_(max=expert_ids.numel() - 1)]
    seg = torch.where(pos < num_tokens_post_pad, seg, expert_ids.numel() + 1)
    key = seg * (numel + 1) + sorted_ids.long()
    return (torch.sort(key).values % (numel + 1)).to(sorted_ids.dtype)


@triton.jit
def _canonical_by_expert_kernel(  # MOEDET
    topk_ids_ptr,
    expert_map_ptr,
    expert_ids_ptr,
    num_tokens_post_pad_ptr,
    sorted_ids_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
    TILE: tl.constexpr,
    HAS_MAP: tl.constexpr,
):
    # One program per expert. The kernel left the expert's tokens in the first slots of its segment; this program
    # finds the segment (the first block with its id), scans all token-expert pairs in index order and writes its own
    # pairs back into those slots, ascending. Each program touches only its own segment.
    e = tl.program_id(0)
    nb = tl.load(num_tokens_post_pad_ptr) // BLOCK_SIZE
    first = nb
    for b0 in range(0, nb, TILE):
        ob = b0 + tl.arange(0, TILE)
        be = tl.load(expert_ids_ptr + ob, mask=ob < nb, other=-2)
        first = tl.minimum(first, tl.min(tl.where(be == e, ob, nb), axis=0))
    if first < nb:
        written = 0
        for i0 in range(0, numel, TILE):
            oi = i0 + tl.arange(0, TILE)
            m = oi < numel
            t = tl.load(topk_ids_ptr + oi, mask=m, other=-1).to(tl.int32)
            if HAS_MAP:
                ex = tl.load(expert_map_ptr + t, mask=m & (t >= 0), other=-1)
            else:
                ex = t
            hit = (m & (ex == e)).to(tl.int32)
            pos = written + tl.cumsum(hit, axis=0) - hit
            tl.store(sorted_ids_ptr + first * BLOCK_SIZE + pos, oi, mask=hit == 1)
            written += tl.sum(hit, axis=0)


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns the token distribution across experts to be compatible with block
    size for matrix multiplication.

    Note: In the case of expert_parallel, moe_align_block_size initially
    considers all experts as valid and aligns all tokens appropriately.
    Before the function returns it marks the experts_ids that are not in
    the current GPU rank as -1 so the MoE matmuls could skip those blocks.
    This requires the num_experts input arg to be the num global experts.

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
    - ignore_invalid_experts: A flag indicating whether to ignore invalid
        experts. When False, all expert_ids in topk_ids will participate in
        counting and ranking, but invalid experts in expert_ids will be marked
        as -1. When True, all invalid expert_ids in topk_ids will be ignored
        and will not participate in counting or ranking, and there will be no
        -1 in expert_ids.

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
    if topk_ids.numel() < num_experts:
        max_num_tokens_padded = min(
            topk_ids.numel() * block_size, max_num_tokens_padded
        )
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)

    ops.moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        expert_map if ignore_invalid_experts else None,
    )
    if _MOE_DET and topk_ids.shape[0] > block_size:  # MOEDET
        use_map = expert_map is not None and ignore_invalid_experts
        _canonical_by_expert_kernel[(expert_map.numel() if use_map else num_experts,)](
            topk_ids,
            expert_map if use_map else topk_ids,
            expert_ids,
            num_tokens_post_pad,
            sorted_ids,
            topk_ids.numel(),
            BLOCK_SIZE=block_size,
            TILE=1024,
            HAS_MAP=use_map,
            num_warps=4,
        )

    if expert_map is not None and not ignore_invalid_experts:
        expert_ids = expert_map[expert_ids]

    return sorted_ids, expert_ids, num_tokens_post_pad


def batched_moe_align_block_size(
    max_tokens_per_batch: int, block_size: int, expert_num_tokens: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Given num_batches, max_tokens_per_batch, block_size and the number of
    valid-tokens in each batch, prepare sorted_token_ids, expert_ids and
    num_tokens_post_pad. sorted_token_ids, expert_ids and num_tokens_post_pad
    have the same semantics as in moe_align_block_size.

    This function is intended to be a drop in replacement for
    moe_align_batch_size for the batched case.

    Parameters:
    - max_tokens_per_batch (int): Number of tokens in each batch (both
        valid and invalid).
    - block_size (int): block_size to align the data to.
    - expert_num_tokens (torch.Tensor): expert_num_tokens[i], indicates
        the number of valid tokens in batch i.

    Returns:
    - sorted_token_ids (torch.Tensor): Torch tensor of size
        (num_batches * max_tokens_per_batch) indicating the token indices for
        that block.
    - expert_ids (torch.Tensor): Torch tensor of size
        ceil((num_batches * max_tokens_per_batch) / block_size) indicating
        what expert to use for each block.
    - num_tokens_post_pad (torch.Tensor): Torch tensor of size 1
        indicating the number of valid blocks with actual data to
        process. This is represented in terms of num tokens.
    Example:
    Let num_batches=5, max_tokens_per_batch=8, block_size=4, and
    expert_num_tokens=[2, 3, 0, 6, 8]. This expert_num_tokens tensor
    indicates that,
     - The first 2 tokens in the 0th batch are valid and the rest 6 are
     invalid (i.e. in the 2D hidden_states tensor of shape,
     [num_batches * max_tokens_per_batch, K], indices 0, 1 are valid)
     - The first 3 tokens in the 1st batch are valid. i.e. indices 8, 9, 10
     - 0 tokens in the 2nd batch are valid
     - first 6 tokens in the  3rd batch are valid. i.e. indices,
     24, 25, 26, 27, 28, 29
     - so on ...

     In this case,
      sorted_token_ids will be [0, 1, 40, 40,
                                8, 9, 10, 40,
                                24, 25, 26, 27,
                                28, 29, 40, 40,
                                32, 33, 34, 35,
                                36, 37, 38, 39,
                                40, 40, 40, 40,
                                (rest all 40, 40, 40, 40)
                                ...]
      Here, 40 represents an invalid index. as there is no token index 40.
      The gemm kernel using this sorted_token_ids is expected to skip the
      gemm computation when it encounters this invalid index.

      expert_ids will be [0, 1, 3, 3, 4, 5, 5, -1, -1, (rest all -1) ...]
      Here, -1 represents an invalid expert. The gemm kernel using this
      expert_ids is expected to skip the gemm computation when it encounters
      an expert of id -1.

      num_tokens_post_pad will be 24 as sorted_token_ids has valid entries
      until 24.
    """

    B = expert_num_tokens.size(0)
    device = expert_num_tokens.device

    # Round up so each batch can be split to blocks evenly.
    max_num_tokens_padded = B * round_up(max_tokens_per_batch, block_size)

    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=device)
    assert max_num_tokens_padded % block_size == 0
    max_num_m_blocks = max_num_tokens_padded // block_size
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=device)
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=device)

    ops.batched_moe_align_block_size(
        max_tokens_per_batch,
        block_size,
        expert_num_tokens,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
    )

    return sorted_ids, expert_ids, num_tokens_post_pad
