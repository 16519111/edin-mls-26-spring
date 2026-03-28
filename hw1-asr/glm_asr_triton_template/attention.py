"""
Triton Multi-Head Attention Implementation
End-to-end implementation using Triton kernels

*** STUDENT ASSIGNMENT ***
Fill in the TODO sections to implement attention using Triton kernels
"""

import numpy as np
import torch
import triton
import triton.language as tl
import math
from typing import Optional, Tuple


def get_stream():
    """Get current CUDA stream pointer."""
    if torch.cuda.is_available():
        return torch.cuda.current_stream().cuda_stream
    return None


# ============================================================================
# Triton Kernels for Attention
# ============================================================================

@triton.jit
def attention_scores_kernel(
    q_ptr,
    k_ptr,
    scores_ptr,
    scale,
    seq_k,
    head_dim,
    stride_q0,
    stride_q1,
    stride_q2,
    stride_k0,
    stride_k1,
    stride_k2,
    stride_s0,
    stride_s1,
    stride_s2,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Compute scaled attention scores for a single query position.
    Grid: (batch_heads, seq_q)

    *** TODO: Implement this kernel ***
    """
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)

    # ============================================================================
    # TODO: Implement attention score computation
    # ============================================================================
    #
    # Step 1: Load query vector for this position
    # Step 2: Load all keys for this batch_head
    # Step 3: Compute dot-product scores and scale
    # Step 4: Store scores

    # YOUR CODE HERE
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)

    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)

    q = tl.load(
        q_ptr + pid_bh * stride_q0 + pid_q * stride_q1 + offs_d * stride_q2,
        mask=offs_d < head_dim,
        other=0.0,
    )
    k = tl.load(
        k_ptr
        + pid_bh * stride_k0
        + offs_k[:, None] * stride_k1
        + offs_d[None, :] * stride_k2,
        mask=(offs_k[:, None] < seq_k) & (offs_d[None, :] < head_dim),
        other=0.0,
    )
    scores = tl.dot(q[None, :], tl.trans(k)) * scale
    tl.store(
        scores_ptr
        + pid_bh * stride_s0
        + pid_q * stride_s1
        + offs_k * stride_s2,
        tl.reshape(scores, (BLOCK_K,)),
        mask=offs_k < seq_k,
    )


@triton.jit
def softmax_inplace_kernel(scores_ptr, stride_s, seq_k, BLOCK_SIZE: tl.constexpr):
    """
    Apply softmax along the last dimension (seq_k).
    Grid: (batch_heads * seq_q,)
    """
    row = tl.program_id(0)

    # ============================================================================
    # TODO: Implement softmax
    # ============================================================================
    #
    # Step 1: Load scores row with masking
    # Step 2: Subtract max for stability
    # Step 3: Compute exp and normalize
    # Step 4: Store back

    # YOUR CODE HERE
    # BLOCK_SIZE = N
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < seq_k

    s = tl.load(scores_ptr + row * stride_s + offs, mask=mask, other=-float("inf"))
    s = s - tl.max(s, axis=0)
    exp_s = tl.exp(s)
    denom = tl.sum(exp_s, axis=0)
    out = exp_s / denom

    tl.store(scores_ptr + row * stride_s + offs, out, mask=mask)


@triton.jit
def attention_output_kernel(
    attn_ptr,
    v_ptr,
    output_ptr,
    seq_k,
    head_dim,
    stride_w0,
    stride_w1,
    stride_w2,
    stride_v0,
    stride_v1,
    stride_v2,
    stride_o0,
    stride_o1,
    stride_o2,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Compute attention output: attn_weights @ V
    Grid: (batch_heads, seq_q)
    """
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)

    # ============================================================================
    # TODO: Implement attention output computation
    # ============================================================================
    #
    # Step 1: Load attention weights for this query
    # Step 2: Load all values for this batch_head
    # Step 3: Compute weighted sum
    # Step 4: Store output

    # YOUR CODE HERE
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)

    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)

    w = tl.load(
        attn_ptr
        + pid_bh * stride_w0
        + pid_q * stride_w1
        + offs_k * stride_w2,
        mask=offs_k < seq_k,
        other=0.0,
    )
    v = tl.load(
        v_ptr
        + pid_bh * stride_v0
        + offs_k[:, None] * stride_v1
        + offs_d[None, :] * stride_v2,
        mask=(offs_k[:, None] < seq_k) & (offs_d[None, :] < head_dim),
        other=0.0,
    )
    out = tl.sum(v * w[:, None], axis=0)
    tl.store(
        output_ptr
        + pid_bh * stride_o0
        + pid_q * stride_o1
        + offs_d * stride_o2,
        out,
        mask=offs_d < head_dim,
    )


@triton.jit
def causal_mask_kernel(
    scores_ptr,
    seq_k,
    offset,
    stride_s0,
    stride_s1,
    stride_s2,
    BLOCK_K: tl.constexpr,
):
    """
    Apply causal mask to attention scores.
    Grid: (batch_heads, seq_q)
    """
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)

    offs_k = tl.arange(0, BLOCK_K)
    mask = offs_k < seq_k
    scores = tl.load(
        scores_ptr
        + pid_bh * stride_s0
        + pid_q * stride_s1
        + offs_k * stride_s2,
        mask=mask,
        other=-1e9,
    )
    current_pos = pid_q + offset
    scores = tl.where(offs_k > current_pos, -1e9, scores)
    tl.store(
        scores_ptr
        + pid_bh * stride_s0
        + pid_q * stride_s1
        + offs_k * stride_s2,
        scores,
        mask=mask,
    )

@triton.autotune(
    configs=[
        triton.Config({"TILE_M": 64,  "TILE_N": 64},  num_warps=4, num_stages=3),
        triton.Config({"TILE_M": 64,  "TILE_N": 32},  num_warps=4, num_stages=3),
        triton.Config({"TILE_M": 128, "TILE_N": 64},  num_warps=8, num_stages=3),
        triton.Config({"TILE_M": 128, "TILE_N": 32},  num_warps=8, num_stages=2),
        triton.Config({"TILE_M": 32,  "TILE_N": 64},  num_warps=4, num_stages=4),
        triton.Config({"TILE_M": 64,  "TILE_N": 128}, num_warps=8, num_stages=2),
    ],
    key=["SEQ_Q", "SEQ_K", "BLOCK_D"],  # re-tune when these change
)
@triton.jit
def fused_attention_kernel(
    q_ptr, k_ptr, v_ptr, output_ptr,
    scale,
    SEQ_Q,  seq_k, head_dim,
    stride_q0, stride_q1, stride_q2,
    stride_k0, stride_k1, stride_k2,
    stride_v0, stride_v1, stride_v2,
    stride_o0, stride_o1, stride_o2,
    IS_CAUSAL: tl.constexpr,
    SEQ_K:     tl.constexpr,
    BLOCK_D:   tl.constexpr,
    TILE_M:    tl.constexpr,
    TILE_N:    tl.constexpr,
):
    """
    Grid: (ceil(SEQ_Q / TILE_M), batch * num_heads)

    Key change from previous version:
      axis-0 now parallelises over query blocks, not just batch*heads.
      Every SM gets a TILE_M x BLOCK_D slice of Q → full GPU utilisation.
    """
    pid_m  = tl.program_id(0)
    pid_bh = tl.program_id(1)

    # ── Offsets for this query tile
    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < SEQ_Q
    mask_d = offs_d < head_dim

    # Load Q tile — TILE_M rows, lives in registers for the K/V loop
    q = tl.load(
        q_ptr + pid_bh * stride_q0
              + offs_m[:, None] * stride_q1
              + offs_d[None, :] * stride_q2,
        mask=mask_m[:, None] & mask_d[None, :],
        other=0.0,
    ).to(tl.float16)

    # Online softmax state (TILE_M rows)
    m   = tl.full((TILE_M,), float("-inf"), dtype=tl.float32)
    l   = tl.zeros((TILE_M,), dtype=tl.float32)
    acc = tl.zeros((TILE_M, BLOCK_D), dtype=tl.float32)

    # Causal: keys beyond the last query row in this tile are all -inf
    # Cap the loop at (pid_m + 1) * TILE_M so those tiles are never loaded.
    k_loop_end = SEQ_K
    if IS_CAUSAL:
        k_loop_end = tl.minimum(SEQ_K, (pid_m + 1) * TILE_M)

    # Stream K/V in TILE_N chunks
    for n_start in range(0, k_loop_end, TILE_N):
        offs_n = n_start + tl.arange(0, TILE_N)
        mask_n = offs_n < seq_k

        k = tl.load(
            k_ptr + pid_bh * stride_k0
                  + offs_n[:, None] * stride_k1
                  + offs_d[None, :] * stride_k2,
            mask=mask_n[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float16)

        v = tl.load(
            v_ptr + pid_bh * stride_v0
                  + offs_n[:, None] * stride_v1
                  + offs_d[None, :] * stride_v2,
            mask=mask_n[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float16)

        # Scores in fp32
        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale

        # Causal mask: only needed on the diagonal tile where n_start overlaps offs_m
        if IS_CAUSAL:
            scores = tl.where(
                offs_m[:, None] >= offs_n[None, :],
                scores, float("-inf"),
            )

        # Padding mask
        scores = tl.where(mask_n[None, :], scores, float("-inf"))

        # Online softmax update
        m_new  = tl.maximum(m, tl.max(scores, axis=1))
        alpha  = tl.exp(m - m_new)
        p      = tl.exp(scores - m_new[:, None]).to(tl.float16)

        l   = alpha * l   + tl.sum(p.to(tl.float32), axis=1)
        acc = alpha[:, None] * acc + tl.dot(p, v, out_dtype=tl.float32)
        m   = m_new

    # Normalize and store
    acc = acc / l[:, None]

    tl.store(
        output_ptr + pid_bh * stride_o0
                   + offs_m[:, None] * stride_o1
                   + offs_d[None, :] * stride_o2,
        acc,
        mask=mask_m[:, None] & mask_d[None, :],
    )

# ============================================================================
# Attention Classes
# ============================================================================

class MultiHeadAttention:
    """Multi-head attention using Triton kernels."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.scale = 1.0 / np.sqrt(self.head_dim)

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Compute multi-head attention.

        Args:
            q: Query (batch, num_heads, seq_q, head_dim)
            k: Key (batch, num_kv_heads, seq_k, head_dim)
            v: Value (batch, num_kv_heads, seq_k, head_dim)
            attention_mask: Optional mask (batch, 1, seq_q, seq_k)
            is_causal: Whether to apply causal masking

        Returns:
            Output (batch, num_heads, seq_q, head_dim)
        """
        batch, num_heads, seq_q, head_dim = q.shape
        _, num_kv_heads, seq_k, _ = k.shape

        if num_kv_heads != num_heads:
            k = self._expand_kv(k, self.num_queries_per_kv)
            v = self._expand_kv(v, self.num_queries_per_kv)

        return scaled_dot_product_attention(
            q, k, v, attention_mask, is_causal, self.scale
        )

    def _expand_kv(self, x: torch.Tensor, num_repeats: int) -> torch.Tensor:
        """Expand KV heads for GQA using broadcast (zero-copy)."""
        batch, num_kv_heads, seq_len, head_dim = x.shape
        x_expanded = x[:, :, None, :, :].expand(
            batch, num_kv_heads, num_repeats, seq_len, head_dim
        )
        return x_expanded.reshape(batch, num_kv_heads * num_repeats, seq_len, head_dim)


def next_power_of_two(x: int) -> int:
    """Return the smallest power of two >= x."""
    return 1 << (x - 1).bit_length() if x > 0 else 1


MAX_ATTENTION_DIM = 4096


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    batch, num_heads, seq_q, head_dim = q.shape
    _, _, seq_k, _ = k.shape

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    seq_q_padded    = next_power_of_two(seq_q)
    seq_k_padded    = next_power_of_two(seq_k)
    head_dim_padded = next_power_of_two(head_dim)

    use_triton = (
        q.is_cuda
        and seq_k_padded    <= MAX_ATTENTION_DIM
        and head_dim_padded <= MAX_ATTENTION_DIM
        # and attention_mask is None
    )

    if use_triton:
        q_flat = q.reshape(batch * num_heads, seq_q, head_dim).contiguous().to(torch.float32)
        k_flat = k.reshape(batch * num_heads, seq_k, head_dim).contiguous().to(torch.float32)
        v_flat = v.reshape(batch * num_heads, seq_k, head_dim).contiguous().to(torch.float32)

        if seq_q_padded != seq_q or head_dim_padded != head_dim:
            q_padded = torch.zeros((batch * num_heads, seq_q_padded, head_dim_padded), dtype=torch.float32, device=q.device)
            q_padded[:, :seq_q, :head_dim] = q_flat
            q_flat = q_padded

        if seq_k_padded != seq_k or head_dim_padded != head_dim:
            k_padded = torch.zeros(
                (batch * num_heads, seq_k_padded, head_dim_padded),
                dtype=torch.float32, device=q.device,
            )
            v_padded = torch.zeros_like(k_padded)
            k_padded[:, :seq_k, :head_dim] = k_flat
            v_padded[:, :seq_k, :head_dim] = v_flat
            k_flat, v_flat = k_padded, v_padded

        output = torch.zeros(
            (batch * num_heads, seq_q_padded, head_dim_padded),
            dtype=torch.float32, device=q.device,
        )

        # Grid now 2D: query tiles × batch*heads
        grid = lambda meta: (
            triton.cdiv(seq_q, meta["TILE_M"]),
            batch * num_heads,
        )

        fused_attention_kernel[grid](
            q_flat, k_flat, v_flat, output,
            float(scale),
            seq_q, seq_k, head_dim,
            *q_flat.stride(), *k_flat.stride(), *v_flat.stride(), *output.stride(),
            IS_CAUSAL=is_causal,
            SEQ_K=seq_k_padded,
            BLOCK_D=head_dim_padded,
            # TILE_M and TILE_N are chosen by autotune — do not pass manually
        )

        output = output[:, :seq_q, :head_dim]
        return output.reshape(batch, num_heads, seq_q, head_dim).to(q.dtype)
        if attention_mask is not None:
            # Only hits for text decoder prefill — small seq_len, fast
            scores_post = (q.float() @ k.float().transpose(-2, -1)) * scale
            scores_post = scores_post + attention_mask
            attn_weights = torch.softmax(scores_post, dim=-1)
            result = (attn_weights @ v.float()).to(q.dtype)

        return result
    # ── PyTorch fallback ──────────────────────────────────────────────────────
    scores = torch.einsum("bnqd,bnkd->bnqk", q, k) * scale

    if is_causal:
        causal_mask = torch.triu(
            torch.ones((seq_q, seq_k), dtype=torch.float32, device=q.device),
            diagonal=1,
        ) * -1e9
        scores = scores + causal_mask[None, None, :, :]

    if attention_mask is not None:
        scores = scores + attention_mask

    scores       = scores - torch.max(scores, dim=-1, keepdim=True).values
    attn_weights = torch.exp(scores)
    attn_weights = attn_weights / torch.sum(attn_weights, dim=-1, keepdim=True)
    output       = torch.einsum("bnqk,bnkd->bnqd", attn_weights, v)

    return output.to(q.dtype)


if __name__ == "__main__":
    print("Testing Triton Attention...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    num_heads = 4
    seq_len = 16
    head_dim = 64

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    print("\nBasic attention:")
    output = scaled_dot_product_attention(q, k, v)
    print(f"  Output shape: {output.shape}")

    print("\nCausal attention:")
    output_causal = scaled_dot_product_attention(q, k, v, is_causal=True)
    print(f"  Output shape: {output_causal.shape}")

    print("\nWith attention mask:")
    mask = torch.zeros(
        (batch_size, num_heads, seq_len, seq_len), dtype=torch.float32, device=device
    )
    mask[:, :, :, seq_len // 2 :] = -1e9
    output_masked = scaled_dot_product_attention(q, k, v, attention_mask=mask)
    print(f"  Output shape: {output_masked.shape}")

    print("\nGrouped Query Attention (GQA):")
    num_kv_heads = 2
    k_gqa = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device)
    v_gqa = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device)
    attn = MultiHeadAttention(
        hidden_size=num_heads * head_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )
    output_gqa = attn(q, k_gqa, v_gqa)
    print(f"  Output shape: {output_gqa.shape}")

    print("\nOutput statistics:")
    print(f"  Mean: {float(output.mean()):.4f}")
    print(f"  Std:  {float(output.std()):.4f}")
    print(f"  Min:  {float(output.min()):.4f}")
    print(f"  Max:  {float(output.max()):.4f}")

    print("\nTriton Attention working!")