"""
Triton Rotary Position Embeddings (RoPE)
End-to-end implementation using Triton kernels

Optimizations over baseline:
  1. Fused apply_rope_kernel — replaces _apply_rope_single (pure PyTorch) with a
     single Triton kernel, eliminating 6+ intermediate tensor allocations and
     separate CUDA kernel launches per op (slice, expand, mul, sub, add, cat).
  2. num_warps tuned per kernel: memory-bound freqs kernel uses 2, rotation kernel uses 4.
  3. Removed redundant .contiguous() copies in apply_rotary_pos_emb.
  4. Skip float32 cast when tensor is already float32.
  5. cos/sin sliced with a view (no copy) before being passed to the kernel.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


def get_stream():
    """Get current CUDA stream pointer."""
    if torch.cuda.is_available():
        return torch.cuda.current_stream().cuda_stream
    return None


# ============================================================================
# Triton Kernels for RoPE
# ============================================================================

@triton.jit
def compute_freqs_kernel(
    positions_ptr,
    inv_freq_ptr,
    cos_ptr,
    sin_ptr,
    seq_len,
    half_dim,
    stride_pos,
    stride_inv,
    stride_cos0,
    stride_cos1,
    stride_sin0,
    stride_sin1,
    BLOCK: tl.constexpr,
):
    """
    Compute cos and sin for rotary embeddings.
    Grid: (seq_len,) — each program handles one sequence position.

    Memory access pattern: memory-bandwidth-bound.
      Reads  inv_freq [half_dim] once per position.
      Writes cos/sin  [rotary_dim] = 2 * half_dim stores per position.
      Arithmetic intensity ≈ (half_dim * 2 flops) / (half_dim * 3 * 4 bytes)
                           ≈ 0.17 FLOP/byte → well below GPU ridge point.
    """
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < half_dim

    # Load scalar position and inverse frequencies
    pos = tl.load(positions_ptr + pid * stride_pos)
    inv = tl.load(inv_freq_ptr + offs * stride_inv, mask=mask, other=0.0)
    freqs = pos * inv

    cos_half = tl.cos(freqs)
    sin_half = tl.sin(freqs)

    # Store into both halves of the cache row (standard RoPE duplication trick:
    # cos[s, :half]  = cos[s, half:] = cos(pos * inv_freq)  so that the
    # rotation formula x1*cos - x2*sin, x2*cos + x1*sin can be applied
    # element-wise without reshaping)
    tl.store(cos_ptr + pid * stride_cos0 + offs * stride_cos1, cos_half, mask=mask)
    tl.store(
        cos_ptr + pid * stride_cos0 + (offs + half_dim) * stride_cos1,
        cos_half,
        mask=mask,
    )
    tl.store(sin_ptr + pid * stride_sin0 + offs * stride_sin1, sin_half, mask=mask)
    tl.store(
        sin_ptr + pid * stride_sin0 + (offs + half_dim) * stride_sin1,
        sin_half,
        mask=mask,
    )


@triton.jit
def apply_rope_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    num_heads,
    seq_len,
    half_dim,
    head_dim,
    stride_xb,
    stride_xh,
    stride_xs,
    stride_xd,
    stride_cs,
    stride_cd,
    stride_ob,
    stride_oh,
    stride_os,
    stride_od,
    BLOCK_D: tl.constexpr,
):
    """
    Fused rotary embedding application kernel.
    Grid: (batch * num_heads * seq_len,)

    Replaces _apply_rope_single entirely. Instead of:
      x1 = x[..., :half]          # slice  → new tensor
      x2 = x[..., half:2*half]    # slice  → new tensor
      cos_expanded = cos[None, None, :, :]   # expand → new tensor
      x1_rot = x1 * cos - x2 * sin           # mul/sub → new tensor
      x2_rot = x2 * cos + x1 * sin           # mul/add → new tensor
      torch.cat([x1_rot, x2_rot], dim=-1)    # cat    → new tensor
    ... (6 separate CUDA kernel launches, 4 temporary allocations)

    This kernel does everything in a single pass with zero intermediate tensors.

    Memory access pattern: memory-bandwidth-bound.
      Reads  x [head_dim], cos/sin [half_dim] per (b, h, s) row.
      Writes out [head_dim] per row.
      Arithmetic intensity ≈ 6 * half_dim flops / (3 * head_dim * 4 bytes)
                           ≈ ~7-8 FLOP/byte → still below ridge point (~200).
    """
    pid = tl.program_id(0)

    # Decode flat index → (batch, head, seq) position
    b = pid // (num_heads * seq_len)
    h = (pid % (num_heads * seq_len)) // seq_len
    s = pid % seq_len

    offs = tl.arange(0, BLOCK_D)
    rot_mask = offs < half_dim  # only rotate the first half_dim elements

    # Base pointers for this (b, h, s) row
    x_base   = x_ptr   + b * stride_xb + h * stride_xh + s * stride_xs
    out_base = out_ptr + b * stride_ob  + h * stride_oh  + s * stride_os

    # Load x1 = x[..., :half_dim] and x2 = x[..., half_dim:2*half_dim]
    x1 = tl.load(x_base + offs * stride_xd,              mask=rot_mask, other=0.0)
    x2 = tl.load(x_base + (offs + half_dim) * stride_xd, mask=rot_mask, other=0.0)

    # Load cos/sin for this sequence position (shape: [half_dim])
    # cos_ptr already has the duplicated layout from compute_freqs_kernel,
    # but here we only need the first half (we apply rotation directly).
    cos_v = tl.load(cos_ptr + s * stride_cs + offs * stride_cd, mask=rot_mask, other=1.0)
    sin_v = tl.load(sin_ptr + s * stride_cs + offs * stride_cd, mask=rot_mask, other=0.0)

    # Apply rotation in-register — no memory traffic for intermediate values
    x1_rot = x1 * cos_v - x2 * sin_v
    x2_rot = x2 * cos_v + x1 * sin_v

    # Store rotated first and second halves
    tl.store(out_base + offs * stride_od,              x1_rot, mask=rot_mask)
    tl.store(out_base + (offs + half_dim) * stride_od, x2_rot, mask=rot_mask)

    # Pass-through: copy dims beyond 2*half_dim unchanged (partial RoPE case)
    pass_offs = offs + half_dim * 2
    pass_mask = pass_offs < head_dim
    x_pass = tl.load(x_base + pass_offs * stride_xd, mask=pass_mask, other=0.0)
    tl.store(out_base + pass_offs * stride_od, x_pass, mask=pass_mask)


# ============================================================================
# RoPE Classes
# ============================================================================

class RotaryEmbedding:
    """Rotary Position Embedding using Triton."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 8192,
        base: float = 10000.0,
        partial_rotary_factor: float = 1.0,
    ):
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.partial_rotary_factor = partial_rotary_factor

        self.rotary_dim = int(dim * partial_rotary_factor)
        self.rotary_dim = self.rotary_dim - (self.rotary_dim % 2)

        inv_freq = 1.0 / (
            base ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim)
        )
        self.inv_freq = inv_freq

        self._update_cache(max_position_embeddings)

    def _update_cache(self, seq_len: int, device: Optional[torch.device] = None):
        """Pre-compute cos and sin using Triton kernel."""
        self.max_seq_len_cached = seq_len
        half_dim = self.rotary_dim // 2
        if device is None:
            device = self.inv_freq.device

        positions = torch.arange(seq_len, dtype=torch.float32, device=device)
        cos_cache = torch.empty((seq_len, self.rotary_dim), dtype=torch.float32, device=device)
        sin_cache = torch.empty((seq_len, self.rotary_dim), dtype=torch.float32, device=device)

        if device.type == "cuda":
            if self.inv_freq.device != device:
                self.inv_freq = self.inv_freq.to(device)

            # next_power_of_2 ensures BLOCK >= half_dim so the mask covers all elements.
            # compute_freqs_kernel is memory-bound → num_warps=2 reduces scheduling
            # overhead without hurting throughput.
            block = triton.next_power_of_2(half_dim)
            compute_freqs_kernel[(seq_len,)](
                positions,
                self.inv_freq,
                cos_cache,
                sin_cache,
                seq_len,
                half_dim,
                positions.stride(0),
                self.inv_freq.stride(0),
                cos_cache.stride(0),
                cos_cache.stride(1),
                sin_cache.stride(0),
                sin_cache.stride(1),
                BLOCK=block,
                num_warps=2,
            )
        else:
            # CPU fallback — plain PyTorch, no Triton
            if self.inv_freq.device != device:
                self.inv_freq = self.inv_freq.to(device)
            freqs = positions[:, None] * self.inv_freq[None, :]
            cos_half = torch.cos(freqs)
            sin_half = torch.sin(freqs)
            cos_cache[:, :half_dim] = cos_half
            cos_cache[:, half_dim : half_dim * 2] = cos_half
            sin_cache[:, :half_dim] = sin_half
            sin_cache[:, half_dim : half_dim * 2] = sin_half

        self.cos_cached = cos_cache
        self.sin_cached = sin_cache

    def __call__(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cos and sin for given positions."""
        seq_len = x.shape[-2]

        if seq_len > self.max_seq_len_cached:
            self._update_cache(seq_len, device=x.device)
        elif self.cos_cached.device != x.device:
            self._update_cache(self.max_seq_len_cached, device=x.device)

        if position_ids is not None:
            cos = self.cos_cached[position_ids].to(x.dtype)
            sin = self.sin_cached[position_ids].to(x.dtype)
            if cos.ndim == 3 and cos.shape[0] == 1:
                cos = cos[0]
                sin = sin[0]
        else:
            # View-based slice — no tensor copy
            cos = self.cos_cached[:seq_len].to(x.dtype)
            sin = self.sin_cached[:seq_len].to(x.dtype)

        return cos, sin


def next_power_of_two(x: int) -> int:
    """Return the smallest power of two >= x."""
    return 1 << (x - 1).bit_length() if x > 0 else 1


MAX_ROPE_DIM = 256


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to Q and K via fused Triton kernel.

    Replaces the old _apply_rope_single (pure PyTorch) with apply_rope_kernel,
    which fuses all 6 ops (slice, expand, mul, sub, add, cat) into a single
    kernel launch with zero intermediate tensors.

    Args:
        q:          [batch, num_q_heads,  seq_len, head_dim]
        k:          [batch, num_kv_heads, seq_len, head_dim]
        cos:        [seq_len, rotary_dim]  (from RotaryEmbedding.__call__)
        sin:        [seq_len, rotary_dim]
        rotary_dim: how many dims to rotate (default: full head_dim)

    Returns:
        q_out, k_out — same shape and dtype as inputs, with RoPE applied.
    """
    batch, num_q_heads, seq_len, head_dim = q.shape
    _, num_kv_heads, _, _ = k.shape

    if rotary_dim is None:
        rotary_dim = head_dim

    half_dim = rotary_dim // 2

    # Trim cos/sin to half_dim using a view (no copy) — the kernel only
    # needs the first half since it loads x1 and x2 separately.
    if cos.shape[1] > half_dim:
        cos = cos[:, :half_dim]
        sin = sin[:, :half_dim]

    # Only cast if not already float32 — avoids a redundant memory pass
    if cos.dtype != torch.float32:
        cos = cos.float()
        sin = sin.float()

    # BLOCK_D must be >= head_dim so the pass-through mask covers all dims
    BLOCK_D = triton.next_power_of_2(head_dim)

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    # Apply RoPE to Q
    # Grid = one program per (batch, head, seq) row
    apply_rope_kernel[batch * num_q_heads * seq_len,](
        q, cos, sin, q_out,
        num_q_heads, seq_len, half_dim, head_dim,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        cos.stride(0), cos.stride(1),
        q_out.stride(0), q_out.stride(1), q_out.stride(2), q_out.stride(3),
        BLOCK_D=BLOCK_D,
        num_warps=4,  # rotation has more arithmetic than freqs → 4 warps better
    )

    # Apply RoPE to K (separate grid for GQA: num_kv_heads may differ from num_q_heads)
    apply_rope_kernel[batch * num_kv_heads * seq_len,](
        k, cos, sin, k_out,
        num_kv_heads, seq_len, half_dim, head_dim,
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        cos.stride(0), cos.stride(1),
        k_out.stride(0), k_out.stride(1), k_out.stride(2), k_out.stride(3),
        BLOCK_D=BLOCK_D,
        num_warps=4,
    )

    return q_out.to(q.dtype), k_out.to(k.dtype)


def apply_partial_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to partial dimensions."""
    return apply_rotary_pos_emb(q, k, cos, sin, rotary_dim)


if __name__ == "__main__":
    print("Testing Triton RoPE...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    num_heads = 4
    seq_len = 16
    head_dim = 64

    rope = RotaryEmbedding(dim=head_dim, max_position_embeddings=1024)

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    cos, sin = rope(q)
    print(f"Cos shape: {cos.shape}")
    print(f"Sin shape: {sin.shape}")

    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    print(f"Q rotated shape: {q_rot.shape}")
    print(f"K rotated shape: {k_rot.shape}")

    print("\nTesting partial RoPE (50%):")
    rope_partial = RotaryEmbedding(dim=head_dim, partial_rotary_factor=0.5)
    cos_p, sin_p = rope_partial(q)
    q_rot_p, k_rot_p = apply_partial_rotary_pos_emb(q, k, cos_p, sin_p, head_dim // 2)
    print(f"Q rotated (partial) shape: {q_rot_p.shape}")

    # ------------------------------------------------------------------
    # Correctness check: compare fused Triton kernel against PyTorch ref
    # ------------------------------------------------------------------
    print("\nRunning correctness check vs PyTorch reference...")

    half_dim = head_dim // 2
    c = cos[:, :half_dim].float()[None, None]  # [1,1,S,half_dim]
    s_ = sin[:, :half_dim].float()[None, None]
    q_f = q.float()
    k_f = k.float()
    q_ref = torch.cat([q_f[..., :half_dim]*c - q_f[..., half_dim:half_dim*2]*s_,
                       q_f[..., half_dim:half_dim*2]*c + q_f[..., :half_dim]*s_], dim=-1)
    k_ref = torch.cat([k_f[..., :half_dim]*c - k_f[..., half_dim:half_dim*2]*s_,
                       k_f[..., half_dim:half_dim*2]*c + k_f[..., :half_dim]*s_], dim=-1)

    q_err = (q_ref - q_rot.float()).abs().max().item()
    k_err = (k_ref - k_rot.float()).abs().max().item()
    print(f"  Q max abs error: {q_err:.2e}")
    print(f"  K max abs error: {k_err:.2e}")
    assert q_err < 1e-4, f"Q error too large: {q_err}"
    assert k_err < 1e-4, f"K error too large: {k_err}"
    print("  PASS")

    print("\nTriton RoPE working!")