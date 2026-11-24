"""Memory optimisation for SAM ViT global attention: row-chunked evaluation (runtime patch, no file edits).

Problem (measured peak working set 3.29 GB on the s0965 pilot): in the 4 global-attention blocks of
ViT-B (indexes 2, 5, 8, 11; 64x64 = 4096 tokens, 12 heads) segment_anything's Attention.forward
materialises the full attention matrix [12, 4096, 4096] float32 (805 MB) and, in
add_decomposed_rel_pos, two more full-size temporaries (attn.view + rel_h, then + rel_w), so ~2.4 GB
is alive at once, plus the softmax copy.

Change: for attention matrices >= MIN_ATTN_BYTES only, process query rows in blocks of CHUNK_ROWS.
Per row, the operations are the same and in the same order:
  (q*scale) @ k^T  ->  (+ rel_h) (+ rel_w)  ->  softmax  ->  @ v
rel_h / rel_w are computed exactly as upstream (same einsum on the full q) and then row-sliced.
Windowed blocks (small matrices) use the untouched upstream forward.

This is ONLY acceptable if outputs are bit-identical to upstream; that is verified empirically by
validate_mem_opt.py (encoder features + end-to-end predictions on already-completed slices), never assumed.
Nothing on disk is modified; ``install()`` patches the class in the current process only, ``uninstall()`` reverts.
"""
from __future__ import annotations

import hashlib
import inspect

import torch
from segment_anything.modeling import image_encoder as IE

CHUNK_ROWS = 512
MIN_ATTN_BYTES = 128 * 2 ** 20
_ORIG_FORWARD = IE.Attention.forward
UPSTREAM_SOURCE_SHA256 = hashlib.sha256(inspect.getsource(_ORIG_FORWARD).encode()).hexdigest()
UPSTREAM_RELPOS_SHA256 = hashlib.sha256(inspect.getsource(IE.add_decomposed_rel_pos).encode()).hexdigest()


def _chunked_forward(self, x: torch.Tensor) -> torch.Tensor:
    B, H, W, _ = x.shape
    N = H * W
    if B * self.num_heads * N * N * 4 < MIN_ATTN_BYTES:
        return _ORIG_FORWARD(self, x)
    # --- identical to upstream up to the attention matrix ---
    qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
    qs = q * self.scale
    kt = k.transpose(-2, -1)
    BH = q.shape[0]
    if self.use_rel_pos:  # same computation as upstream add_decomposed_rel_pos, kept full-size (small)
        Rh = IE.get_rel_pos(H, H, self.rel_pos_h)
        Rw = IE.get_rel_pos(W, W, self.rel_pos_w)
        r_q = q.reshape(BH, H, W, q.shape[-1])
        rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh).reshape(BH, N, H)
        rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw).reshape(BH, N, W)
    out = torch.empty((BH, N, v.shape[-1]), dtype=v.dtype, device=v.device)
    for s in range(0, N, CHUNK_ROWS):
        e = min(N, s + CHUNK_ROWS)
        a = qs[:, s:e] @ kt
        if self.use_rel_pos:
            a = (a.view(BH, e - s, H, W) + rel_h[:, s:e, :, None] + rel_w[:, s:e, None, :]).view(BH, e - s, N)
        a = a.softmax(dim=-1)
        out[:, s:e] = a @ v
        del a
    x = out.view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
    return self.proj(x)


def install() -> dict:
    IE.Attention.forward = _chunked_forward
    return status()


def uninstall() -> None:
    IE.Attention.forward = _ORIG_FORWARD


def status() -> dict:
    return {"active": IE.Attention.forward is _chunked_forward, "chunk_rows": CHUNK_ROWS, "min_attn_bytes": MIN_ATTN_BYTES,
            "upstream_attention_forward_sha256": UPSTREAM_SOURCE_SHA256,
            "upstream_add_decomposed_rel_pos_sha256": UPSTREAM_RELPOS_SHA256,
            "patched_forward_sha256": hashlib.sha256(inspect.getsource(_chunked_forward).encode()).hexdigest()}
