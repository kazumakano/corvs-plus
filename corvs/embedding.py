import math
from typing import Optional
import einops
import torch
from torch import jit
from torch.nn import functional as F
from torch.types import Device
from torchtune import modules


def create_sin_pos_emb(dim: int, seq_len: int, base: float = 10000, device: Device = None) -> torch.FloatTensor:
    """
    Create a sinusoidal position embedding.

    Parameters
    ----------
    dim : int
        Dimension.
    seq_len : int
        Sequence length.
    base : float
        Base frequency.
        Maximum period is `2π * base`.
    device : int | str | device | None
        Computation device.

    Returns
    -------
    emb : FloatTensor
        Position embedding.
        Shape is (seq_len, dim).
    """

    if dim % 2 != 0:
        raise ValueError("dimension must be even")

    freq = (-math.log(base) * torch.arange(0, dim, step=2, dtype=torch.float64, device=device) / dim).exp()    # (dim / 2, )
    pos = torch.arange(seq_len, dtype=torch.float64, device=device).unsqueeze(1)    # (seq_len, 1)

    emb = torch.empty(seq_len, dim, dtype=torch.float32, device=device)
    emb[:, ::2] = torch.sin(freq * pos)
    emb[:, 1::2] = torch.cos(freq * pos)

    return emb

class RotaryPositionalEmbeddings(modules.RotaryPositionalEmbeddings):
    def forward(self, x: torch.FloatTensor, *, input_pos: Optional[torch.IntTensor] = None) -> torch.FloatTensor:    # (batch, seq, head, dim) -> (batch, seq, head, dim)
        """
        Modified from `torchtune.modules.RotaryPositionalEmbeddings.forward()`.
        This method automatically rebuild cache when input is longer than cache.
        """

        seq_len = x.size(1)

        if self.cache.shape[0] < seq_len:
            self.build_rope_cache(max_seq_len=seq_len)

        rope_cache = self.cache[:seq_len] if input_pos is None else self.cache[input_pos]

        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        x_out = torch.stack((xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1], xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1]), dim=-1)

        x_out = x_out.flatten(start_dim=3)
        return x_out.type_as(x)

    def multi_head_attention_forward(
            self,
            query: torch.FloatTensor,
            key: torch.FloatTensor,
            value: torch.FloatTensor,
            embed_dim_to_check: int,
            num_heads: int,
            in_proj_weight: torch.FloatTensor | None,
            in_proj_bias: torch.FloatTensor | None,
            bias_k: torch.FloatTensor | None,
            bias_v: torch.FloatTensor | None,
            add_zero_attn: bool,
            dropout_p: float,
            out_proj_weight: torch.FloatTensor,
            out_proj_bias: torch.FloatTensor | None,
            training: bool = True,
            key_padding_mask: Optional[torch.BoolTensor | torch.FloatTensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[torch.BoolTensor | torch.FloatTensor] = None,
            use_separate_proj_weight: bool = False,
            q_proj_weight: Optional[torch.FloatTensor] = None,
            k_proj_weight: Optional[torch.FloatTensor] = None,
            v_proj_weight: Optional[torch.FloatTensor] = None,
            static_k: Optional[torch.FloatTensor] = None,
            static_v: Optional[torch.FloatTensor] = None,
            average_attn_weights: bool = True,
            is_causal: bool = False
        ) -> tuple[torch.FloatTensor, torch.FloatTensor | None]:
        """
        Modified from `torch.nn.functional.multi_head_attention_forward()`.
        This method applies RoPE to query and key.
        """

        is_batched = F._mha_shape_check(query, key, value, key_padding_mask, attn_mask, num_heads)

        if not is_batched:
            query = query.unsqueeze(1)
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask.unsqueeze(0)

        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape

        key_padding_mask = F._canonical_mask(mask=key_padding_mask, mask_name="key_padding_mask", other_type=F._none_or_dtype(attn_mask), other_name="attn_mask", target_type=query.dtype)

        if is_causal and attn_mask is None:
            raise RuntimeError(
                "Need attn_mask if specifying the is_causal hint. "
                "You may use the Transformer module method "
                "`generate_square_subsequent_mask` to create this mask."
            )

        if is_causal and key_padding_mask is None and not need_weights:
            attn_mask = None
        else:
            attn_mask = F._canonical_mask(mask=attn_mask, mask_name="attn_mask", other_type=None, other_name="", target_type=query.dtype, check_other=False)

            if key_padding_mask is not None:
                is_causal = False

        assert embed_dim == embed_dim_to_check, f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
        if isinstance(embed_dim, torch.Tensor):
            head_dim = embed_dim.div(num_heads, rounding_mode="trunc")
        else:
            head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
        if use_separate_proj_weight:
            assert key.shape[:2] == value.shape[:2], f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
        else:
            assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

        if not use_separate_proj_weight:
            assert in_proj_weight is not None, "use_separate_proj_weight is False but in_proj_weight is None"
            q, k, v = F._in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
        else:
            assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
            assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
            assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
            if in_proj_bias is None:
                b_q = b_k = b_v = None
            else:
                b_q, b_k, b_v = in_proj_bias.chunk(3)
            q, k, v = F._in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                correct_2d_size = tgt_len, src_len
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_3d_size = bsz * num_heads, tgt_len, src_len
                if attn_mask.shape != correct_3d_size:
                    raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
            else:
                raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

        if bias_k is not None and bias_v is not None:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
            k = torch.cat((k, bias_k.repeat(1, bsz, 1)))
            v = torch.cat((v, bias_v.repeat(1, bsz, 1)))
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))
        else:
            assert bias_k is None
            assert bias_v is None

        q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        if static_k is None:
            k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        else:
            assert static_k.size(0) == bsz * num_heads, f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
            assert static_k.size(2) == head_dim, f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
            k = static_k
        if static_v is None:
            v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        else:
            assert static_v.size(0) == bsz * num_heads, f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
            assert static_v.size(2) == head_dim, f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
            v = static_v

        # apply RoPE
        q = einops.rearrange(q, "(b nh) s hd -> b s nh hd", b=bsz, nh=num_heads)
        q = self(q)
        q = einops.rearrange(q, "b s nh hd -> (b nh) s hd")
        k = einops.rearrange(k, "(b nh) s hd -> b s nh hd", b=bsz, nh=num_heads)
        k = self(k)
        k = einops.rearrange(k, "b s nh hd -> (b nh) s hd")

        if add_zero_attn:
            zero_attn_shape = bsz * num_heads, 1, head_dim
            k = torch.cat((k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)), dim=1)
            v = torch.cat((v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)), dim=1)
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))

        src_len = k.size(1)

        if key_padding_mask is not None:
            if not jit.is_scripting() and not jit.is_tracing():
                F._check_key_padding_mask(key_padding_mask, src_len, bsz)

            key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                attn_mask = attn_mask + key_padding_mask

        if not training:
            dropout_p = 0.0

        if need_weights:
            _B, _Nt, E = q.shape
            q_scaled = q * math.sqrt(1.0 / float(E))

            assert not (is_causal and attn_mask is None), "FIXME: is_causal not implemented for need_weights"

            if attn_mask is not None:
                attn_output_weights = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))
            else:
                attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
            attn_output_weights = F.softmax(attn_output_weights, dim=-1)
            if dropout_p > 0.0:
                attn_output_weights = F.dropout(attn_output_weights, p=dropout_p)

            attn_output = torch.bmm(attn_output_weights, v)

            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
            attn_output = F.linear(attn_output, out_proj_weight, bias=out_proj_bias)
            attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            if average_attn_weights:
                attn_output_weights = attn_output_weights.mean(dim=1)

            if not is_batched:
                attn_output = attn_output.squeeze(1)
                attn_output_weights = attn_output_weights.squeeze(0)
            return attn_output, attn_output_weights
        else:
            if attn_mask is not None:
                if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
                    attn_mask = attn_mask.unsqueeze(0)
                else:
                    attn_mask = attn_mask.view(bsz, num_heads, -1, src_len)

            q = q.view(bsz, num_heads, tgt_len, head_dim)
            k = k.view(bsz, num_heads, src_len, head_dim)
            v = v.view(bsz, num_heads, src_len, head_dim)

            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
            attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)

            attn_output = F.linear(attn_output, out_proj_weight, bias=out_proj_bias)
            attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
            if not is_batched:
                attn_output = attn_output.squeeze(1)
            return attn_output, None
