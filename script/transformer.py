import math
from typing import Any, Callable, Optional
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules import activation as act
from torchtune import modules


def create_sin_pos_embed(dim: int, time_len: int, max_period: float = 10000) -> torch.Tensor:
    """
    Create a sinusoidal position embedding.

    Parameters
    ----------
    dim : int
        Dimension.
    time_len : int
        Time length.
    max_period : float
        Maximum sinusoidal period.

    Returns
    -------
    embed : Tensor[float32]
        Position embedding.
        Shape is (time_len, dim).
    """

    freq = torch.exp(-math.log(max_period) * torch.arange(0, dim, step=2, dtype=torch.float64) / dim)    # (dim / 2, )
    pos = torch.arange(time_len, dtype=torch.float64).unsqueeze(1)    # (time_len, 1)

    embed = torch.empty(time_len, dim, dtype=torch.float32)
    embed[:, ::2] = torch.sin(freq * pos)
    embed[:, 1::2] = torch.cos(freq[:dim // 2] * pos)

    return embed

class RotaryPositionalEmbeddings(modules.RotaryPositionalEmbeddings):
    def forward(self, x: torch.FloatTensor, input_pos: Optional[torch.IntTensor] = None) -> torch.FloatTensor:
        """
        Modified from `RotaryPositionalEmbeddings.forward()`.
        """

        seq_len = x.size(1)

        rope_cache = self.cache[:seq_len] if input_pos is None else self.cache[input_pos]

        xshaped = x.float().reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2)

        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        x_out = torch.stack([xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1], xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1]], dim=-1)

        x_out = x_out.flatten(start_dim=3)
        return x_out.type_as(x)

    def multi_head_attention_forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, embed_dim_to_check: int, num_heads: int, in_proj_weight: torch.Tensor | None, in_proj_bias: torch.Tensor | None, bias_k: torch.Tensor | None, bias_v: torch.Tensor | None, add_zero_attn: bool, dropout_p: float, out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor | None, training: bool = True, key_padding_mask: Optional[torch.Tensor] = None, need_weights: bool = True, attn_mask: Optional[torch.Tensor] = None, use_separate_proj_weight: bool = False, q_proj_weight: Optional[torch.Tensor] = None, k_proj_weight: Optional[torch.Tensor] = None, v_proj_weight: Optional[torch.Tensor] = None, static_k: Optional[torch.Tensor] = None, static_v: Optional[torch.Tensor] = None, average_attn_weights: bool = True, is_causal: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Modified from `multi_head_attention_forward()`.
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
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
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
        q = self(q.view(bsz, num_heads, tgt_len, head_dim).transpose(1, 2)).transpose(1, 2).contiguous().view(bsz * num_heads, tgt_len, head_dim)
        k = self(k.view(bsz, num_heads, src_len, head_dim).transpose(1, 2)).transpose(1, 2).contiguous().view(bsz * num_heads, src_len, head_dim)

        if add_zero_attn:
            zero_attn_shape = bsz * num_heads, 1, head_dim
            k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
            v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))

        src_len = k.size(1)

        if key_padding_mask is not None:
            if not torch.jit.is_scripting() and not torch.jit.is_tracing():
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
            attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
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

            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal)
            attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)

            attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
            attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
            if not is_batched:
                attn_output = attn_output.squeeze(1)
            return attn_output, None

class RotaryMultiheadAttention(nn.MultiheadAttention):
    def __init__(self, embed_dim: int, num_heads: int, time_len: int, dropout: float = 0.0, bias: bool = True, add_bias_kv: bool = False, add_zero_attn: bool = False, kdim: Optional[int] = None, vdim: Optional[int] = None, batch_first: bool = False, device: Any = None, dtype: Any = None) -> None:
        super().__init__(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim, batch_first, device, dtype)
        self.rope = RotaryPositionalEmbeddings(embed_dim // num_heads, max_seq_len=time_len)

    def forward(self, query: torch.FloatTensor, key: torch.FloatTensor, value: torch.FloatTensor, key_padding_mask: Optional[torch.BoolTensor] = None, need_weights: bool = True, attn_mask: Optional[torch.BoolTensor] = None, average_attn_weights: bool = True, is_causal: bool = False) -> tuple[torch.FloatTensor, torch.FloatTensor | None]:
        """
        Modidied from `MultiheadAttention.forward()`.
        This method applies RoPE to query and key.
        """

        why_not_fast_path = ""
        if attn_mask is not None and torch.is_floating_point(attn_mask) or key_padding_mask is not None and torch.is_floating_point(key_padding_mask):
            why_not_fast_path = "floating-point masks are not supported for fast path."

        is_batched = query.dim() == 3

        key_padding_mask = F._canonical_mask(mask=key_padding_mask, mask_name="key_padding_mask", other_type=F._none_or_dtype(attn_mask), other_name="attn_mask", target_type=query.dtype)

        attn_mask = F._canonical_mask(mask=attn_mask, mask_name="attn_mask", other_type=None, other_name="", target_type=query.dtype, check_other=False)

        is_fastpath_enabled = torch.backends.mha.get_fastpath_enabled()

        if not is_fastpath_enabled:
            why_not_fast_path = "torch.backends.mha.get_fastpath_enabled() was not True"
        elif not is_batched:
            why_not_fast_path = f"input not batched; expected query.dim() of 3 but got {query.dim()}"
        elif query is not key or key is not value:
            why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
        elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
        elif self.in_proj_weight is None:
            why_not_fast_path = "in_proj_weight was None"
        elif query.dtype != self.in_proj_weight.dtype:
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_weight ({self.in_proj_weight.dtype}) don't match"
        elif self.training:
            why_not_fast_path = "training is enabled"
        elif (self.num_heads % 2) != 0:
            why_not_fast_path = "self.num_heads is not even"
        elif not self.batch_first:
            why_not_fast_path = "batch_first was not True"
        elif self.bias_k is not None:
            why_not_fast_path = "self.bias_k was not None"
        elif self.bias_v is not None:
            why_not_fast_path = "self.bias_v was not None"
        elif self.add_zero_attn:
            why_not_fast_path = "add_zero_attn was enabled"
        elif not self._qkv_same_embed_dim:
            why_not_fast_path = "_qkv_same_embed_dim was not True"
        elif query.is_nested and (key_padding_mask is not None or attn_mask is not None):
            why_not_fast_path = "supplying both src_key_padding_mask and src_mask at the same time is not supported with NestedTensor input"
        elif torch.is_autocast_enabled():
            why_not_fast_path = "autocast is enabled"

        if not why_not_fast_path:
            tensor_args = query, key, value, self.in_proj_weight, self.in_proj_bias, self.out_proj.weight, self.out_proj.bias
            if torch.overrides.has_torch_function(tensor_args):
                why_not_fast_path = "some Tensor argument has_torch_function"
            elif act._is_make_fx_tracing():
                why_not_fast_path = "we are running make_fx tracing"
            elif not all(act._check_arg_device(x) for x in tensor_args):
                why_not_fast_path = (
                    "some Tensor argument's device is neither one of "
                    f"cpu, cuda or {torch.utils.backend_registration._privateuse1_backend_name}"
                )
            elif torch.is_grad_enabled() and any(act._arg_requires_grad(x) for x in tensor_args):
                why_not_fast_path = (
                    "grad is enabled and at least one of query or the "
                    "input/output projection weights or biases requires_grad"
                )
            if not why_not_fast_path:
                merged_mask, mask_type = self.merge_masks(attn_mask, key_padding_mask, query)

                if self.in_proj_bias is not None and self.in_proj_weight is not None:
                    return torch._native_multi_head_attention(query, key, value, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.out_proj.weight, self.out_proj.bias, merged_mask, need_weights, average_attn_weights, mask_type)

        any_nested = query.is_nested or key.is_nested or value.is_nested
        assert not any_nested, f"MultiheadAttention does not support NestedTensor outside of its fast path. The fast path was not hit because {why_not_fast_path}"

        if self.batch_first and is_batched:
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = self.rope.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal
            )
        else:
            attn_output, attn_output_weights = self.rope.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal
            )
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights

class RoFormerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model: int, nhead: int, time_len: int, dim_feedforward: int = 2048, dropout: float = 0.1, activation: str | Callable[[torch.Tensor], torch.Tensor] = F.relu, layer_norm_eps: float = 0.00001, batch_first: bool = False, norm_first: bool = False, bias: bool = True, device: Any = None, dtype: Any = None) -> None:
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, batch_first, norm_first, bias, device, dtype)
        self.self_attn = RotaryMultiheadAttention(d_model, nhead, time_len, dropout=dropout, bias=bias, batch_first=batch_first, device=device, dtype=dtype)
