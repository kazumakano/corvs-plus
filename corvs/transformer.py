from typing import Callable, Literal, Optional
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules import activation as act
from torch.types import Device
from torchtune import modules
from corvs.embedding import RotaryPositionalEmbeddings


class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    """
    Modified from `torch.nn.TransformerEncoderLayer`.
    This class supports SwiGLU and RMSNorm.
    """

    def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            activation: Literal["relu", "gelu", "swiglu"] | Callable[[torch.FloatTensor], torch.FloatTensor] = F.relu,
            norm: Literal["layer", "rms"] = "layer",
            norm_eps: float = 1e-5,
            batch_first: bool = False,
            norm_first: bool = False,
            bias: bool = True,
            device: Device = None,
            dtype: Optional[torch.device] = None
        ) -> None:

        if activation == "swiglu":
            super().__init__(d_model, nhead, dim_feedforward, dropout, layer_norm_eps=norm_eps, batch_first=batch_first, norm_first=norm_first, bias=bias, device=device, dtype=dtype)

            self.activation = "swiglu"
            self.activation_relu_or_gelu = 0
            self.ff = modules.FeedForward(
                nn.Linear(d_model, dim_feedforward, bias=bias, device=device, dtype=dtype),
                nn.Linear(dim_feedforward, d_model, bias=bias, device=device, dtype=dtype),
                up_proj=nn.Linear(d_model, dim_feedforward, bias=bias, device=device, dtype=dtype)
            )
            del self.linear1, self.dropout, self.linear2
        else:
            super().__init__(d_model, nhead, dim_feedforward, dropout, activation, norm_eps, batch_first, norm_first, bias, device, dtype)

        if norm == "rms":
            self.norm1 = nn.RMSNorm(d_model, eps=norm_eps, device=device, dtype=dtype)
            self.norm2 = nn.RMSNorm(d_model, eps=norm_eps, device=device, dtype=dtype)

    def _ff_block(self, x: torch.FloatTensor) -> torch.FloatTensor:
        if self.activation == "swiglu":
            return self.dropout2(self.ff(x))
        else:
            return super()._ff_block(x)

class RotaryMultiheadAttention(nn.MultiheadAttention):
    """
    Modidied from `torch.nn.MultiheadAttention`.
    This class supports RoPE.
    """

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            time_len: int = 4096,
            dropout: float = 0.0,
            bias: bool = True,
            add_bias_kv: bool = False,
            add_zero_attn: bool = False,
            kdim: Optional[int] = None,
            vdim: Optional[int] = None,
            rope_base: int = 10000,
            batch_first: bool = False,
            device: Device = None,
            dtype: Optional[torch.device] = None
        ) -> None:

        super().__init__(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim, batch_first, device, dtype)
        self.rope = RotaryPositionalEmbeddings(embed_dim // num_heads, time_len, rope_base)

    def forward(
            self,
            query: torch.FloatTensor,
            key: torch.FloatTensor,
            value: torch.FloatTensor,
            key_padding_mask: Optional[torch.BoolTensor | torch.FloatTensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[torch.BoolTensor | torch.FloatTensor] = None,
            average_attn_weights: bool = True,
            is_causal: bool = False
        ) -> tuple[torch.FloatTensor, torch.FloatTensor | None]:

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
                    raise NotImplementedError("naive multi head attention is not implemented")

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

    def _reset_parameters(self) -> None:
        super()._reset_parameters()
        if hasattr(self, "rope"):
            self.rope.rope_init()

class RoFormerEncoderLayer(TransformerEncoderLayer):
    def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward: int = 2048,
            time_len: int = 4096,
            dropout: float = 0.1,
            activation: Literal["relu", "gelu", "swiglu"] | Callable[[torch.FloatTensor], torch.FloatTensor] = F.relu,
            norm: Literal["layer", "rms"] = "layer",
            norm_eps: float = 1e-5,
            rope_base: int = 10000,
            batch_first: bool = False,
            norm_first: bool = False,
            bias: bool = True,
            device: Device = None,
            dtype: Optional[torch.device] = None
        ) -> None:

        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, norm, norm_eps, batch_first, norm_first, bias, device, dtype)
        self.self_attn = RotaryMultiheadAttention(d_model, nhead, time_len, dropout, bias, rope_base=rope_base, batch_first=batch_first, device=device, dtype=dtype)
