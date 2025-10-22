import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import math
from typing import Optional

class LoRAParameter(nn.Module):
    def __init__(self,orig_param,A_param,B_param,lora_alpha,r):
        super().__init__()
        self.orig_param = orig_param
        self.lora_A_param = A_param
        self.lora_B_param = B_param
        self.lora_alpha = lora_alpha
        self.r = r
        self.scaling = self.lora_alpha / self.r

    def combined_matrix(self):
        """
        组合原始参数矩阵和 LoRA 矩阵，得到一个合并后的矩阵
        """
        lora_contribution = self.lora_A_param.T @ self.lora_B_param.T * self.scaling
        return self.orig_param + lora_contribution


class LoRAMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        lora_r=16,
        num_loras=16,
        lora_alpha=1,
        device=None,
        dtype=None,
    ) -> None:
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be greater than 0,"
                f" got embed_dim={embed_dim} and num_heads={num_heads} instead"
            )
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.lora_alpha = lora_alpha
        self.r = lora_r

        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.in_proj_weight = nn.Parameter(
                torch.empty((3 * embed_dim, embed_dim), **factory_kwargs)
            )
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim, **factory_kwargs))

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True, **factory_kwargs)

        self.loras_A_in = [nn.Parameter(self.in_proj_weight.new_zeros((self.r, 3 * embed_dim))) for _ in range(num_loras)]
        self.loras_B_in = [nn.Parameter(self.in_proj_weight.new_zeros((embed_dim, self.r))) for _ in range(num_loras)]

        self.loras_A_out = [nn.Parameter(self.out_proj.weight.new_zeros((self.r, embed_dim))) for _ in range(num_loras)]
        self.loras_B_out = [nn.Parameter(self.out_proj.weight.new_zeros((embed_dim, self.r))) for _ in range(num_loras)]

        self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix

        self.in_proj_weight_withloras = nn.ModuleList([LoRAParameter(self.in_proj_weight,self.loras_A_in[i],self.loras_B_in[i],self.lora_alpha,self.r) for i in range(num_loras)])
        self.out_proj_weight_withloras = nn.ModuleList([LoRAParameter(self.out_proj.weight,self.loras_A_out[i],self.loras_B_out[i],self.lora_alpha,self.r) for i in range(num_loras)])

        self.in_proj_weight.requires_grad = False
        self.in_proj_bias.requires_grad = False
        self.out_proj.requires_grad = False

        self.reset_parameters()
        self.bias_k = self.bias_v = None
        self.add_zero_attn = False
        self.dropout = 0.0

    def reset_parameters(self):
        if hasattr(self, 'loras_A_in'):
            # initialize B the same way as the default for nn.Linear and A to zero
            # this is different than what is described in the paper but should not affect performance
            for i in range(len(self.loras_A_in)):
                nn.init.kaiming_uniform_(self.loras_A_in[i], a=math.sqrt(5))
                nn.init.zeros_(self.loras_B_in[i])
                nn.init.kaiming_uniform_(self.loras_A_out[i], a=math.sqrt(5))
                nn.init.zeros_(self.loras_B_out[i])

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            key_padding_mask: Optional[torch.Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[torch.Tensor] = None,
            average_attn_weights: bool = True,
            is_causal: bool = False,
            lora_index: int = 0
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        in_proj_weight = self.in_proj_weight_withloras[lora_index].combined_matrix()
        out_proj_weight = self.out_proj_weight_withloras[lora_index].combined_matrix()

        attn_output, attn_output_weights = F.multi_head_attention_forward(
            query,
            key,
            value,
            self.embed_dim,
            self.num_heads,
            in_proj_weight,
            self.in_proj_bias,
            self.bias_k,
            self.bias_v,
            self.add_zero_attn,
            self.dropout,
            out_proj_weight,
            self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )

        return attn_output, attn_output_weights


# 定义 LoRA 模块
class LoRALayer():
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x


class LoRALinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_r: int = 16,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        num_loras: int = 16,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if lora_r > 0:
            self.loras_A = nn.ParameterList([nn.Parameter(self.weight.new_zeros((lora_r, in_features))) for _ in range(num_loras)])
            self.loras_B = nn.ParameterList([nn.Parameter(self.weight.new_zeros((out_features, lora_r))) for _ in range(num_loras)])
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'loras_A'):
            # initialize B the same way as the default for nn.Linear and A to zero
            # this is different than what is described in the paper but should not affect performance
            for lora_A in self.loras_A:
                nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
            for lora_B in self.loras_B:
                nn.init.zeros_(lora_B)

    def forward(self, x: torch.Tensor, lora_index=0):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0:
            result = F.linear(x, T(self.weight), bias=self.bias)
            lora_result = (self.lora_dropout(x) @ self.loras_A[lora_index].transpose(0, 1) @ self.loras_B[lora_index].transpose(0, 1)) * self.scaling
            result = result + lora_result
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlockAllwithLoRA(nn.Module):
    def __init__(self, d_model: int, n_head: int, lora_r: int, num_loras: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = LoRAMultiheadAttention(d_model, n_head,lora_r=lora_r,num_loras=num_loras)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", LoRALinear(d_model, d_model * 4,lora_r=lora_r,num_loras=num_loras)),
            ("gelu", QuickGELU()),
            ("c_proj", LoRALinear(d_model * 4, d_model,lora_r=lora_r,num_loras=num_loras))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor,lora_index):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask,lora_index=lora_index)[0]

    def forward(self, x: torch.Tensor, lora_index=0):
        x = x + self.attention(self.ln_1(x),lora_index)
        x_mlp = self.mlp[0](self.ln_2(x),lora_index)
        x_mlp = self.mlp[1](x_mlp)
        x_mlp = self.mlp[2](x_mlp,lora_index)
        x = x + x_mlp
        return x


class ResidualAttentionBlockMLPwithLoRA(nn.Module):
    def __init__(self, d_model: int, n_head: int, lora_r: int, num_loras: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", LoRALinear(d_model, d_model * 4,lora_r=lora_r,num_loras=num_loras)),
            ("gelu", QuickGELU()),
            ("c_proj", LoRALinear(d_model * 4, d_model,lora_r=lora_r,num_loras=num_loras))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor, lora_index=0):
        x = x + self.attention(self.ln_1(x))
        x_mlp = self.mlp[0](self.ln_2(x),lora_index)
        x_mlp = self.mlp[1](x_mlp)
        x_mlp = self.mlp[2](x_mlp,lora_index)
        x = x + x_mlp
        return x


class ResidualAttentionBlockwithoutLoRA(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None

        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor,lora_index=0):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MMTransformer_withlora(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, lora_r: int=16, num_loras: int=16, lora_layers:int=2, lora_mode:str='mlp'):
        super().__init__()
        self.width = width
        self.layers = layers
        assert lora_layers<=layers
        print('LoRA Mode:{}'.format(lora_mode))
        if lora_mode == 'all':
            self.resblocks = nn.ModuleList([ResidualAttentionBlockwithoutLoRA(width, heads,attn_mask) for _ in range(layers-lora_layers)]+
                                       [ResidualAttentionBlockAllwithLoRA(width, heads, lora_r, num_loras,attn_mask) for _ in range(lora_layers)])
        elif lora_mode == 'mlp':
            self.resblocks = nn.ModuleList(
                [ResidualAttentionBlockwithoutLoRA(width, heads, attn_mask) for _ in range(layers - lora_layers)] +
                [ResidualAttentionBlockMLPwithLoRA(width, heads, lora_r, num_loras, attn_mask) for _ in range(lora_layers)])
        else:
            raise("Error")

    def forward(self, x: torch.Tensor, data_type="RGB"):
        lora_mapping = {"RGB": 0, "NIR": 1, "CP": 2, "SK": 3, "TEXT": 4,
                        "NIR+CP":5,"NIR+SK":6,"NIR+TEXT":7,"CP+SK":8,"CP+TEXT":9,"SK+TEXT":10,
                        "NIR+CP+SK":11,"NIR+CP+TEXT":12,"NIR+SK+TEXT":13,"CP+SK+TEXT":14,
                        "NIR+CP+SK+TEXT":15}
        lora_index = lora_mapping.get(data_type, 0)
        for block in self.resblocks:
            x = block(x, lora_index)
        return x