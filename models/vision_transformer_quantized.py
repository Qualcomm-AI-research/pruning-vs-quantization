import torch
import torch.nn as nn
from benchmarks.transformers.vit.vision_transformer import vit_small_patch16_224
from benchmarks.utils.quantization.click_options import QuantSetup
from cirqus.model_repository.interface import enable_repository
from qrunchy.quantization.autoquant_utils import quantize_model
from qrunchy.quantization.base_quantized_classes import QuantizedActivation
from qrunchy.quantization.base_quantized_model import QuantizedModel
from qrunchy.quantization.hijacker import QuantizationHijacker


class QuantizedTensor(QuantizationHijacker):
    def __init__(self, x, *args, activation=None, **kwargs):
        super().__init__(*args, activation=activation, **kwargs)

        self.weight = x

    def get_tensor(self):
        x, _ = self.get_params()
        return x


class QuantizedResidualPositionEmbedding(QuantizedModel):
    def __init__(self, cls_token, pos_embed, **quant_params):
        super(QuantizedResidualPositionEmbedding, self).__init__()

        self.cls_token = QuantizedTensor(cls_token, **quant_params)
        self.pos_embed = QuantizedTensor(pos_embed, **quant_params)
        self.concat_cls_token_act_quantizer = QuantizedActivation(**quant_params)
        self.add_pos_embed_act_quantizer = QuantizedActivation(**quant_params)

    def forward(self, x):
        """
        Parameters
        ----------
        x : (B, num_patches, embed_size)
            Output after the patch embedding layer.
        """
        B = x.size(0)

        cls_tokens = self.cls_token.get_tensor().expand(
            B, -1, -1
        )  # (1, 1, embed_size) -> (B, 1, embed_size)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches + 1, embed_size)
        x = self.concat_cls_token_act_quantizer(x)

        x = x + self.pos_embed.get_tensor()
        x = self.add_pos_embed_act_quantizer(x)
        return x


class QuantizedAttention(QuantizedModel):
    def __init__(self, org_attn, **quant_params):
        super(QuantizedAttention, self).__init__()

        self.num_heads = org_attn.num_heads
        self.scale = org_attn.scale

        self.qkv = quantize_model(org_attn.qkv, **quant_params)
        self.attn_drop = org_attn.attn_drop
        self.proj = quantize_model(org_attn.proj, **quant_params)
        self.proj_drop = org_attn.proj_drop

        self.attn_scores_act_quantizer = QuantizedActivation(**quant_params)
        self.attn_probs_act_quantizer = QuantizedActivation(**quant_params)
        self.attn_out_act_quantizer = QuantizedActivation(**quant_params)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_scores_raw = q.matmul(k.transpose(-2, -1))
        attn_scores = self.scale * attn_scores_raw
        attn_scores = self.attn_scores_act_quantizer(attn_scores)

        attn_probs = attn_scores.softmax(dim=-1)
        attn_probs = self.attn_probs_act_quantizer(attn_probs)

        attn = self.attn_drop(attn_probs)  # Dropout(p=0), does nothing

        x = attn.matmul(v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.attn_out_act_quantizer(x)

        x = self.proj(x)  # output of is already quantized
        x = self.proj_drop(x)  # Dropout(p=0), does nothing
        return x


class QuantizedVisionTransformerBlock(QuantizedModel):
    def __init__(self, org_block, **quant_params):
        super(QuantizedVisionTransformerBlock, self).__init__()

        self.norm1 = quantize_model(org_block.norm1, **quant_params)
        self.attn = QuantizedAttention(org_block.attn, **quant_params)
        self.drop_path = nn.Identity()  # so-called DropPath is also possible here
        self.norm2 = quantize_model(org_block.norm2, **quant_params)

        self.attn_res_act_quantizer = QuantizedActivation(**quant_params)
        self.mlp_res_act_quantizer = QuantizedActivation(**quant_params)

        mlp = org_block.mlp
        self.mlp = quantize_model(
            nn.Sequential(mlp.fc1, mlp.act, mlp.fc2, mlp.drop), **quant_params
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = self.attn_res_act_quantizer(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = self.mlp_res_act_quantizer(x)
        return x


@enable_repository
class QuantizedVisionTransformer(QuantizedModel):
    def __init__(
        self, org_model, input_size=(1, 3, 224, 224), quant_setup=QuantSetup.all, **quant_params
    ):
        super().__init__(input_size)

        self.patch_embed = quantize_model(org_model.patch_embed, **quant_params)
        self.quant_pos_embed = QuantizedResidualPositionEmbedding(
            cls_token=org_model.cls_token, pos_embed=org_model.pos_embed, **quant_params
        )
        self.pos_drop = org_model.pos_drop

        self.blocks = nn.ModuleList(
            [
                QuantizedVisionTransformerBlock(org_model.blocks[i], **quant_params)
                for i in range(len(org_model.blocks))
            ]
        )

        self.norm = quantize_model(org_model.norm, **quant_params)
        self.head = quantize_model(org_model.head, **quant_params)

    def forward_features(self, x):
        x = self.patch_embed(x)  # (B, num_patches, embed_size)
        x = self.quant_pos_embed(x)  # (B, num_patches + 1, embed_size)
        x = self.pos_drop(x)  # Dropout(p=0), does nothing
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def vit_small_patch16_224_quantized(qparams=None, load_type="fp32", pretrained=False, **kwargs):
    fp32_model = vit_small_patch16_224(pretrained=pretrained, **kwargs)

    # Quantize
    print("Quantization parameters:", qparams)
    quant_model = QuantizedVisionTransformer(fp32_model, **qparams)

    return quant_model
