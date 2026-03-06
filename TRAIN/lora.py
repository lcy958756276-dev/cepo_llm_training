import torch
import torch.nn as nn
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
import copy
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
import torch.nn.functional as F
class LoraLinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,      # 原来的线性层
        r: int = 8,                 # lora rank
        alpha: int = 16,            # lora alpha
        dropout_p: float = 0.0,     # lora dropout
        test_mode: bool = False,    # 测试模式，用于控制 lora_B 是否为全零
    ):
        super(LoraLinear, self).__init__()
        self.base_layer = copy.deepcopy(base_layer)
        self.r = r
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout_p)

        # 定义 lora_A 和 lora_B 为 Parameter
        self.lora_A = nn.Parameter(torch.empty((r, base_layer.in_features), dtype=base_layer.weight.dtype))
        self.lora_B = nn.Parameter(torch.empty((base_layer.out_features, r), dtype=base_layer.weight.dtype))

        # 初始化 lora 矩阵
        nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
        if test_mode:
            nn.init.normal_(self.lora_B, mean=0.0, std=0.02)
        else:
            nn.init.zeros_(self.lora_B)

        # 冻结原来的层的参数
        for param in self.base_layer.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaling = float(self.alpha) / float(self.r)     # lora 缩放系数
        lora_adjustment = F.linear(self.dropout(x), self.lora_A)
        lora_adjustment = F.linear(lora_adjustment, self.lora_B)
        return self.base_layer(x) + lora_adjustment * scaling


    """
    model.model.layers.0.self_attn.q_proj
    Qwen3ForCausalLM
 ├── model  (Qwen3Model)
 └── lm_head (Linear)
Qwen3Model(
  (embed_tokens): Embedding(151936, 8192)  # 词表大小 151,936，嵌入维度 8192
  (layers): ModuleList(
    (0-39): 40 x Qwen3DecoderLayer(  # 40 层 Transformer Block
      (self_attn): QWEN2_ATTENTION_CLASSES[...](  # 注意力层
        (q_proj): Linear(in_features=8192, out_features=8192, bias=False)
        (k_proj): Linear(in_features=8192, out_features=1024, bias=False)  # 注意：Qwen3 可能使用分组查询注意力 (GQA)
        (v_proj): Linear(in_features=8192, out_features=1024, bias=False)
        (o_proj): Linear(in_features=8192, out_features=8192, bias=False)
        (rotary_emb): Qwen3RotaryEmbedding()  # RoPE 旋转位置编码
      )
      (mlp): Qwen3MLP(  # 前馈网络
        (gate_proj): Linear(in_features=8192, out_features=49152, bias=False)
        (up_proj): Linear(in_features=8192, out_features=49152, bias=False)
        (down_proj): Linear(in_features=49152, out_features=8192, bias=False)
        (act_fn): SiLU()  # SwiGLU 激活函数
      )
      (input_layernorm): Qwen3RMSNorm()  # RMSNorm
      (post_attention_layernorm): Qwen3RMSNorm()
    )
  )
  (norm): Qwen3RMSNorm()
)
    """
def replace_linear_with_lora(
    module: nn.Module,
    r: int = 8,
    alpha: int = 32,
    dropout_p: float = 0.05,
    target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
):
    """
    递归遍历 module 树，将指定名字的 Linear 替换为 LoRALinear。
    """

    for name, child in module.named_children():
        # 如果遇到 Linear 并且名字匹配
        if isinstance(child, nn.Linear) and name in target_modules:
            lora_layer = LoraLinear(child, r=r, alpha=alpha, dropout_p=dropout_p)
            setattr(module, name, lora_layer)

        else:
            # 向下递归
            replace_linear_with_lora(
                child,
                r=r,
                alpha=alpha,
                dropout_p=dropout_p,
                target_modules=target_modules
            )

def print_trainable_parameters(model: nn.Module):
    """
    打印可训练参数，表现和 PeftModel 的 print_trainable_parameters 方法类似
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_percentage = 100 * trainable_params / total_params
  
    # 返回可训练参数量、所有参数量、可训练参数量占比（百分比）
    print(f"trainable params: {trainable_params:,} || all params: {total_params:,} || trainable%: {trainable_percentage:.4f}")


def save_lora(model, path):
    """
    只保存 LoRA 参数
    """
    lora_state = {}

    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            lora_state[name] = param.detach().cpu()

    torch.save(lora_state, path)

    print(f"LoRA weights saved to {path}")
    print(f"LoRA param count: {len(lora_state)}")

def load_lora(model, path, device="cuda"):
    """
    加载 LoRA 参数
    """
    lora_state = torch.load(path, map_location=device)

    model.load_state_dict(lora_state, strict=False)

    print(f"LoRA weights loaded from {path}")