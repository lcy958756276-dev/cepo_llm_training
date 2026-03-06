import sys
import os
#sys.path.append(os.path.dirname(os.path.dirname(__file__)))#把项目的上级目录加入 Python 的模块搜索路径
#__file__ 表示 当前 Python 文件的路径。
#dirname() 取 上一级目录。
#Python 在 import 模块时，会在 sys.path 这些目录里查找。
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
data_path = os.path.join(ROOT_DIR, "data", "train_CEPO.jsonl")
save_dir = os.path.join(ROOT_DIR, "checkpoint")
os.makedirs(save_dir, exist_ok=True)
import torch
import torch.nn as nn
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
import copy
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
import heapq
import pandas as pd
from datasets import Dataset
from lora import replace_linear_with_lora, print_trainable_parameters,save_lora,load_lora
from utils.utils_cepo4 import compute_seq_logprob,cepo_loss_separate
from transformers import BitsAndBytesConfig

top_k = 5
device = "cuda" if torch.cuda.is_available() else "cpu" 
best_checkpoints = []  # 小顶堆 [(loss, path), ...]

def save_checkpoint(model, step, loss):
    global best_checkpoints

    path = os.path.join(save_dir, f"step_{step}.pt")
    save_lora(model, path)

    heapq.heappush(best_checkpoints, (-loss, path))

    # 如果超过 top_k，删除最差的
    if len(best_checkpoints) > top_k:
        worst_loss, worst_path = heapq.heappop(best_checkpoints)
        if os.path.exists(worst_path):
            os.remove(worst_path)

#model_path = "Qwen/Qwen3-8B"
model_path = "Qwen/Qwen2.5-1.5B-Instruct"
#后期可以考虑
#ref_model → 4bit
#policy_model → bf16

# ===== 数据 =====
df = pd.read_json(data_path,lines=True)
ds = Dataset.from_pandas(df)
train_data = ds
train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
save_every=1#验证，先改一下！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
total_step = 0
global_loss = 0.0
num_epochs = 3
logging_steps = 5
max_grad_norm = 1.0
accum_step=4
best_loss = float("inf")
total_norm=0.0

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# ===== policy model（训练用）=====
policy_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# 注入 LoRA
replace_linear_with_lora(
    policy_model,
    r=8,
    alpha=32,
    dropout_p=0.05,
    target_modules=("q_proj", "k_proj", "v_proj", "o_proj")
)


# 冻结所有非 LoRA 参数
for name, p in policy_model.named_parameters():
    if "lora_A" not in name and "lora_B" not in name:
        p.requires_grad = False

print("Policy model:")
print_trainable_parameters(policy_model)


# ===== reference model（完全冻结）=====
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)
ref_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad = False

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, policy_model.parameters()),
    lr=2e-5
)
num_training_steps = len(train_loader) * num_epochs // accum_step
num_warmup_steps = int(0.05 * num_training_steps)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)


policy_model.train()
optimizer.zero_grad()


for epoch in range(num_epochs):

    epoch_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for step, batch in enumerate(pbar):

        sample = {
            "prompt": batch["prompt"][0],
            "chosen_list": batch["chosen_list"][0],
            "reject": batch["reject"][0]
        }
        loss,loss_rank1,loss_eq1 = cepo_loss_separate(
            model=policy_model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            prompt=sample["prompt"],
            good=sample["chosen_list"],
            bad_blocks=sample["reject"],
            beta=1.0,
            lambda_eq=0.1,
            device="cuda"
        )
        print(f"lossrank:{loss_rank1.item():.4f},losseq:{loss_eq1.item():.4f}")
        loss=loss / accum_step
        loss.backward()#计算梯度，不更新参数，梯度会累积accumstep次，直到zero_grad() 在更新后清空梯度
        loss_val = loss.item()*accum_step
        epoch_loss += loss_val
        global_loss += loss_val
        total_step += 1

        if (step+1)%accum_step==0:
            total_norm = torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, policy_model.parameters()),
                max_grad_norm
            )
            optimizer.step()#更新参数
            scheduler.step()
            optimizer.zero_grad()

        if total_step % logging_steps == 0:
            avg_loss = global_loss / total_step
            print(f"Step {total_step} | "f"Avg Loss {avg_loss:.4f} | "f"GradNorm {total_norm:.4f}",flush=True)

            
        # 自动保存 checkpoint
        if total_step>0 and total_step % save_every == 0:
            save_checkpoint(policy_model, total_step, avg_loss)
            print(f"Checkpoint saved at step {total_step}")
    print(f"Epoch {epoch+1} finished | "f"Epoch Avg Loss {epoch_loss/len(train_loader):.4f}",flush=True)
    
