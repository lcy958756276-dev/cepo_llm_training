from datasets import Dataset
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, TaskType, get_peft_model
from utils.utils_cepo2 import cepo_loss_separate
import torch.nn.functional as F

# ================================
# 1️⃣ 自定义 Trainer
# ================================
class CepoTrainer(Trainer):

    def __init__(self, ref_model, tokenizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model
        self.tokenizer = tokenizer

    def compute_loss(self, model, inputs, return_outputs=False):

        total_loss = 0

        # batch 是 list[dict]
        for sample in inputs:

            loss = cepo_loss_separate(
                model=model,
                ref_model=self.ref_model,
                tokenizer=self.tokenizer,
                sample=sample,
                beta=1.0,
                lambda_eq=0.1,
                device=model.device
            )

            total_loss += loss

        total_loss = total_loss / len(inputs)

        return total_loss


# ================================
#2️⃣ 主函数
# ================================
if __name__ == "__main__":

    model_path = "./LLM-Research/Qwen/Qwen3-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # 冻结 reference model
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # LoRA
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )
    model = get_peft_model(model, config)
    print(model.print_trainable_parameters())

    # ===== 数据 =====
    df = pd.read_json("train_CEPO.jsonl")
    ds = Dataset.from_pandas(df)

    # 不要 tokenization
    train_dataset = ds

    # ===== 训练参数 =====
    args = TrainingArguments(
        output_dir="./output/cepo_lora",
        per_device_train_batch_size=1,   # ⚠ CEPO 很重   注意这里学习略是否需要线性变化
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-5,
        logging_steps=10,
        save_steps=100,
        gradient_checkpointing=True,
        bf16=True
    )

    trainer = CepoTrainer(
        ref_model=ref_model,
        tokenizer=tokenizer,
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=lambda x: x  # 直接返回原始结构
    )

    trainer.train()