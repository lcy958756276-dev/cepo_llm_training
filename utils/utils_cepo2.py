import torch
import torch.nn.functional as F

def compute_seq_logprob(model, enc, prompt_len):
    out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])#输出的是整个序列上每个位置的分数分布
    logits = out.logits[:, :-1, :]  # [B, T-1, V]
    
    # 挑出每个位置的tokenid
    labels = enc["input_ids"][:, 1:]  # [B, T-1]
    
    log_probs = F.log_softmax(logits, dim=-1)  # token 级 log-prob
    token_lp = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    #先将变成【B,T-1,1】最后一维度挑出tokneid位置的概率，最终[B, T-1]

    # mask 只对 cand 部分有效
    seq_len = token_lp.size(1)
    mask = torch.zeros_like(token_lp)
    mask[:, prompt_len-1:seq_len] = 1.0#创建mask只保留生成部分的token的log-prob

    return (token_lp * mask).sum(dim=-1)/ mask.sum(dim=-1)#这里就是生成，部分的对每个答案的tokenid的概率值提取
#！！！！！！！！！！！！！！！！！！！！！！！！！！padding=True没有修正！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！

def cepo_loss_separate(
    model, ref_model, tokenizer,
    prompt: str,
    good: list[str],
    bad_blocks: list[list[str]],
    beta: float = 1.0,
    lambda_eq: float = 0.1,   # ⭐ 新增：good 等价约束权重
    device: str = "cuda"
):
    """
    CEPO loss：- block-level partial Plackett–Luce ranking- + good 内等价约束
    """
    prompt_ids = tokenizer(prompt)["input_ids"]
    prompt_len = len(prompt_ids)

    # ----------------------------------------
    # 1) good candidates
    # ----------------------------------------
    good_texts = [prompt + cand for cand in good]
    if good_texts:
        good_enc = tokenizer(good_texts, return_tensors="pt", padding=True).to(device)#padding会让所有序列都填充（pad）到本批次中最长序列的长度
        with torch.no_grad():
            good_lp_ref = compute_seq_logprob(ref_model, good_enc, prompt_len)
        good_lp_model = compute_seq_logprob(model, good_enc, prompt_len)
        good_scores = beta * (good_lp_model - good_lp_ref)   # shape [n_good]
    else:
        good_scores = torch.tensor([], device=device)

    # ----------------------------------------
    # 2) bad candidates（block separately）
    # ----------------------------------------
    bad_scores_list = []
    for block in bad_blocks:
        if not block:
            continue
        block_texts = [prompt + cand for cand in block]
        block_enc = tokenizer(block_texts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            block_lp_ref = compute_seq_logprob(ref_model, block_enc, prompt_len)
        block_lp_model = compute_seq_logprob(model, block_enc, prompt_len)
        block_scores = beta * (block_lp_model - block_lp_ref)
        bad_scores_list.append(block_scores.mean())  # block 内平均

    # ----------------------------------------
    # 3) good 等价约束 loss
    # ----------------------------------------
    if good_scores.numel() > 1:
        good_mean = good_scores.mean()
        loss_eq = ((good_scores - good_mean) ** 2).mean()
    else:
        loss_eq = torch.tensor(0.0, device=device)

    # ----------------------------------------
    # 4) block-level scores
    # ----------------------------------------
    block_scores = []
    if good_scores.numel() > 0:
        block_scores.append(good_scores.mean())
    block_scores.extend(bad_scores_list)
"""
block_scores = [
  mean(good_scores),
  s_c,
  s_d
]
"""
    # ----------------------------------------
    # 5) partial Plackett–Luce loss（原逻辑）
    # ----------------------------------------
    loss_rank = 0.0
    for i in range(len(block_scores) - 1):
        num = torch.exp(block_scores[i])
        denom = sum(torch.exp(x) for x in block_scores[i:])
        loss_rank = loss_rank - torch.log(num / denom)

    # ----------------------------------------
    # 6) 总 loss
    # ----------------------------------------
    total_loss = loss_rank + lambda_eq * loss_eq
    return total_loss

"""
一、先说 bad_scores_list 到底长什么样
假设一个具体例子
prompt = "c"
good = ["A", "B"]

bad_blocks = [
    ["D1", "D2"],   # block 1（同一“坏等级”）
    ["E"]           # block 2（更坏）
]
beta = 1.0

1️⃣ 第一个 bad block：["D1", "D2"]
block_texts = ["cD1", "cD2"]


假设 compute_seq_logprob 算出来的是：

model:     [-2.0, -1.8]
ref_model: [-2.5, -2.0]


那么 DPO score（逐条）是：

block_scores = beta * (block_lp_model - block_lp_ref)
             = [0.5, 0.2]


你这里做的是：

block_scores.mean() = (0.5 + 0.2) / 2 = 0.35


所以：

bad_scores_list.append(0.35)

2️⃣ 第二个 bad block：["E"]
block_texts = ["cE"]


假设：

model:     [-3.0]
ref_model: [-2.8]


DPO score：

[-0.2]


block 平均还是它自己：

bad_scores_list.append(-0.2)

3️⃣ 最终 bad_scores_list
bad_scores_list = [
    tensor(0.35),   # bad block 1
    tensor(-0.2)    # bad block 2（更差）
]


⚠️ 注意：
bad_scores_list 里每一个元素 = 一个 block 的“整体质量”

二、现在 block_scores 是什么？

你前面已经算了：

good_scores = tensor([0.8, 0.6])  # 举例
good_block_score = good_scores.mean() = 0.7


然后你构造的是：

block_scores = [
    0.7,     # good block
    0.35,    # bad block 1
    -0.2     # bad block 2
]


这个顺序非常关键：

👉 good ≻ bad_block_1 ≻ bad_block_2

这已经是一个**部分排序（partial order）**了。
"""