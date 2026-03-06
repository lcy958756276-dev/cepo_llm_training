#先用小模型试一试再说，这一块更新的batchsize的问题，更安全  修改地方：1.good和bad同时进行forward  2. 对pad进行mask，，final_mask = gen_mask * attn_mask
import torch
import torch.nn.functional as F

def compute_seq_logprob(model, enc, prompt_len):
    """
    计算每个序列生成部分的平均 log-prob
    enc: tokenizer 输出（已 padding）
    prompt_len: prompt token 数
    """
    out = model(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"]
    )
    logits = out.logits[:, :-1, :]          # [B, T-1, V]
    labels = enc["input_ids"][:, 1:]        # [B, T-1]
    attn_mask = enc["attention_mask"][:, 1:]  # [B, T-1]
    log_probs = F.log_softmax(logits, dim=-1)
    token_lp = log_probs.gather(
        dim=-1,
        index=labels.unsqueeze(-1)
    ).squeeze(-1)  # [B, T-1]
    # ---------------------------
    # 生成部分 mask
    # ---------------------------
    seq_len = token_lp.size(1)
    gen_mask = torch.zeros_like(token_lp)
    gen_mask[:, prompt_len-1:seq_len] = 1.0
    # ---------------------------
    # padding mask
    # ---------------------------
    final_mask = gen_mask * attn_mask

    # 防止除0
    denom = final_mask.sum(dim=-1).clamp(min=1e-8)
    return (token_lp * final_mask).sum(dim=-1) / denom

def cepo_loss_separate(
    model, ref_model, tokenizer,
    prompt: str,
    good: list[str],
    bad_blocks: list[list[str]],
    beta: float = 0.5,
    lambda_eq: float = 0.2,
    device: str = "cuda"
):
    """
    CEPO loss:
    - block-level partial Plackett–Luce ranking
    - good 内等价约束
    - 单样本内合并 forward
    """

    prompt_ids = tokenizer(prompt)["input_ids"]
    prompt_len = len(prompt_ids)

    # ----------------------------------------
    # 1️⃣ 扁平化所有 candidate
    # ----------------------------------------
    all_texts = []
    block_sizes = []

    # good block
    good_texts = [prompt + cand for cand in good]
    if good_texts:
        all_texts.extend(good_texts)
        block_sizes.append(len(good_texts))
    else:
        block_sizes.append(0)

    # bad blocks
    for block in bad_blocks:
        if block:
            block_texts = [prompt + cand for cand in block]
            all_texts.extend(block_texts)
            block_sizes.append(len(block))
        else:
            block_sizes.append(0)

    if len(all_texts) == 0:
        return torch.tensor(0.0, device=device)

    # ----------------------------------------
    # 2️⃣ 一次 tokenize + forward
    # ----------------------------------------
    enc = tokenizer(
        all_texts,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        lp_ref = compute_seq_logprob(ref_model, enc, prompt_len)

    lp_model = compute_seq_logprob(model, enc, prompt_len)

    scores = beta * (lp_model - lp_ref)  # shape [N_total]

    # ----------------------------------------
    # 3️⃣ 按 block 切分
    # ----------------------------------------
    block_scores = []
    idx = 0

    # good
    n_good = block_sizes[0]
    if n_good > 0:
        good_scores = scores[idx:idx+n_good]
        good_block_score = good_scores.mean()
        block_scores.append(good_block_score)
        idx += n_good
    else:
        good_scores = torch.tensor([], device=device)

    # bad blocks
    for size in block_sizes[1:]:
        if size > 0:
            block_scores.append(scores[idx:idx+size].mean())
            idx += size

    # ----------------------------------------
    # 4️⃣ good 等价约束
    # ----------------------------------------
    if good_scores.numel() > 1:
        good_mean = good_scores.mean()
        loss_eq = ((good_scores - good_mean) ** 2).mean()
    else:
        loss_eq = torch.tensor(0.0, device=device)

    # ----------------------------------------
    # 5️⃣ Partial Plackett–Luce
    # ----------------------------------------
    loss_rank = torch.tensor(0.0, device=device)

    for i in range(len(block_scores) - 1):
        loss_rank -= (
            block_scores[i]
            - torch.logsumexp(
                torch.stack(block_scores[i:]),
                dim=0
            )
        )

    # ----------------------------------------
    # 6️⃣ 总 loss
    # ----------------------------------------
    total_loss = loss_rank + lambda_eq * loss_eq

    return total_loss,loss_rank,loss_eq