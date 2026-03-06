#peft库写的
import torch
import torch.nn.functional as F

#算 log π(y | x)
def log_prob(model, input_ids, attention_mask, prompt_len):
    """
    input_ids: [B, T]
    prompt_len: prompt token 长度（前 prompt_len 个 token 不计入）
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1]              # [B, T-1, V]
    labels = input_ids[:, 1:]                     # [B, T-1]

    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(
        dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)

    # mask 掉 prompt 部分
    mask = torch.zeros_like(token_log_probs)
    mask[:, prompt_len-1:] = 1.0

    seq_log_prob = (token_log_probs * mask).sum(dim=-1)
    return seq_log_prob

def cepo_loss_one_sample(
    model,
    ref_model,
    tokenizer,
    prompt,
    good_list,
    bad_list,
    beta=0.1,
    device="cuda"
):
    def build_batch(outputs):
        texts = [prompt + o for o in outputs]#将prompt和每个输出拼在一起texts = [ x + g1 , x + g2 ]
        enc = tokenizer(
            texts, return_tensors="pt", padding=True
        ).to(device)
        prompt_len = len(tokenizer(prompt)["input_ids"])
        return enc, prompt_len#输出的是prompt问题的长度和整体的tokenizer

    # ===== good =====
    enc_g, prompt_len = build_batch(good_list)
    s_g = log_prob(model, **enc_g, prompt_len=prompt_len)
    s_g_ref = log_prob(ref_model, **enc_g, prompt_len=prompt_len)

    # log-mean-exp（CEPO 关键）
    Sg = torch.logsumexp(s_g, dim=0) - torch.log(
        torch.tensor(len(good_list), device=device)
    )#log( (exp(s_g1) + exp(s_g2)) / 2 )
    Sg_ref = torch.logsumexp(s_g_ref, dim=0) - torch.log(
        torch.tensor(len(good_list), device=device)
    )

    # ===== bad =====
    enc_b, _ = build_batch(bad_list)
    s_b = log_prob(model, **enc_b, prompt_len=prompt_len)
    s_b_ref = log_prob(ref_model, **enc_b, prompt_len=prompt_len)

    Sb = torch.logsumexp(s_b, dim=0) - torch.log(
        torch.tensor(len(bad_list), device=device)
    )
    Sb_ref = torch.logsumexp(s_b_ref, dim=0) - torch.log(
        torch.tensor(len(bad_list), device=device)
    )

    # ===== CEPO preference gap =====
    delta = (Sg - Sb) - (Sg_ref - Sb_ref)

    # ===== final loss =====
    loss = -F.logsigmoid(beta * delta)
    return loss
"""
s_g      = [s_g1, s_g2]        # model 的 logπ
s_g_ref  = [ŝ_g1, ŝ_g2]        # ref_model 的 logπ

s_b      = [s_b1]
s_b_ref  = [ŝ_b1]

"""