import torch
import torch.nn.functional as F

def compute_seq_logprob(model, enc, prompt_len):
    out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    logits = out.logits[:, :-1, :]  # [B, T-1, V]
    
    labels = enc["input_ids"][:, 1:]  # [B, T-1]
    
    log_probs = F.log_softmax(logits, dim=-1)  # token 级 log-prob
    token_lp = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    seq_len = token_lp.size(1)
    mask = torch.zeros_like(token_lp)
    mask[:, prompt_len-1:seq_len] = 1.0

    return (token_lp * mask).sum(dim=-1)

    import torch
import torch.nn.functional as F

def dpo_loss_single(
    model,
    ref_model,
    tokenizer,
    prompt: str,
    good: str,
    bad: str,
    beta: float = 1.0,
    device: str = "cuda"
):
    # prompt 长度
    prompt_ids = tokenizer(prompt)["input_ids"]
    prompt_len = len(prompt_ids)
    texts = [
        prompt + good,
        prompt + bad
    ]
    enc = tokenizer(texts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        ref_lp = compute_seq_logprob(ref_model, enc, prompt_len)
    model_lp = compute_seq_logprob(model, enc, prompt_len)
    good_lp_model, bad_lp_model = model_lp[0], model_lp[1]
    good_lp_ref, bad_lp_ref = ref_lp[0], ref_lp[1]
    good_score = good_lp_model - good_lp_ref
    bad_score  = bad_lp_model  - bad_lp_ref
    diff = beta * (good_score - bad_score)
    loss = -F.logsigmoid(diff)
    return loss
