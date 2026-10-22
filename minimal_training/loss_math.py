"""Batch-one reductions and the checkpoint's top-64/128-union KL objective."""
import torch

NOISE_EPS = .001
TAIL_FLOOR = 1e-5
FORBIDDEN_RESPONSE_IDS = (*range(18, 30), 100)


def teacher_topk(logits):
    # Remove incompatible IDs from selected support, not the full normalizer.
    logp = logits.float().log_softmax(-1)
    legal = logp.clone()
    legal[..., list(FORBIDDEN_RESPONSE_IDS)] = -torch.inf
    values, indices = legal.topk(64, dim=-1)
    return values, indices


def union_support(primary, secondary):
    """Keep primary top-64, then highest secondary IDs, until 128 unique IDs."""
    width = min(128, secondary.shape[-1])
    candidates = torch.cat([primary, secondary.detach().topk(width, dim=-1).indices], -1)
    # Width is at most 192; stable sorting preserves primary ownership/order.
    order = candidates.argsort(dim=-1, stable=True)
    sorted_ids = candidates.gather(-1, order)
    unique = torch.ones_like(sorted_ids, dtype=torch.bool)
    unique[..., 1:] = sorted_ids[..., 1:] != sorted_ids[..., :-1]
    unique = torch.zeros_like(unique).scatter(-1, order, unique)
    selected = unique & (unique.cumsum(-1) <= width)
    slots = (~selected).to(torch.int8).argsort(dim=-1, stable=True)[..., :width]
    return candidates.gather(-1, slots), slots


def selected_logprobs(logits, indices):
    logits = logits.float()
    return logits.gather(-1, indices) - logits.logsumexp(-1, keepdim=True)


def _head_and_tail(logp):
    head = logp.double().exp()
    mass = head.sum(-1, keepdim=True)
    raw_tail = 1 - mass
    saturated = raw_tail <= 0
    head = head * torch.where(saturated, (1 - TAIL_FLOOR) / mass.clamp_min(torch.finfo(torch.float64).tiny), 1.)
    tail = torch.where(saturated, TAIL_FLOOR, raw_tail)
    return head, tail


def tail_kl(target_logp, student_logp, *, reverse=False):
    """KL on selected singleton tokens plus one residual tail category."""
    target = torch.cat(_head_and_tail(target_logp), -1)
    student = torch.cat(_head_and_tail(student_logp), -1)
    p, q = (student, target) if reverse else (target, student)
    tiny = torch.finfo(torch.float64).tiny
    return (p * (p.clamp_min(tiny).log() - q.clamp_min(tiny).log())).sum(-1).to(student_logp.dtype)


def opd_token_losses(ar, diffusion, teacher_logp, teacher_ids):
    """AR learns reverse teacher KL; only diffusion gets consistency gradients."""
    support, slots = union_support(teacher_ids, ar)
    head, tail = _head_and_tail(teacher_logp.detach())
    target_prob = torch.where(slots < teacher_ids.shape[-1],
        head.gather(-1, slots.clamp_max(teacher_ids.shape[-1] - 1)),
        tail / (ar.shape[-1] - teacher_ids.shape[-1]))
    target_logp = target_prob.clamp_min(torch.finfo(torch.float64).tiny).log()
    ar_kl = tail_kl(target_logp, selected_logprobs(ar, support), reverse=True)
    with torch.no_grad():
        support, _ = union_support(ar.detach().topk(64, dim=-1).indices, diffusion)
        target_logp = selected_logprobs(ar.detach(), support)
    diff_kl = tail_kl(target_logp, selected_logprobs(diffusion, support))
    return ar_kl, diff_kl


def text_denominators(target_count, masked_count, *, opd):
    # SFT already applies 1/p to each masked CE. OPD explicitly does not.
    return target_count, ((1 + NOISE_EPS) / 2 * target_count if opd else max(1, masked_count))


def image_mean(loss_sum, supervised_count):
    return loss_sum / max(1, supervised_count)


def image_group_table(labels):
    counts = torch.bincount(labels, minlength=16384)
    order = labels.argsort(stable=True)
    offsets = counts.cumsum(0) - counts
    column = torch.arange(int(counts.max()), device=labels.device)
    valid = column[None] < counts[:, None]
    members = order[(offsets[:, None] + column).clamp_max(len(labels) - 1)]
    return members, valid


def image_token_losses(logits, targets, groups, members, valid):
    log_z = logits.logsumexp(-1)
    exact = log_z - logits.gather(-1, targets[:, None]).squeeze(-1)
    group = groups[targets]
    group_logits = logits.gather(-1, members[group]).masked_fill(~valid[group], -torch.inf)
    grouped = log_z - group_logits.logsumexp(-1)
    return exact, grouped


def image_condition_dropout(ids, target, source, draw):
    """Exclusive CFG dropout bins: prompt .1, source .3, both .05."""
    edit = bool(source.any())
    mode = (1 if draw < .1 else 2 if edit and draw < .4 else 3 if edit and draw < .45 else 0)
    eos = ids.eq(11)
    prefix = eos.long().cumsum(-1).sub(eos.long()).lt(2)
    if not edit:
        mask = prefix & ~target if mode == 1 else torch.zeros_like(target)
    elif mode == 1:
        mask = ~target & ~source
    elif mode == 2:
        mask = source
    elif mode == 3:
        mask = (prefix | source) & ~target
    else:
        mask = torch.zeros_like(target)
    return mask, source & (mode in (2, 3)), mode
