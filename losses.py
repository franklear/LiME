import functools

import torch
import torch.nn.functional as F


def smart_sliced_loss(agg):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(a, b, all_to_all=False, **kwargs):
            assert callable(f)
            assert callable(agg)

            a_is_sliced = isinstance(a, list)
            b_is_sliced = isinstance(b, list)

            all_ret = []
            if all_to_all:
                if not a_is_sliced:
                    a = [a]
                if not b_is_sliced:
                    b = [b]

                for a_ in a:
                    for b_ in b:
                        torch.testing.assert_close(a_.shape[0], b_.shape[0])

                        cur = f(a_, b_, **kwargs)
                        all_ret.append(cur)
            else:
                if a_is_sliced and not b_is_sliced:
                    b = b.split(a[0].shape[0])
                elif not a_is_sliced and b_is_sliced:
                    a = a.split(b[0].shape[0])
                elif not a_is_sliced and not b_is_sliced:
                    a = [a]
                    b = [b]

                torch.testing.assert_close(len(a), len(b))

                for a_, b_ in zip(a, b):
                    torch.testing.assert_close(a_.shape[0], b_.shape[0])

                    cur = f(a_, b_, **kwargs)
                    all_ret.append(cur)

            ret = agg(all_ret)
            return ret
        return wrapper
    return decorator


def _agg_kl_loc_loss(all_ret):
    ret = torch.stack(all_ret).mean()
    return ret


@smart_sliced_loss(_agg_kl_loc_loss)
def kl_loc_loss(pre, post, mask=None):
    pre = pre.to(torch.float32)
    post = post.to(torch.float32)

    sequence = pre.dim() == 3
    pre_ = pre.view(-1, pre.shape[-1])
    post_ = post.view(pre_.shape)
    assert pre_.shape[0] == post_.shape[0]

    if not sequence:
        if pre_.shape[-1] == 1:  # No masking needed for binary classification
            return (pre.sigmoid() * (F.logsigmoid(pre) - F.logsigmoid(post))).mean() + (
                (-pre).sigmoid() * (F.logsigmoid(-pre) - F.logsigmoid(-post))
            ).mean()
        else:
            return (pre_.softmax(-1) * (pre_.log_softmax(-1) - post_.log_softmax(-1))).sum(-1).mean()
    else:  # We have sequences of predictions; masking needed
        if pre_.shape[-1] > 1:
            assert mask is not None
            mask_ = mask.view(pre_.shape[0])
            kl = (pre_.softmax(-1) * (pre_.log_softmax(-1) - post_.log_softmax(-1))).sum(-1)
            return (kl * mask_).sum() / mask_.sum()

    raise NotImplementedError


def binary_log_probs(pred, targ):
    neg_mask = torch.ones_like(pred)
    neg_mask[targ == 0] *= -1
    pred = pred * neg_mask
    log_probs = F.logsigmoid(pred)
    acc = (log_probs.exp() > 0.5).float().mean()
    return {
        "acc": acc,
        "log_prob": log_probs.mean(),
        "prob": log_probs.exp().mean(),
        "nll": -log_probs.mean(),
        "n_tokens": torch.tensor(log_probs.shape[0], dtype=log_probs.dtype, device=log_probs.device)
    }


def multiclass_log_probs(pred, targ, shift=True):
    NULL_TOKEN = 0  # a placeholder used for masked target locations

    pred = pred.clone()
    targ = targ.clone()
    if shift and pred.dim() == 3:  # Dealing with sequences
        pred = pred[:, :-1]  # Remove last prediction in sequence
        targ = targ[:, 1:]  # Shift to align predictions and targets

    mask = targ != -100
    targ[~mask] = NULL_TOKEN  # Can be any valid token, since we'll throw them out
    unmasked_log_probs = pred.log_softmax(-1).gather(-1, targ.unsqueeze(-1)).squeeze(-1)

    pred_ids = pred.argmax(-1).masked_fill(~mask, NULL_TOKEN)
    correct = pred_ids == targ
    if pred.dim() == 3:
        correct = (pred_ids == targ).all(-1)  # We want to get the whole sequence right
    acc = correct.float().mean()

    n_tokens = mask.float().sum()
    log_prob = (unmasked_log_probs * mask.float()).sum() / n_tokens
    prob = (unmasked_log_probs.exp() * mask.float()).sum() / n_tokens
    return {
        "acc": acc,
        "log_prob": log_prob,
        "prob": prob,
        "n_tokens": n_tokens,
        "nll": -log_prob
    }


def _agg_masked_log_probs(all_ret):
    ret = {k: [] for k in all_ret[0].keys()}
    for r in all_ret:
        for k, v in r.items():
            ret[k].append(v)
    ret = {k: torch.stack(v).mean() for k, v in ret.items()}
    return ret


@smart_sliced_loss(_agg_masked_log_probs)
def masked_log_probs(pred, targ, shift=True):
    pred = pred.to(torch.float32)

    if not (pred.dim() == 2 or pred.dim() == 3):
        raise RuntimeError(f"Expected pred to have 2 or 3 dimensions, got {pred.shape}")

    if pred.shape[-1] == 1:
        return binary_log_probs(pred, targ)
    else:
        return multiclass_log_probs(pred, targ, shift=shift)
