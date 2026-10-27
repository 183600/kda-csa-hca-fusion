"""run_quality split module: errors."""

from __future__ import annotations

from ._shared import *  # noqa: F401,F403

from .tasks import (  # noqa: F401
    SYNTHETIC_TASKS,
)


_ERROR_CATEGORY_DEFINITIONS = {
    'mqar': OrderedDict([
        ('confused_with_other_value',
         'predicted the value paired with a DIFFERENT key (association '
         'mix-up between the n_kv pairs; grows with n_kv)'),
        ('echoed_key',
         'predicted a key token instead of its value (retrieval stopped at '
         'the key; positional copy of the cue region)'),
        ('copied_distractor_token',
         'predicted a filler token from the noise region (attention/state '
         'dilution instead of retrieval)'),
        ('other', 'none of the above'),
    ]),
    'kv_update': OrderedDict([
        ('used_stale_value',
         'predicted the OLD value for the cued key (delta-rule update failed '
         'to overwrite; classic interference failure)'),
        ('confused_with_other_new_value',
         'predicted another key\'s new value (update happened, association '
         'mix-up)'),
        ('echoed_key',
         'predicted a key token instead of its latest value'),
        ('other', 'none of the above'),
    ]),
    'long_range': OrderedDict([
        ('echoed_key',
         'predicted the key itself instead of its value'),
        ('copied_adjacent_token',
         'predicted the distractor immediately after the pair (local copy '
         'instead of long-range retrieval)'),
        ('other', 'none of the above'),
    ]),
    'multi_agg': OrderedDict([
        ('runner_up_category',
         'predicted the second-most-frequent category (off-by-one count; '
         'aggregation is close but not exact)'),
        ('echoed_last_token',
         'predicted the final token itself (recency copy rather than '
         'majority aggregation)'),
        ('other', 'none of the above'),
    ]),
}


def _classify_task_errors(task, meta, target, pred, cue_pos, n_kv):
    """Classify every sample in a batch into a wrong-answer category.

    Returns a list of category names (``'correct'`` for correct rows).
    Vectorized masks decide the category; the per-sample loop is only over
    the batch dimension and uses the task's known layout, not heuristics
    on the raw ids.
    """
    ids = meta['input_ids']
    b = int(ids.shape[0])
    device = ids.device
    correct = (pred == target)
    cat = ['correct'] * b
    extras = {}
    if task == 'mqar':
        keys = ids[:, 0:2 * n_kv:2]
        vals = ids[:, 1:2 * n_kv:2]
        is_other_val = (vals == pred[:, None]).any(dim=1)
        is_key = (keys == pred[:, None]).any(dim=1)
        noise = ids[:, 2 * n_kv:cue_pos]
        is_noise = ((noise == pred[:, None]).any(dim=1)
                    if noise.shape[1] else torch.zeros(
                        b, dtype=torch.bool, device=device))
        for i in range(b):
            if bool(correct[i]):
                continue
            if bool(is_other_val[i]):
                cat[i] = 'confused_with_other_value'
            elif bool(is_key[i]):
                cat[i] = 'echoed_key'
            elif bool(is_noise[i]):
                cat[i] = 'copied_distractor_token'
            else:
                cat[i] = 'other'
    elif task == 'kv_update':
        vals_old = ids[:, 1:2 * n_kv:2]
        keys2 = ids[:, 2 * n_kv:4 * n_kv:2]
        vals2 = ids[:, 2 * n_kv + 1:4 * n_kv:2]
        j = meta['key_idx']
        cue_key = ids[:, cue_pos]
        old_for_cue = vals_old.gather(1, j[:, None]).squeeze(1)
        is_stale = (pred == old_for_cue)
        is_other_new = (vals2 == pred[:, None]).any(dim=1)
        is_key2 = (keys2 == pred[:, None]).any(dim=1)
        extras['cue_key'] = cue_key
        extras['stale_value'] = old_for_cue
        for i in range(b):
            if bool(correct[i]):
                continue
            if bool(is_stale[i]):
                cat[i] = 'used_stale_value'
            elif bool(is_other_new[i]):
                cat[i] = 'confused_with_other_new_value'
            elif bool(is_key2[i]):
                cat[i] = 'echoed_key'
            else:
                cat[i] = 'other'
    elif task == 'long_range':
        p = meta['key_pos']
        key = ids.gather(1, p[:, None]).squeeze(1)
        is_key = (pred == key)
        adj_pos = torch.clamp(p + 2, max=max(0, cue_pos - 1))
        adj = ids.gather(1, adj_pos[:, None]).squeeze(1)
        is_adj = (pred == adj)
        for i in range(b):
            if bool(correct[i]):
                continue
            if bool(is_key[i]):
                cat[i] = 'echoed_key'
            elif bool(is_adj[i]):
                cat[i] = 'copied_adjacent_token'
            else:
                cat[i] = 'other'
    elif task == 'multi_agg':
        counts = torch.as_tensor(meta['counts'], device=device)
        top2 = counts.topk(2, dim=1).values
        pred_count = counts.gather(1, pred[:, None]).squeeze(1)
        is_runner_up = (pred_count == top2[:, 1])
        is_last = (pred == ids[:, -1])
        extras['counts'] = counts
        for i in range(b):
            if bool(correct[i]):
                continue
            if bool(is_runner_up[i]):
                cat[i] = 'runner_up_category'
            elif bool(is_last[i]):
                cat[i] = 'echoed_last_token'
            else:
                cat[i] = 'other'
    return cat, extras


def _error_confidence(logits, target, pred, i):
    """Model confidence for one misclassified sample, without extra RNG.

    A wrong answer with ``pred_prob ~= 0.5`` and ``margin ~= 0`` is a
    near-tie (the sample is ambiguous for this model); a wrong answer with
    ``pred_prob ~ 1`` is a confident blind spot. The aggregate accuracy
    cannot distinguish them, and they imply different next experiments
    (more data / calibration vs. a new mechanism). Returns ``None`` when
    the row is non-finite, so a diverged forward cannot fabricate a
    plausible confidence block or poison strict JSON.
    """
    row = logits[i].detach().float()
    if row.numel() == 0 or not bool(torch.isfinite(row).all()):
        return None
    probs = torch.softmax(row, dim=-1)
    pred_id = int(pred[i])
    target_id = int(target[i])
    pred_prob = float(probs[pred_id])
    target_prob = float(probs[target_id])
    target_rank = int((row > row[target_id]).sum().item()) + 1
    # Shannon entropy in nats, normalized by the vocabulary maximum so the
    # value is comparable across vocab sizes (1 = uniform / maximally
    # ambiguous, 0 = one-hot).
    log_probs = torch.log_softmax(row, dim=-1)
    entropy = float(-(probs * log_probs).sum())
    vocab = int(row.numel())
    return {
        'pred_prob': pred_prob,
        'target_prob': target_prob,
        'margin': pred_prob - target_prob,
        'target_rank': target_rank,
        'entropy_nats': entropy,
        'entropy_ratio': (entropy / math.log(vocab)) if vocab > 1 else 0.0,
    }


def _error_example(task, meta, target, pred, i, cue_pos, extras,
                   confidence=None):
    """Decode one misclassified sample into human-readable task terms."""
    ids = meta['input_ids']
    ex = {
        'input_ids': [int(v) for v in ids[i].tolist()],
        'target': int(target[i]),
        'pred': int(pred[i]),
        'cue_pos': int(cue_pos),
    }
    if confidence:
        ex.update(confidence)
    if task in ('mqar', 'kv_update'):
        ex['key_idx'] = int(meta['key_idx'][i])
    if task == 'kv_update':
        ex['cued_key'] = int(extras['cue_key'][i])
        ex['stale_value'] = int(extras['stale_value'][i])
    elif task == 'long_range':
        ex['key_pos'] = int(meta['key_pos'][i])
        ex['distance'] = int(meta['distance'][i])
    elif task == 'multi_agg':
        ex['counts'] = [int(c) for c in meta['counts'][i]]
        ex['pred_count'] = int(extras['counts'][i, int(pred[i])])
    return ex


def analyze_errors(layer, head, embed, *, op_name, task='mqar', seq_len=16,
                   n_kv=1, vocab=16, device='cpu', n_batches=2, batch=64,
                   seed=42, max_examples=2):
    """Bad-case analysis: classify and print held-out misclassifications.

    Runs the trained model (read-only) on ``n_batches`` x ``batch`` samples
    from the dedicated ``_ERROR_ANALYSIS_SEED_OFFSET`` stream, classifies
    every wrong prediction into a task-level category (see
    ``_ERROR_CATEGORY_DEFINITIONS``), keeps up to ``max_examples`` decoded
    examples per category, and prints a compact per-category summary with
    the first example. Synthetic labels are exact, so a wrong answer
    cannot be label noise: the category distribution localizes the
    mechanism blind spot (association mix-up, failed update, positional
    copy, ...) that the aggregate accuracy hides.

    The report is JSON-serializable and records the stream/seed so an
    example can be regenerated exactly.
    """
    if isinstance(device, str):
        device = torch.device(device)
    if task not in SYNTHETIC_TASKS:
        raise ValueError(
            f"task={task!r} must be one of {sorted(SYNTHETIC_TASKS)}.")
    make_batch = SYNTHETIC_TASKS[task]['make']
    gen = make_seeded_generator(seed + _ERROR_ANALYSIS_SEED_OFFSET,
                                device=device)
    was_training = {m: m.training for m in (layer, head, embed)}
    counts: dict = OrderedDict()
    examples: dict = OrderedDict()
    # Per-category confidence accumulators: is the category dominated by
    # near-ties (ambiguous samples) or by high-confidence mistakes (blind
    # spots)? Keeping only the first example would hide that distinction.
    conf_acc: dict = OrderedDict()
    n_total = n_wrong = 0
    dist_correct: dict = {}
    dist_total: dict = {}
    try:
        layer.eval()
        head.eval()
        embed.eval()
        with torch.no_grad():
            for _ in range(int(n_batches)):
                x_emb, target, cue_pos, meta = make_batch(
                    batch, seq_len, n_kv, vocab, embed, device,
                    generator=gen, return_meta=True)
                h = layer(x_emb)
                logits = head(h, cue_pos)
                if not bool(torch.isfinite(logits).all()):
                    # A NaN/Inf output has no category: argmax would pick an
                    # arbitrary class and the report would show a plausible
                    # error mix for a diverged model. Refuse the analysis;
                    # staged triage belongs to the training/eval path.
                    n_bad = int((~torch.isfinite(logits)).sum())
                    raise ValueError(
                        f'analyze_errors received {n_bad} non-finite '
                        f'logit(s) for task={task!r}; run '
                        f'diagnose_nonfinite() on the eval forward pass '
                        f'instead of interpreting the bad-case mix')
                pred = logits.argmax(-1)
                cats, extras = _classify_task_errors(
                    task, meta, target, pred, cue_pos, n_kv)
                n_total += int(target.numel())
                for i, c in enumerate(cats):
                    if c == 'correct':
                        continue
                    n_wrong += 1
                    counts[c] = counts.get(c, 0) + 1
                    confidence = _error_confidence(logits, target, pred, i)
                    if confidence is not None:
                        conf = conf_acc.setdefault(c, {
                            'n_scored': 0,
                            'n_confident_wrong': 0,
                            'sum_pred_prob': 0.0,
                            'sum_target_prob': 0.0,
                            'sum_margin': 0.0,
                            'sum_entropy_ratio': 0.0,
                        })
                        conf['n_scored'] += 1
                        conf['n_confident_wrong'] += int(
                            confidence['pred_prob'] >= 0.5)
                        conf['sum_pred_prob'] += confidence['pred_prob']
                        conf['sum_target_prob'] += confidence['target_prob']
                        conf['sum_margin'] += confidence['margin']
                        conf['sum_entropy_ratio'] += confidence[
                            'entropy_ratio']
                    bucket = examples.setdefault(c, [])
                    if len(bucket) < int(max_examples):
                        bucket.append(_error_example(
                            task, meta, target, pred, i, cue_pos, extras,
                            confidence=confidence))
                if task == 'long_range':
                    ok = (pred == target)
                    d = meta['distance']
                    for i in range(int(target.numel())):
                        di = int(d[i])
                        dist_total[di] = dist_total.get(di, 0) + 1
                        dist_correct[di] = dist_correct.get(di, 0) + int(ok[i])
    finally:
        for m, was in was_training.items():
            m.train(was)

    def _confidence_block(name):
        acc = conf_acc.get(name)
        if not acc or not acc['n_scored']:
            return None
        n = acc['n_scored']
        return {
            'n_scored': int(n),
            'n_confident_wrong': int(acc['n_confident_wrong']),
            'rate_confident_wrong': acc['n_confident_wrong'] / n,
            'mean_pred_prob': acc['sum_pred_prob'] / n,
            'mean_target_prob': acc['sum_target_prob'] / n,
            'mean_margin': acc['sum_margin'] / n,
            'mean_entropy_ratio': acc['sum_entropy_ratio'] / n,
            'note': ('confidence = top-1 probability of the WRONG answer. '
                     'High pred_prob is a blind spot; margins near zero / '
                     'entropy near 1 are ambiguous near-ties.'),
        }

    categories = OrderedDict()
    for name, definition in _ERROR_CATEGORY_DEFINITIONS[task].items():
        n = counts.get(name, 0)
        categories[name] = {
            'count': int(n),
            'rate_of_wrong': (n / n_wrong) if n_wrong else 0.0,
            'rate_of_all': (n / n_total) if n_total else 0.0,
            'definition': definition,
            'confidence': _confidence_block(name),
        }
    for name in counts:
        if name not in categories:
            categories[name] = {
                'count': int(counts[name]),
                'rate_of_wrong': counts[name] / n_wrong if n_wrong else 0.0,
                'rate_of_all': (counts[name] / n_total) if n_total else 0.0,
                'definition': 'category not covered by the task layout',
                'confidence': _confidence_block(name),
            }

    scored = [c for c in conf_acc.values() if c['n_scored']]
    n_scored_wrong = sum(c['n_scored'] for c in scored)
    confidence_summary = None
    if n_scored_wrong:
        n_conf = sum(c['n_confident_wrong'] for c in scored)
        confidence_summary = {
            'n_scored_wrong': int(n_scored_wrong),
            'n_confident_wrong': int(n_conf),
            'rate_confident_wrong': n_conf / n_scored_wrong,
            'mean_pred_prob': (sum(c['sum_pred_prob'] for c in scored)
                               / n_scored_wrong),
            'mean_target_prob': (sum(c['sum_target_prob'] for c in scored)
                                 / n_scored_wrong),
            'mean_margin': (sum(c['sum_margin'] for c in scored)
                            / n_scored_wrong),
            'note': ('High confidence on wrong answers points to a '
                     'systematic blind spot; low margin / high entropy '
                     'points to genuinely ambiguous samples.'),
        }

    by_distance = None
    if dist_total:
        by_distance = []
        for lo in sorted(dist_total):
            by_distance.append({
                'distance': int(lo),
                'n': int(dist_total[lo]),
                'accuracy': (dist_correct[lo] / dist_total[lo]
                             if dist_total[lo] else None),
            })

    if n_wrong:
        top = sorted(categories.items(), key=lambda kv: -kv[1]['count'])
        summary = ', '.join(f"{c}={v['count']}" for c, v in top if v['count'])
        logger.info(f"  [bad-case] {op_name}/{task} seed={seed}: "
                    f"{n_wrong}/{n_total} wrong ({100.0 * n_wrong / max(1, n_total):.1f}%); "
                    f"{summary}")
        shown = 0
        for c, v in top:
            if not v['count'] or c not in examples or not examples[c]:
                continue
            ex = examples[c][0]
            p_pred = ('n/a' if ex.get('pred_prob') is None
                      else format(ex['pred_prob'], '.3f'))
            p_target = ('n/a' if ex.get('target_prob') is None
                        else format(ex['target_prob'], '.3f'))
            meta_show = {
                k: v2 for k, v2 in ex.items()
                if k not in ('input_ids', 'target', 'pred', 'cue_pos',
                             'pred_prob', 'target_prob', 'margin',
                             'target_rank', 'entropy_nats', 'entropy_ratio')}
            logger.info(f"  [bad-case] e.g. {c}: target={ex['target']} "
                        f"pred={ex['pred']} cue_pos={ex['cue_pos']} "
                        f"p_pred={p_pred} p_target={p_target} "
                        f"meta={meta_show}")
            logger.info(f"  [bad-case]   input_ids={ex['input_ids']}")
            shown += 1
            if shown >= 2:
                break
    else:
        logger.info(f"  [bad-case] {op_name}/{task} seed={seed}: no errors "
                    f"in {n_total} held-out samples")

    return {
        'op': op_name,
        'task': task,
        'n_kv': int(n_kv),
        'seed': int(seed),
        'stream': 'held-out error-analysis stream',
        'generator_seed': int(seed + _ERROR_ANALYSIS_SEED_OFFSET),
        'n_batches': int(n_batches),
        'batch': int(batch),
        'n_samples': int(n_total),
        'n_wrong': int(n_wrong),
        'error_rate': (n_wrong / n_total) if n_total else None,
        'categories': categories,
        'confidence': confidence_summary,
        'examples': examples,
        'accuracy_by_distance_bin': by_distance,
        'note': ('Errors are inspected on a stream never used for training, '
                 'validation monitoring, or the headline eval. Synthetic '
                 'labels are exact, so wrong answers are mechanism blind '
                 'spots / positional shortcuts, not label noise; use the '
                 'category mix to pick the next architecture experiment.'),
    }


__all__ = [
    '_ERROR_CATEGORY_DEFINITIONS',
    '_classify_task_errors',
    '_error_confidence',
    '_error_example',
    'analyze_errors',
]
