"""run_quality split module: tasks."""

from __future__ import annotations

from ._shared import *  # noqa: F401,F403


SMALL_MODEL_SPEC = {
    # CSA sub-layer
    'csa_c':      32,   # compressed KV dim
    'csa_cI':     16,   # indexer key dim
    'csa_dc':     64,   # down-projected query dim
    'csa_m':      4,    # compression factor
    'csa_nh':     2,    # number of attention heads
    'csa_nIh':    2,    # number of indexer heads
    'csa_topk':   2,    # top-k blocks per query (must be < n_blocks so the
                        #   sparse selection is actually exercised; with
                        #   seq_len=16 and csa_m=4 (n_blocks=4) a topk of 4
                        #   would collapse CSA to dense attention)
    'csa_sliding_window': 4,
    # HCA sub-layer (m2 >> m so HCA produces fewer compressed blocks)
    'hca_c':      32,
    'hca_dc':     64,
    'hca_m2':     8,    # heavy compression (2x CSA's m=4)
    'hca_nh':     2,
    'hca_sliding_window': 4,
    # Partial RoPE (DeepSeek-V4 §2.3.3) for the CSA/HCA core attention —
    # paper-faithful default: rotate the last 64 dims of the queries /
    # compressed KV entries / sliding-window keys and inverse-rotate the
    # core-attention output. Clamped to the head dim (c=32 -> rope_dim
    # 32) for this small spec. Set to None/0 to disable.
    'csa_rope_dim': 64,
    'hca_rope_dim': 64,
    'rope_base':  10000.0,
}


def make_mqar_batch(batch: int, seq_len: int, n_kv: int, vocab: int,
                    embed: nn.Embedding, device=None, generator=None,
                    return_meta: bool = False):
    """Build an MQAR batch with *learnable* embeddings.

    Layout per sequence (length ``seq_len``):
      positions [0, 2*n_kv): alternating (key, value) tokens
      positions [2*n_kv, seq_len-1): distractor tokens (sampled from the
        vocab ids that are neither a key nor a value, so the answer token
        never appears redundantly in the filler)
      position  seq_len-1: a cue key (repeat of one of the keys)
    Target = the value that was paired with the cued key.

    Fully vectorized: all ``batch`` sequences are constructed in parallel
    via a single argsort-based per-row permutation and advanced indexing,
    avoiding the per-sample Python loop and ``.item()`` calls that broke
    async kernel launches.

    ``generator`` (optional ``torch.Generator``) makes the batch generation
    reproducible and INDEPENDENT of the global RNG. This matters for
    multi-operator comparisons: each operator has a different parameter
    count, so it consumes a different number of global-RNG draws during
    init. Without a dedicated generator, the *same* seed produced
    *different* training batches across operators — a silent confound in
    the multi-seed CI. Pass a per-seed generator to guarantee that, for a
    given seed, all operators see the same training data.

    ``return_meta`` (keyword-only, default False) preserves the historical
    3-tuple return ``(x_emb, target, cue_pos)``. When True, a fourth element
    is returned: ``{'key_idx': j}`` where ``j`` is the ``[batch]`` tensor of
    which KV pair (0..n_kv-1) was cued for each sequence. Downstream
    mechanism diagnostics (e.g. ``run_mechanism_ablation``'s indexer
    gold-block recall) need to localize the cued pair WITHOUT re-deriving
    the task layout, so the index is surfaced here rather than duplicated
    in the caller.
    """
    if device is None:
        device = embed.weight.device

    # NOTE: use ``raise ValueError`` (NOT ``assert``) so the checks
    # survive ``python -O`` / ``PYTHONOPTIMIZE=1`` — ``assert`` statements
    # are silently stripped under optimization, which would re-expose the
    # silent shape corruption / trivially-solvable-task bugs these guards
    # are meant to prevent. Mirrors the convention established in ops_kda.py.
    if n_kv < 1:
        raise ValueError(f"n_kv={n_kv} must be >= 1")
    # Guard against silent shape corruption: if 2*n_kv exceeds vocab, the
    # argsort slice returns fewer than 2*n_kv ids and keys/vals end up with
    # mismatched shapes ([batch, vocab//2] vs the expected [batch, n_kv]).
    # If 2*n_kv >= seq_len, the cue at position seq_len-1 would overwrite a
    # KV pair (or fall inside the KV region), making the task trivially
    # solvable or unsolvable.
    if 2 * n_kv > vocab:
        raise ValueError(
            f"2*n_kv={2*n_kv} must be <= vocab={vocab} (need 2*n_kv distinct ids)")
    if 2 * n_kv >= seq_len:
        raise ValueError(
            f"2*n_kv={2*n_kv} must be < seq_len={seq_len} (need room for cue token)")

    cue_pos = seq_len - 1
    n_noise = seq_len - 2 * n_kv - 1       # filler slots between KV region and cue

    # One uniformly-random permutation of [0, vocab) per sequence, obtained
    # by argsort of iid uniform keys (ties have measure zero in float32).
    # Slicing the first 2*n_kv columns yields 2*n_kv distinct ids per row.
    pair_ids = torch.rand(batch, vocab, device=device, generator=generator).argsort(dim=-1)
    keys = pair_ids[:, 0:2 * n_kv:2]   # [batch, n_kv]
    vals = pair_ids[:, 1:2 * n_kv:2]   # [batch, n_kv]

    # Random noise base (covers noise positions; KV positions overwritten below).
    # Noise MUST be drawn only from the distractor ids — the ids in [0, vocab)
    # that are neither a key nor a value for that sequence. Drawing the filler
    # from the full vocabulary lets the cued *answer* token reappear in the
    # noise as a redundant copy that token-level operators (softmax/KDA) can
    # copy without resolving the key-value association, while the block-
    # compressed operators (CSA/HCA) only see it after compression/dilution —
    # a structural difficulty mismatch that would bias the reported ranking.
    distractor = pair_ids[:, 2 * n_kv:]          # [batch, vocab - 2*n_kv]
    if n_noise > 0:
        if 2 * n_kv >= vocab:
            raise ValueError(
                f"vocab={vocab} leaves no distractor ids for {n_noise} noise "
                f"slots; need vocab > 2*n_kv={2 * n_kv}")
        x = torch.empty(batch, seq_len, dtype=torch.long, device=device)
        noise_draw = torch.randint(
            0, distractor.shape[1], (batch, n_noise),
            device=device, generator=generator)
        x[:, 2 * n_kv:cue_pos] = distractor.gather(1, noise_draw)
    else:
        x = torch.empty(batch, seq_len, dtype=torch.long, device=device)

    # Place (key, value) pairs at positions [0, 2*n_kv) via broadcast
    # advanced indexing: x[b_idx, even_pos] has shape [batch, n_kv].
    b_idx = torch.arange(batch, device=device).unsqueeze(-1)          # [batch, 1]
    even_pos = torch.arange(0, 2 * n_kv, 2, device=device)            # [n_kv]
    odd_pos = torch.arange(1, 2 * n_kv, 2, device=device)             # [n_kv]
    x[b_idx, even_pos] = keys
    x[b_idx, odd_pos] = vals

    # Vectorized cue selection: pick one key index per sequence in parallel.
    j = torch.randint(0, n_kv, (batch,), device=device, generator=generator)  # [batch]
    row = torch.arange(batch, device=device)                          # [batch]
    x[:, cue_pos] = keys[row, j]
    target = vals[row, j]

    # Apply embedding (gradients flow to the embedding table).
    x_emb = embed(x)
    if return_meta:
        # ``input_ids`` is included so downstream bad-case analysis can
        # decode an error in task terms (which key was cued, what the
        # other values were) without re-deriving the layout.
        return x_emb, target, cue_pos, {'key_idx': j, 'input_ids': x}
    return x_emb, target, cue_pos


def make_kv_update_batch(batch: int, seq_len: int, n_kv: int, vocab: int,
                         embed: nn.Embedding, device=None, generator=None,
                         return_meta: bool = False):
    """Key-value UPDATE task: same key re-presented with a NEW value.

    Layout per sequence (length ``seq_len``):
      positions [0, 2*n_kv):          (key, value_old) pairs
      positions [2*n_kv, 4*n_kv):     the SAME keys in a shuffled order,
                                       each paired with value_new
      positions [4*n_kv, seq_len-1):  distractor tokens (never a key or
                                       any old/new value)
      position  seq_len-1:            a cue key (repeat of one key)
    Target = the NEWEST value paired with the cued key. The model must
    overwrite the earlier association, i.e. memory UPDATE + interference
    handling, rather than plain recall.

    Constraints: ``3*n_kv <= vocab`` (keys + old/new values must be
    distinct) and ``4*n_kv < seq_len`` (room for the cue). ``return_meta``
    additionally returns ``{'key_idx': j}`` like ``make_mqar_batch``.
    """
    if device is None:
        device = embed.weight.device
    if n_kv < 1:
        raise ValueError(f"n_kv={n_kv} must be >= 1")
    if 3 * n_kv > vocab:
        raise ValueError(
            f"3*n_kv={3 * n_kv} must be <= vocab={vocab} (need n_kv keys "
            f"plus 2*n_kv distinct old/new values)")
    if 4 * n_kv >= seq_len:
        raise ValueError(
            f"4*n_kv={4 * n_kv} must be < seq_len={seq_len} (old pairs + "
            f"updated pairs + cue token)")

    cue_pos = seq_len - 1
    n_noise = seq_len - 4 * n_kv - 1
    # The filler must come from ids that are neither keys nor old/new
    # values; an exactly-full vocabulary leaves no such ids, and drawing
    # ``randint(0, 0, ...)`` would crash instead of reporting the bad
    # geometry.
    if n_noise > 0 and 3 * n_kv >= vocab:
        raise ValueError(
            f"vocab={vocab} leaves no distractor ids for {n_noise} noise "
            f"slots; need vocab > 3*n_kv={3 * n_kv} when seq_len exceeds "
            f"the paired region plus the cue.")

    perm = torch.rand(batch, vocab, device=device,
                      generator=generator).argsort(dim=-1)
    keys = perm[:, 0:n_kv]                    # [batch, n_kv]
    vals_old = perm[:, n_kv:2 * n_kv]         # [batch, n_kv]
    vals_new = perm[:, 2 * n_kv:3 * n_kv]     # [batch, n_kv]
    distractor = perm[:, 3 * n_kv:]           # excludes keys and ALL values

    # Shuffled second presentation: the cued key must be found by CONTENT,
    # not by position alignment with the first presentation.
    order2 = torch.rand(batch, n_kv, device=device,
                        generator=generator).argsort(dim=-1)
    keys2 = keys.gather(1, order2)
    vals2 = vals_new.gather(1, order2)

    x = torch.empty(batch, seq_len, dtype=torch.long, device=device)
    if n_noise > 0:
        noise_draw = torch.randint(
            0, distractor.shape[1], (batch, n_noise),
            device=device, generator=generator)
        x[:, 4 * n_kv:cue_pos] = distractor.gather(1, noise_draw)

    b_idx = torch.arange(batch, device=device).unsqueeze(-1)
    x[b_idx, torch.arange(0, 2 * n_kv, 2, device=device)] = keys
    x[b_idx, torch.arange(1, 2 * n_kv, 2, device=device)] = vals_old
    x[b_idx, torch.arange(2 * n_kv, 4 * n_kv, 2, device=device)] = keys2
    x[b_idx, torch.arange(2 * n_kv + 1, 4 * n_kv, 2, device=device)] = vals2

    j = torch.randint(0, n_kv, (batch,), device=device, generator=generator)
    row = torch.arange(batch, device=device)
    x[:, cue_pos] = keys[row, j]
    # Target = the NEWEST value paired with the cued key. ``vals_new`` is
    # indexed by the ORIGINAL key index (the same index space as ``keys``
    # and ``j``), so the answer is directly ``vals_new[row, j]``. Note that
    # ``order2`` maps second-presentation slots -> original key indices, so
    # ``order2[row, j]`` would instead select the new value of whichever
    # *other* key happens to occupy slot ``j`` (the inverse permutation
    # ``order2.argsort(-1)`` is what locates the cued key's slot).
    target = vals_new[row, j]

    x_emb = embed(x)
    if return_meta:
        return x_emb, target, cue_pos, {'key_idx': j, 'input_ids': x}
    return x_emb, target, cue_pos


def make_long_range_batch(batch: int, seq_len: int, n_kv: int, vocab: int,
                          embed: nn.Embedding, device=None, generator=None,
                          return_meta: bool = False):
    """Long-range retrieval: ONE (key, value) pair at a random distance.

    Layout per sequence: a single (key, value) pair at a uniformly random
    position ``p in [0, seq_len-3]``, distractors everywhere else, and the
    cue key at the final position. Target = the paired value.
    ``distance = (seq_len-1) - p`` spans ``[2, seq_len-1]`` per sequence
    (``p=0`` is the farthest placement), so accuracy-vs-distance curves
    (binned from ``return_meta``'s ``distance`` field) measure long-range
    LOCALISATION, not just recall.

    ``n_kv`` is accepted for signature compatibility and ignored (the
    distance is the controlled variable, not the pair count).
    """
    if device is None:
        device = embed.weight.device
    if vocab < 3:
        raise ValueError(
            f"vocab={vocab} must be >= 3 (one key, one value, and at least "
            f"one distractor id)")
    if seq_len < 4:
        raise ValueError(
            f"seq_len={seq_len} must be >= 4 (key + value + at least one "
            f"distractor + cue)")

    cue_pos = seq_len - 1
    perm = torch.rand(batch, vocab, device=device,
                      generator=generator).argsort(dim=-1)
    key = perm[:, 0]                # [batch]
    val = perm[:, 1]                # [batch]
    distractor = perm[:, 2:]

    # Distractor base everywhere, then overwrite the pair + cue.
    noise_draw = torch.randint(
        0, distractor.shape[1], (batch, seq_len - 1),
        device=device, generator=generator)
    x = torch.empty(batch, seq_len, dtype=torch.long, device=device)
    x[:, :cue_pos] = distractor.gather(1, noise_draw)

    p = torch.randint(0, seq_len - 2, (batch,),
                      device=device, generator=generator)
    row = torch.arange(batch, device=device)
    x[row, p] = key
    x[row, p + 1] = val
    x[:, cue_pos] = key
    target = val

    x_emb = embed(x)
    if return_meta:
        return x_emb, target, cue_pos, {
            'distance': (cue_pos - p), 'key_pos': p, 'input_ids': x}
    return x_emb, target, cue_pos


def make_multi_agg_batch(batch: int, seq_len: int, n_kv: int, vocab: int,
                         embed: nn.Embedding, device=None, generator=None,
                         return_meta: bool = False):
    """Multi-site aggregation: predict the MAJORITY category.

    The sequence is a bag of ``vocab``-category tokens (length
    ``seq_len``); the target is the majority category. The answer depends
    on MANY distant positions simultaneously — no single-position
    retrieval solves it, which is exactly the capability HCA's dense
    global aggregation (and to a lesser degree any recurrent accumulator)
    is hypothesised to provide. The readout position is the final token
    (itself a category token and part of the count).

    Ties for the majority are resolved deterministically: the
    smallest-index tied class wins one flipped token from the next tied
    class, so the target is always UNIQUE and the chance level is exactly
    ``1 / vocab``.

    ``n_kv`` is accepted for signature compatibility and ignored
    (``vocab`` doubles as the class count).
    """
    if device is None:
        device = embed.weight.device
    n_classes = vocab
    if n_classes < 2:
        raise ValueError(f"vocab(=n_classes)={vocab} must be >= 2")
    if seq_len < 3:
        raise ValueError(f"seq_len={seq_len} must be >= 3")

    cue_pos = seq_len - 1
    x = torch.randint(0, n_classes, (batch, seq_len),
                      device=device, generator=generator)
    counts = torch.zeros(batch, n_classes, device=device, dtype=torch.long)
    counts.scatter_add_(1, x, torch.ones_like(x))

    max_count = counts.max(dim=1, keepdim=True).values
    tied = counts == max_count
    needs_fix = tied.sum(dim=1) > 1
    fix_rows = needs_fix.nonzero(as_tuple=True)[0]
    if fix_rows.numel() > 0:
        # winner = smallest tied class index; loser = the next tied class.
        winner = tied[fix_rows].float().argmax(dim=1)
        non_winner_tied = tied[fix_rows].clone()
        non_winner_tied[torch.arange(fix_rows.numel(), device=device),
                        winner] = False
        loser = non_winner_tied.float().argmax(dim=1)
        # Flip ONE token of the loser class to the winner: the winner then
        # leads by 2 and is the strict majority.
        match = x[fix_rows] == loser.unsqueeze(1)
        first_pos = match.float().argmax(dim=1)
        x[fix_rows, first_pos] = winner
        counts = torch.zeros(batch, n_classes, device=device,
                             dtype=torch.long)
        counts.scatter_add_(1, x, torch.ones_like(x))
    target = counts.argmax(dim=1)

    x_emb = embed(x)
    if return_meta:
        return x_emb, target, cue_pos, {
            'counts': counts.tolist(),
            'n_distinct': (counts > 0).sum(dim=1).tolist(),
            'input_ids': x,
        }
    return x_emb, target, cue_pos


SYNTHETIC_TASKS = {
    'mqar': {
        'make': make_mqar_batch, 'seq_len': 16, 'vocab': 16,
        'n_kv_swept': True,
        'description': 'associative recall of one of n_kv KV pairs '
                       '(canonical probe)',
    },
    'kv_update': {
        'make': make_kv_update_batch, 'seq_len': 16, 'vocab': 32,
        'n_kv_swept': True, 'default_n_kv': 2,
        'description': 'same key re-presented with a NEW value; retrieve '
                       'the LATEST value (memory update + interference)',
    },
    'long_range': {
        'make': make_long_range_batch, 'seq_len': 64, 'vocab': 16,
        'n_kv_swept': False, 'default_n_kv': 1,
        'description': 'single KV pair at a uniformly random distance '
                       '(accuracy-vs-distance curves from the meta field)',
    },
    'multi_agg': {
        'make': make_multi_agg_batch, 'seq_len': 33, 'vocab': 4,
        'n_kv_swept': False, 'default_n_kv': 1,
        'description': 'majority category over the whole sequence '
                       '(multi-site aggregation; no single-position '
                       'retrieval shortcut)',
    },
}


_RULE_NAMES = {
    'mqar': 'recency_key_lookup',
    'kv_update': 'recency_key_lookup',
    'long_range': 'recency_key_lookup',
    'multi_agg': 'majority_count',
}


def _rule_recency_key_lookup(input_ids, cue_pos):
    """Symbolic recency lookup over raw token ids.

    For each row: scan ``input_ids[b, :cue_pos]`` for the cue token, take
    its LAST occurrence, and predict the immediately following token.
    Falls back to token id 0 when the cue never occurs before the cue
    position (impossible for the generated tasks, kept as a total-
    function guard).
    """
    batch = input_ids.shape[0]
    preds = torch.zeros(batch, dtype=input_ids.dtype, device=input_ids.device)
    for b in range(batch):
        cue = input_ids[b, cue_pos]
        history = input_ids[b, :cue_pos]
        matches = (history == cue).nonzero(as_tuple=False)
        if matches.numel():
            i = int(matches[-1].item())
            preds[b] = input_ids[b, i + 1]
            # The token following the last cue occurrence must itself be
            # in-sequence (i + 1 <= cue_pos); for the generated layouts
            # the pair precedes the cue, so this always holds.
    return preds


def _rule_majority_count(input_ids):
    """Symbolic majority-count rule (multi_agg's exact target rule).

    Counts every token in the sequence (the readout position is part of
    the bag) and predicts the argmax class. The generator breaks ties so
    the target is always a unique strict majority; the rule therefore
    cannot be ambiguous on generated batches.
    """
    vocab = int(input_ids.max().item()) + 1
    counts = torch.zeros(input_ids.shape[0], vocab,
                         dtype=torch.long, device=input_ids.device)
    counts.scatter_add_(1, input_ids,
                        torch.ones_like(input_ids, dtype=torch.long))
    return counts.argmax(dim=1)


def evaluate_rule_baseline(task, n_kv, seq_len, vocab, device,
                           n_batches=4, batch=64):
    """Evaluate the task's non-learned rule on a dedicated RNG stream.

    Returns a JSON-safe record with the rule accuracy, the chance level,
    the stream provenance, and an explicit ROLE note. The stream seed
    offset (+8M) is disjoint from training / eval / monitoring / error
    analysis, so enabling this reference cannot change any reported
    metric. No parameters are trained and nothing is tuned: the rule is
    evaluated once on fresh batches.
    """
    make_fn = SYNTHETIC_TASKS[task]['make']
    rule = _RULE_NAMES.get(task)
    if rule is None:
        raise ValueError(f'no rule-based reference registered for task '
                         f'{task!r}')
    # Zero-initialized throwaway embedding (the rule never reads it). The
    # construction is RNG-forked so probing the reference cannot perturb
    # the global stream the sweep seeds itself from.
    with torch.random.fork_rng():
        embed = nn.Embedding(vocab, 1)
        with torch.no_grad():
            embed.weight.zero_()
    embed = embed.to(device)
    generator = make_seeded_generator(_RULE_BASELINE_SEED_OFFSET,
                                      device=device)
    correct, total = 0, 0
    with torch.no_grad():
        for _ in range(n_batches):
            _x_emb, target, cue_pos, meta = make_fn(
                batch, seq_len, n_kv, vocab, embed, device,
                generator=generator, return_meta=True)
            input_ids = meta['input_ids']
            if rule == 'majority_count':
                preds = _rule_majority_count(input_ids)
            else:
                preds = _rule_recency_key_lookup(input_ids, cue_pos)
            correct += int((preds == target).sum().item())
            total += int(target.numel())
    accuracy = (correct / total) if total else None
    return {
        'task': task,
        'n_kv': n_kv,
        'rule': rule,
        'accuracy': accuracy,
        'n_sequences': total,
        'n_batches': n_batches,
        'batch': batch,
        'chance_acc': 1.0 / vocab,
        'seed_offset': _RULE_BASELINE_SEED_OFFSET,
        'role': 'non_learned_rule_reference',
        'tunable_hyperparameters': [],
        'note': ('Task-ceiling / learnability reference only: a symbolic '
                 'rule with no parameters, evaluated once on a dedicated '
                 'RNG stream. It is NOT the comparison baseline for '
                 'superiority claims (those require beats_tuned_baseline '
                 'against the tuned learned baseline); it establishes '
                 'that the synthetic task is solvable by an exact rule '
                 'and how much of it the fixed-budget probe recovers.'),
    }


TASK_HYPOTHESES = OrderedDict([
    ('mqar', {
        'hypothesis': (
            'Associative recall requires an in-context memory that binds '
            'each key token to a later value token and retrieves it from '
            'the cue, independent of distance and of interference from '
            'other pairs.'),
        'prediction': (
            'Memoryless models stay at chance; mixers with content-based '
            'addressing (softmax, KDA delta rule, sparse CSA retrieval) '
            'rise above chance; accuracy degrades with n_kv (more '
            'interference).'),
        'falsified_if': (
            'The memoryless ``linear`` baseline beats chance (task is '
            'solvable without token mixing => confounded), or no memory '
            'operator exceeds the tuned baseline.'),
    }),
    ('kv_update', {
        'hypothesis': (
            'When the same key is re-presented with a new value, the '
            'memory must OVERWRITE the old association (delta-rule '
            'update), not merely accumulate it.'),
        'prediction': (
            'Operators with an explicit delta/erase rule (KDA) keep one '
            'value per key and answer with the latest; pure accumulators '
            'mix old and new values and lose accuracy as interference '
            'grows.'),
        'falsified_if': (
            'An operator with no overwrite mechanism matches or beats the '
            'delta-rule operator, or the error mix is dominated by '
            '``used_stale_value`` for the delta-rule operator.'),
    }),
    ('long_range', {
        'hypothesis': (
            'Long-context retrieval is a LOCALISATION problem: the model '
            'must keep early (key, value) pairs addressable as the '
            'distance grows, not only aggregate them.'),
        'prediction': (
            'O(1)-state recurrent operators (KDA / Mamba-2 / RWKV-7) show '
            'accuracy that degrades with the key-cue distance; CSA/HCA '
            'compression + sparse selection should flatten the far-distance '
            'curve because old blocks remain individually addressable.'),
        'falsified_if': (
            'Accuracy is flat or distance-independent for every operator '
            '(the probe does not exercise long-range memory), or the '
            'accuracy-vs-distance curve is non-monotone in a way the '
            'mechanism cannot explain.'),
    }),
    ('multi_agg', {
        'hypothesis': (
            'Majority-counting is a MULTI-SITE aggregation problem: no '
            'single-position retrieval solves it; the model must combine '
            'evidence from all tokens.'),
        'prediction': (
            'Dense global aggregation (HCA compressed MQA, any recurrent '
            'accumulator with slow decay) beats operators whose memory is '
            'read at one location; ``linear`` stays at chance.'),
        'falsified_if': (
            'A no-mixing or single-retrieval-only model matches the '
            'aggregation-capable operators (the task has a positional '
            'shortcut), or errors concentrate in ``echoed_last_token`` '
            '(recency copy rather than aggregation).'),
    }),
])


OPERATOR_HYPOTHESES = OrderedDict([
    ('linear', {
        'mechanism': 'no token mixing (position-wise MLP only)',
        'hypothesis': (
            'CONTROL: without inter-token information flow the target of a '
            'memory task cannot be computed.'),
        'predicted_signature': 'chance accuracy on all four probes',
    }),
    ('softmax', {
        'mechanism': 'dense causal attention (exact retrieval over all '
                     'past tokens)',
        'hypothesis': (
            'The simplest mixer is the right upper-bound baseline: full '
            'attention can represent any past-token lookup exactly, so it '
            'should win MQAR / kv_update / long_range once trained.'),
        'predicted_signature': (
            'highest or near-highest accuracy on recall probes; cost grows '
            'with T (the trade-off the efficient operators are trying to '
            'beat).'),
    }),
    ('kda', {
        'mechanism': 'recurrent delta rule with per-channel log-decay '
                     '(O(1) state)',
        'hypothesis': (
            'A delta-rule state implements key-value binding with '
            'overwrite, saturating its capacity as the number of pairs '
            'grows.'),
        'predicted_signature': (
            'strong on mqar/kv_update at small n_kv, degrading with n_kv; '
            'no distance-dependent collapse while state capacity suffices.'),
    }),
    ('mamba2', {
        'mechanism': 'selective state-space (SSD) recurrence, O(1) matrix '
                     'state',
        'hypothesis': (
            'A time-varying linear recurrence provides the same O(1)-state '
            'route with a different (input-selective, non-delta) update '
            'rule, so it should trade recall precision for stability.'),
        'predicted_signature': (
            'competitive with KDA at small n_kv; different failure mix '
            '(dilution vs overwrite) in the bad-case categories.'),
    }),
    ('rwkv7', {
        'mechanism': 'RWKV-7 delta-style recurrence (third memory route)',
        'hypothesis': (
            'An independent delta-style memory implementation should track '
            'KDA on recall probes; the trio KDA/Mamba-2/RWKV-7 separates '
            '"delta-style binding" from "the specific KDA implementation".'),
        'predicted_signature': (
            'accuracy within the KDA/Mamba-2 band at matched budget.'),
    }),
    ('csa', {
        'mechanism': 'compressed sparse attention: overlapped block KV '
                     'compression + learned top-k indexer',
        'hypothesis': (
            'Sparse per-block retrieval keeps old associations individually '
            'addressable at bounded cost, so far-distance recall should not '
            'collapse the way a fixed-size recurrent state does.'),
        'predicted_signature': (
            'flat-ish accuracy-vs-distance curve on long_range; errors '
            'concentrated in indexer selection misses (gold-block recall '
            'below 1.0) rather than in value dilution.'),
    }),
    ('hca', {
        'mechanism': 'heavy compression + dense MQA over few entries '
                     '(global aggregation)',
        'hypothesis': (
            'Global summarisation should help aggregate tasks (multi_agg) '
            'and can substitute for exact retrieval when compression is '
            'fine enough.'),
        'predicted_signature': (
            'best or near-best on multi_agg; sharper degradation than CSA '
            'on mqar/long_range because per-block detail is compressed '
            'further.'),
    }),
])


__all__ = [
    'SMALL_MODEL_SPEC',
    'make_mqar_batch',
    'make_kv_update_batch',
    'make_long_range_batch',
    'make_multi_agg_batch',
    'SYNTHETIC_TASKS',
    '_RULE_NAMES',
    '_rule_recency_key_lookup',
    '_rule_majority_count',
    'evaluate_rule_baseline',
    'TASK_HYPOTHESES',
    'OPERATOR_HYPOTHESES',
]
