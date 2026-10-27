"""run_quality split module: utils."""

from __future__ import annotations

from ._shared import *  # noqa: F401,F403


def _parse_equivalence_margin(env_var, default):
    """Parse the pre-specified TOST margin from an env var.

    ``None`` / unset -> ``default``; ``0`` -> ``None`` (disable TOST);
    a malformed or non-positive value falls back to ``default`` with a
    warning. Returning ``None`` is the explicit "no equivalence testing"
    signal consumed by ``_pairwise_op_comparisons``.
    """
    raw = os.environ.get(env_var)
    if raw is None or not str(raw).strip():
        return default
    try:
        val = float(raw)
    except ValueError:
        logger.warning(
            f'{env_var}={raw!r} is not a float; using the pre-specified '
            f'default margin {default!r}')
        return default
    if val == 0.0:
        return None
    if not math.isfinite(val) or val < 0.0:
        logger.warning(
            f'{env_var}={raw!r} must be finite and >= 0; using the '
            f'pre-specified default margin {default!r}')
        return default
    return val


def _build_param_groups(*modules, weight_decay=0.01):
    """Build AdamW parameter groups with proper weight-decay exclusion.

    Standard ML practice: embeddings, biases, and LayerNorm parameters should
    NOT be weight-decayed. Weight decay on these can hurt training quality
    (e.g. shrinking the embedding table towards zero degrades representation
    quality; decaying LayerNorm affine params breaks normalization statistics).

    Grouping rules:
      * Parameters from ``nn.Embedding`` modules -> no decay
      * Parameters from ``nn.LayerNorm`` modules -> no decay
      * 1-D parameters (biases, e.g. ``nn.Conv1d.bias``, ``nn.Linear.bias``)
        -> no decay
      * All other parameters (e.g. ``nn.Linear.weight``, ``nn.Conv1d.weight``,
        ``nn.Parameter`` tensors of ndim >= 2) -> decay

    Note: we check the MODULE TYPE (not the parameter name) to identify
    embeddings and LayerNorms, because ``named_parameters()`` on a submodule
    returns names RELATIVE to that submodule (e.g. just ``weight``, not
    ``embed.weight``). A name-based heuristic would miss the embedding table.

    The CSA/HCA positional biases (``Ba``, ``Bb``, ``B_idx``, ``B_pos``) are
    2-D ``nn.Parameter`` tensors of shape ``(m, c)`` or ``(m, c_I)`` but
    function analogously to embeddings (they are lookup tables indexed by
    block-position, not weights applied to activations). Decaying them
    shrinks the table toward zero and degrades the model's ability to
    represent position-dependent compression patterns. We exclude them
    from decay via an explicit name suffix match (their ndim == 2 means
    the generic ``p.ndim <= 1`` rule does not catch them).

    Returns a list of param groups suitable for ``torch.optim.AdamW``.
    """
    # nn.Parameter names ending in any of these suffixes are positional
    # bias tables and should NOT be weight-decayed. Listed explicitly
    # because they are 2-D (so the ``ndim <= 1`` rule misses them) and
    # are not submodules of nn.Embedding/nn.LayerNorm (so the module-type
    # rule misses them too).
    _POSITIONAL_BIAS_SUFFIXES = ('Ba', 'Bb', 'B_idx', 'B_pos')

    no_decay = []
    decay = []
    for module in modules:
        # Collect the ids of parameters that belong to Embedding / LayerNorm
        # submodules. We walk the module tree once so that nested modules
        # (e.g. ``layer.norm`` inside ``ResidualAttnLayer``) are correctly
        # identified by their module type, not by a fragile name match.
        no_decay_ids = set()
        for submod in module.modules():
            if isinstance(submod, (nn.Embedding, nn.LayerNorm)):
                for p in submod.parameters(recurse=False):
                    no_decay_ids.add(id(p))
        for name, p in module.named_parameters():
            if not p.requires_grad:
                continue
            # Strip the trailing ``.weight`` / ``.bias`` etc. so the suffix
            # check below matches the leaf parameter's own name, not the
            # parent module's. e.g. ``csa_layers.0.Ba`` -> ``Ba``.
            leaf_name = name.rsplit('.', 1)[-1]
            # Exclude: Embedding/LayerNorm params (by module type), 1-D params
            # (biases), and 2-D positional-bias parameters (Ba/Bb/B_idx/B_pos,
            # which function like embeddings and would be shrunk by decay).
            is_no_decay = (
                id(p) in no_decay_ids
                or p.ndim <= 1
                or leaf_name in _POSITIONAL_BIAS_SUFFIXES
            )
            if is_no_decay:
                no_decay.append(p)
            else:
                decay.append(p)
    return [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ]


def _fmt_tstat(t, width=12, prec=2):
    """Format a t-statistic for display, None-safe.

    The one-sample t-test vs chance is undefined for n == 1 or when the
    sample standard deviation is zero; in those cases ``t_stat`` is None
    and we render ``"n/a"`` instead of a number.
    """
    if t is None:
        return 'n/a'.rjust(width)
    return f'{t:.{prec}f}'.rjust(width)


__all__ = [
    '_parse_equivalence_margin',
    '_build_param_groups',
    '_fmt_tstat',
]
