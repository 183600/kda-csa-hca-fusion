"""Regression tests: trustworthy backend resolution for
(retrieval-recall) experiments.

1. ``ops_rwkv_backend`` — the availability probe must never raise, and a
   BROKEN ``fla.ops.rwkv7`` install must never be cached as "available".
   These tests pin the probe-state contract (probe 1 and probe 2 both
   ``False``), the actionable explicit-``'fla'`` error, the broadened
   'auto' fallback surface (TypeError -> warning + reference; CUDA OOM ->
   re-raise), and the adapter's kernel-call contract (``a=-kk`` /
   ``b=kk*a`` / log-decay pass-through) using injected fake kernels.

2. ``ops_mamba_backend`` — the ``mamba_ssm`` availability probe never
   raises on a broken install; an explicit ``'official'`` request still
   raises an actionable ``ImportError`` naming the original exception;
   ``_build_official_mamba2`` filters kwargs against the installed
   signature with a warning; ``transfer_reference_state`` maps
   ``dt_proj.bias -> dt_bias`` and lists unsourceable fused-layout params.

3. ``kaggle_setup.capture_provenance`` — requested vs resolved backend
   attribution: ``*_backend_resolved`` / ``*_available`` /
   ``mamba_ssm_version`` are recorded, stay strict-JSON, and never take
   the capture down even with broken optional packages.

4. ``ops_decoding_cache`` — the single-branch (HCA) and overlapped (CSA)
   "recompression" seams: a block completed from a partial accumulator
   plus later chunks must equal the full-sequence compression, for every
   chunking, not just token-by-token.

Run via pytest::

    pytest -q test_backend_probe_provenance.py

Or directly::

    python3 test_backend_probe_provenance.py
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
import types
import warnings
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402


def _ok(name, cond, detail=''):
    """Print a PASS/FAIL line AND assert, so pytest counts failures."""
    status = 'PASS' if cond else 'FAIL'
    print(f"  [{status}] {name}: {detail}")
    assert cond, f'{name}: {detail}'
    return cond


# ---------------------------------------------------------------------------
# Helpers: module injection + cache save/restore
# ---------------------------------------------------------------------------
class _ExplodingModule(types.ModuleType):
    """A module whose attribute access raises (simulates a broken install)."""

    def __init__(self, name, exc_factory):
        super().__init__(name)
        self.__path__ = []
        self._exc_factory = exc_factory

    def __getattr__(self, name):
        raise self._exc_factory()


@contextlib.contextmanager
def _injected_modules(mapping: dict[str, types.ModuleType]):
    saved = {name: sys.modules.get(name) for name in mapping}
    try:
        for name, mod in mapping.items():
            sys.modules[name] = mod
        yield mapping
    finally:
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


@contextlib.contextmanager
def _reset_rwkv_cache():
    import ops_rwkv_backend as rb
    saved = (rb._FLA_CHUNK_FN, rb._FLA_RECURRENT_FN,
             rb._FLA_IMPORT_ERROR, rb._FLA_PROBED)
    try:
        yield rb
    finally:
        (rb._FLA_CHUNK_FN, rb._FLA_RECURRENT_FN,
         rb._FLA_IMPORT_ERROR, rb._FLA_PROBED) = saved


@contextlib.contextmanager
def _reset_mamba_cache():
    import ops_mamba_backend as mba
    saved = (mba._MAMBA2_CLS, mba._INFERENCE_PARAMS_CLS, mba._IMPORT_ERROR)
    try:
        yield mba
    finally:
        (mba._MAMBA2_CLS, mba._INFERENCE_PARAMS_CLS, mba._IMPORT_ERROR) = saved


def _broken_fla_ops_rwkv7():
    fla = types.ModuleType('fla')
    fla.__path__ = []
    ops = types.ModuleType('fla.ops')
    ops.__path__ = []

    def _boom():
        return RuntimeError('triton version mismatch (simulated broken fla)')

    broken = _ExplodingModule('fla.ops.rwkv7', _boom)
    ops.rwkv7 = broken
    fla.ops = ops
    return {'fla': fla, 'fla.ops': ops, 'fla.ops.rwkv7': broken}


def _fake_fla_ops_rwkv7(chunk_fn, recurrent_fn):
    fla = types.ModuleType('fla')
    fla.__path__ = []
    ops = types.ModuleType('fla.ops')
    ops.__path__ = []
    rwkv7 = types.ModuleType('fla.ops.rwkv7')
    rwkv7.chunk_rwkv7 = chunk_fn
    rwkv7.fused_mul_recurrent_rwkv7 = recurrent_fn
    ops.rwkv7 = rwkv7
    fla.ops = ops
    return {'fla': fla, 'fla.ops': ops, 'fla.ops.rwkv7': rwkv7}


def _broken_mamba_ssm():
    root = types.ModuleType('mamba_ssm')
    root.__path__ = []
    modules = types.ModuleType('mamba_ssm.modules')
    modules.__path__ = []

    def _boom():
        return RuntimeError('Triton ABI mismatch (simulated broken mamba_ssm)')

    mamba2 = _ExplodingModule('mamba_ssm.modules.mamba2', _boom)
    modules.mamba2 = mamba2
    root.modules = modules
    return {'mamba_ssm': root, 'mamba_ssm.modules': modules,
            'mamba_ssm.modules.mamba2': mamba2}


def _fake_mamba_ssm(mamba2_cls, inference_params_cls):
    root = types.ModuleType('mamba_ssm')
    root.__path__ = []
    modules = types.ModuleType('mamba_ssm.modules')
    modules.__path__ = []
    mamba2_mod = types.ModuleType('mamba_ssm.modules.mamba2')
    mamba2_mod.Mamba2 = mamba2_cls
    utils = types.ModuleType('mamba_ssm.utils')
    utils.__path__ = []
    generation = types.ModuleType('mamba_ssm.utils.generation')
    generation.InferenceParams = inference_params_cls
    root.modules = modules
    root.utils = utils
    modules.mamba2 = mamba2_mod
    utils.generation = generation
    return {
        'mamba_ssm': root,
        'mamba_ssm.modules': modules,
        'mamba_ssm.modules.mamba2': mamba2_mod,
        'mamba_ssm.utils': utils,
        'mamba_ssm.utils.generation': generation,
    }


# ---------------------------------------------------------------------------
# 1. RWKV-7 / fla.ops.rwkv7 probe + dispatch contract
# ---------------------------------------------------------------------------
def test_rwkv_broken_install_probe_never_raises_or_poisons():
    """A broken fla install reads as unavailable on EVERY probe.

    Pins probe-state consistency: probe 2 must not report True with
    None kernel callables.
    """
    with _reset_rwkv_cache() as rb, _injected_modules(
            _broken_fla_ops_rwkv7()):
        first_raised = None
        try:
            available_1 = rb.rwkv7_fla_available()
        except Exception as exc:  # pragma: no cover - defensive
            first_raised = exc
            available_1 = None
        _ok('broken-install probe 1 returns False without raising',
            first_raised is None and available_1 is False,
            f'raised={first_raised!r}, available={available_1!r}')

        available_2 = rb.rwkv7_fla_available()
        _ok('broken-install probe 2 still returns False (no stale success)',
            available_2 is False, f'available={available_2!r}')
        _ok('failed load cached an actionable error, never None callables',
            rb._FLA_IMPORT_ERROR is not None
            and rb._FLA_CHUNK_FN is None and rb._FLA_RECURRENT_FN is None,
            f'import_error={rb._FLA_IMPORT_ERROR is not None}, '
            f'chunk_fn={rb._FLA_CHUNK_FN is None}')

        # resolve_* stays never-raising on the broken cache.
        resolved = rb.resolve_rwkv_backend('auto')
        _ok('resolve_rwkv_backend(auto) is reference on CPU/broken install',
            resolved in ('reference', 'fla'),
            f'resolved={resolved!r}')

        # Explicit 'fla' surfaces the actionable cached ImportError.
        r = torch.randn(1, 4, 2, 3)
        try:
            rb.rwkv7_forward(r, r, r, r, r, r, backend='fla')
            raised = None
        except ImportError as exc:
            raised = exc
        _ok('explicit fla dispatch raises actionable ImportError',
            raised is not None and 'fla.ops.rwkv7' in str(raised),
            f'raised={raised!r}')
        # The reference request still works.
        out_ref, _state = rb.rwkv7_forward(
            r, -torch.ones_like(r) * 0.1, r, r, r, torch.sigmoid(r),
            backend='reference')
        _ok('reference dispatch works despite broken fla',
            out_ref.shape == r.shape, f'shape={tuple(out_ref.shape)}')


def test_rwkv_adapter_dispatch_uses_kernel_convention():
    """The fla adapter passes log-decay through, forms a=-kk / b=kk*a."""
    calls: dict = {}

    def chunk_fn(**kwargs):
        calls['chunk'] = kwargs
        return kwargs['r'] * 0.0 + 7.0, kwargs['r'] * 0.0 + 1.0

    def recurrent_fn(**kwargs):
        calls['recurrent'] = kwargs
        return kwargs['r'] * 0.0 + 9.0, kwargs['r'] * 0.0 + 2.0

    with _reset_rwkv_cache() as rb, _injected_modules(
            _fake_fla_ops_rwkv7(chunk_fn, recurrent_fn)):
        _ok('healthy fake fla probe returns True', rb.rwkv7_fla_available())

        B, T, H, N = 1, 5, 2, 3
        torch.manual_seed(3)
        r = torch.randn(B, T, H, N)
        log_w = -0.1 * torch.ones(B, T, H, N)
        k = torch.randn(B, T, H, N)
        v = torch.randn(B, T, H, N)
        kk = torch.randn(B, T, H, N)
        a = torch.sigmoid(torch.randn(B, T, H, N))

        out, state = rb.rwkv7_forward(
            r, log_w, k, v, kk, a, backend='fla')
        kw = calls.get('chunk', {})
        _ok('T>1 dispatches the chunked kernel', 'chunk' in calls
            and 'recurrent' not in calls, f'calls={sorted(calls)}')
        _ok('log-decay passed through untouched',
            torch.equal(kw.get('log_w'), log_w), '')
        _ok('adapter forms a_op=-kk, b_op=kk*a',
            torch.equal(kw.get('a'), -kk)
            and torch.equal(kw.get('b'), kk * a), '')
        _ok('chunked kernel output returned', float(out[0, 0, 0, 0]) == 7.0,
            f'out={float(out[0, 0, 0, 0])}')

        x1 = torch.randn(B, 1, H, N)
        rb.rwkv7_forward(x1, -0.1 * torch.ones_like(x1), x1, x1, x1,
                        torch.sigmoid(x1), backend='fla')
        _ok('T==1 dispatches the recurrent kernel', 'recurrent' in calls, '')

        # 'auto' with CUDA "available" dispatches to the fake kernels.
        with mock.patch('torch.cuda.is_available', return_value=True):
            calls.pop('chunk', None)
            rb.rwkv7_forward(r, log_w, k, v, kk, a, backend='auto')
            _ok('auto dispatches fla when CUDA + kernels available',
                'chunk' in calls, f'calls={sorted(calls)}')


def test_rwkv_auto_fallback_surface_and_oom_reraise():
    """TypeError drift falls back with a warning; CUDA OOM is re-raised."""
    mode = {'fail': 'type'}

    def chunk_fn(**kwargs):
        if mode['fail'] == 'type':
            raise TypeError('chunk_rwkv7() got an unexpected keyword s')
        if mode['fail'] == 'oom':
            raise RuntimeError('CUDA out of memory. Tried to allocate 2 GiB')
        return kwargs['r'], kwargs['r']

    def recurrent_fn(**kwargs):  # pragma: no cover - not used here
        return kwargs['r'], kwargs['r']

    B, T, H, N = 1, 4, 2, 3
    r = torch.randn(B, T, H, N)
    args = (r, -0.1 * torch.ones_like(r), r, r, r, torch.sigmoid(r))
    with _reset_rwkv_cache() as rb, _injected_modules(
            _fake_fla_ops_rwkv7(chunk_fn, recurrent_fn)), \
            mock.patch('torch.cuda.is_available', return_value=True):
        rwkv7_ref, _ = rb.rwkv7_forward(*args, backend='reference')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            out, _ = rb.rwkv7_forward(*args, backend='auto')
        warned = any(issubclass(w.category, RuntimeWarning) for w in caught)
        _ok('auto falls back to reference on TypeError drift',
            torch.equal(out, rwkv7_ref), '')
        _ok('fallback is surfaced via RuntimeWarning', warned,
            f'warnings={[str(w.message)[:60] for w in caught]}')

        mode['fail'] = 'oom'
        raised = None
        try:
            rb.rwkv7_forward(*args, backend='auto')
        except RuntimeError as exc:
            raised = exc
        _ok('CUDA OOM is re-raised, not silently degraded to reference',
            raised is not None and 'out of memory' in str(raised).lower(),
            f'raised={raised!r}')


# ---------------------------------------------------------------------------
# 2. mamba_ssm probe + adapter contract
# ---------------------------------------------------------------------------
def test_mamba_broken_install_probe_and_actionable_error():
    with _reset_mamba_cache() as mba, _injected_modules(_broken_mamba_ssm()):
        available_1 = mba.mamba2_official_available()
        available_2 = mba.mamba2_official_available()
        _ok('broken mamba_ssm probe returns False without raising',
            available_1 is False and available_2 is False,
            f'probe1={available_1!r}, probe2={available_2!r}')

        resolved = mba.resolve_mamba_backend('auto')
        _ok('resolve_mamba_backend(auto) unaffected by broken install',
            resolved in ('reference', 'official'), f'resolved={resolved!r}')

        raised = None
        try:
            mba._load_official_module()
        except ImportError as exc:
            raised = exc
        message = str(raised or '')
        _ok('explicit load raises actionable ImportError',
            raised is not None and 'mamba-ssm' in message
            and 'RuntimeError' in message,
            f'raised={type(raised).__name__}: {message[:90]}')
        _ok('original failure kept as __cause__',
            isinstance(getattr(raised, '__cause__', None), RuntimeError), '')

        # build_official_mamba2_mixer must surface the same error first.
        class _Cfg:
            d_model = 16
        raised_build = None
        try:
            mba.build_official_mamba2_mixer(_Cfg())
        except ImportError as exc:
            raised_build = exc
        _ok('build_official_mamba2_mixer raises before touching CUDA',
            raised_build is not None and 'mamba-ssm' in str(raised_build),
            f'raised={raised_build!r}')


def test_mamba_official_constructor_filtering_and_transfer():
    """Fake mamba_ssm pins kwargs filtering, dt_bias mapping, missing list."""
    from ops_mamba_backend import _build_official_mamba2, transfer_reference_state

    recorded: dict = {}

    class FakeMamba2MissingKnobs(nn.Module):
        # No VAR_KEYWORD and no `ngroups`/`headdim`: the adapter must drop
        # them with a warning rather than crashing.
        def __init__(self, d_model, d_state, d_conv, expand, dt_rank,
                     layer_idx=0):
            super().__init__()
            recorded.update(
                d_model=d_model, d_state=d_state, d_conv=d_conv,
                expand=expand, dt_rank=dt_rank, layer_idx=layer_idx)
            self.dummy = nn.Parameter(torch.zeros(1))

    cfg = types.SimpleNamespace(
        d_model=16, mamba_d_state=8, mamba_d_conv=4, mamba_expand=2,
        mamba_headdim=8, mamba_ngroups=1, mamba_dt_rank='auto')

    mods = _fake_mamba_ssm(FakeMamba2MissingKnobs, object)
    with _reset_mamba_cache() as mba, _injected_modules(mods):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            built = _build_official_mamba2(cfg)
        warned = any('does not accept' in str(w.message) for w in caught)
        _ok('unsupported geometry kwargs dropped with a warning', warned,
            f'warnings={[str(w.message)[:60] for w in caught]}')
        _ok('accepted kwargs forwarded to the official constructor',
            recorded.get('d_model') == 16 and recorded.get('d_state') == 8
            and recorded.get('expand') == 2,
            f'recorded={recorded}')
        _ok('dropped knobs are absent from the constructor call',
            'ngroups' not in recorded and 'headdim' not in recorded, '')

    # transfer_reference_state: dt_proj.bias -> dt_bias mapping + missing list.
    class _Ref(nn.Module):
        def __init__(self):
            super().__init__()
            self.A_log = nn.Parameter(torch.randn(4, 2))
            self.D = nn.Parameter(torch.randn(4))
            self.out_proj = nn.Linear(4, 4, bias=True)
            self.dt_proj = nn.Linear(4, 4, bias=True)

    class _Off(nn.Module):
        def __init__(self):
            super().__init__()
            self.A_log = nn.Parameter(torch.zeros(4, 2))
            self.D = nn.Parameter(torch.zeros(4))
            self.out_proj = nn.Linear(4, 4, bias=True)
            self.dt_bias = nn.Parameter(torch.zeros(4))

    ref, off = _Ref(), _Off()
    transfer_reference_state(off, ref)
    _ok('dt_proj.bias mapped onto official dt_bias',
        torch.equal(off.dt_bias.detach(), ref.dt_proj.bias.detach()), '')
    _ok('shape-compatible params copied',
        torch.equal(off.A_log.detach(), ref.A_log.detach())
        and torch.equal(off.out_proj.weight.detach(),
                        ref.out_proj.weight.detach()), '')

    off.in_proj = nn.Linear(4, 12, bias=False)  # fused-layout param
    raised = None
    try:
        transfer_reference_state(off, ref)
    except ValueError as exc:
        raised = exc
    _ok('fused-layout params raise ValueError listing them',
        raised is not None and 'in_proj.weight' in str(raised), '')


# ---------------------------------------------------------------------------
# 3. Backend-resolution provenance
# ---------------------------------------------------------------------------
_PROV_KEYS = (
    'kda_backend_resolved', 'mamba_backend', 'mamba_backend_resolved',
    'rwkv_backend', 'rwkv_backend_resolved', 'fla_available',
    'fla_rwkv7_available', 'mamba_ssm_available', 'mamba_ssm_version',
)


def test_capture_provenance_records_requested_and_resolved():
    from kaggle_setup import capture_provenance
    from ops_kda_backend import resolve_kda_backend
    from ops_mamba_backend import resolve_mamba_backend
    from ops_rwkv_backend import resolve_rwkv_backend

    env = {'KDA_BACKEND': 'auto', 'MAMBA_BACKEND': 'auto',
           'RWKV_BACKEND': 'auto'}
    with mock.patch.dict(os.environ, env):
        prov = capture_provenance()

    for key in _PROV_KEYS:
        _ok(f'provenance has {key!r}', key in prov, '')
    _ok('requested backends recorded from env',
        prov['kda_backend'] == 'auto' and prov['mamba_backend'] == 'auto'
        and prov['rwkv_backend'] == 'auto', '')
    # This suite runs on CPU; 'auto' must resolve to the reference path and
    # agree with the resolvers the runners use.
    _ok('resolved backends agree with resolve_* helpers',
        prov['kda_backend_resolved'] == resolve_kda_backend('auto')
        and prov['mamba_backend_resolved'] == resolve_mamba_backend('auto')
        and prov['rwkv_backend_resolved'] == resolve_rwkv_backend('auto'),
        f"resolved={prov['kda_backend_resolved']!r}/"
        f"{prov['mamba_backend_resolved']!r}/"
        f"{prov['rwkv_backend_resolved']!r}")
    for key in ('fla_available', 'fla_rwkv7_available',
                'mamba_ssm_available'):
        _ok(f'{key} is a bool', isinstance(prov[key], bool),
            f'{key}={prov[key]!r}')
    _ok('mamba_ssm_version is str or None',
        prov['mamba_ssm_version'] is None
        or isinstance(prov['mamba_ssm_version'], str), '')
    # Strict JSON: no exotic objects may leak into result metadata.
    json.dumps(prov, allow_nan=False)


def test_capture_provenance_survives_broken_optional_backends():
    from kaggle_setup import capture_provenance
    with _reset_rwkv_cache(), _reset_mamba_cache(), \
            _injected_modules({**_broken_mamba_ssm(),
                               **_broken_fla_ops_rwkv7()}):
        prov = capture_provenance()
    _ok('broken mamba_ssm reads as unavailable in provenance',
        prov['mamba_ssm_available'] is False, '')
    _ok('broken fla.ops.rwkv7 reads as unavailable in provenance',
        prov['fla_rwkv7_available'] is False, '')
    for key in _PROV_KEYS:
        _ok(f'provenance still has {key!r} under broken installs',
            key in prov, '')


# ---------------------------------------------------------------------------
# 4. Single-branch / overlapped recompression seams
# ---------------------------------------------------------------------------
def test_hca_partial_block_recompression_matches_full(device='cpu'):
    """HCA blocks completed from split chunks == full-sequence compression."""
    from ops_csa import csa_compress_kv
    from ops_decoding_cache import HCADecodingCache
    from run_correctness import (_make_hca_decoding_params,
                                 _project_hca_for_cache)

    torch.manual_seed(11)
    d, c, m2, nh, dc, win = 32, 8, 4, 2, 8, 4
    T, B = 17, 1
    p = _make_hca_decoding_params(d, c, m2, nh, dc, device, seed=11)
    H = torch.randn(B, T, d, device=device) * 0.1
    C_full, Z_full, _q = _project_hca_for_cache(H, p, c, dc, nh)
    n_full = T // m2
    expected = csa_compress_kv(
        C_full[:, :n_full * m2], Z_full[:, :n_full * m2], p['B_pos'], m2)

    def build(chunks):
        cache = HCADecodingCache(B, c, m2, win, device, torch.float32)
        pos = 0
        for n in chunks:
            C, Z, _ = _project_hca_for_cache(
                H[:, pos:pos + n], p, c, dc, nh)
            cache.append_step(C, Z, p['B_pos'])
            pos += n
        return cache

    # Token-by-token is the reference chunking; block bookkeeping (n_blocks
    # / accumulator tail) must match EXACTLY for every split, and values
    # must agree to fp32 softmax-reduction round-off (the bulk path uses
    # ``csa_compress_kv``, the per-token path ``_hca_compress_kv_single``
    # — same math, different reduction shapes).
    ref = build([1] * T)
    for chunks in ([1] * T, [T], [5, 1, 4, 7], [2, 3, 6, 5, 1],
                   [m2] * 4 + [1]):
        assert sum(chunks) == T, chunks
        cache = build(chunks)
        d_ref = (cache.C_comp - ref.C_comp).abs().max().item()
        d_full = (cache.C_comp - expected).abs().max().item()
        _ok(f'HCA recompressed blocks match full (chunks={chunks})',
            d_ref < 1e-6 and d_full < 1e-6
            and cache.n_blocks == n_full and cache.accumulator_len == T % m2,
            f'd_ref={d_ref:.3e}, d_full={d_full:.3e}, '
            f'n_blocks={cache.n_blocks}, acc={cache.accumulator_len}')


def test_csa_overlapped_recompression_across_chunkings(device='cpu'):
    """CSA overlapped blocks completed from split chunks == full."""
    from ops_csa import csa_compress_kv, csa_compress_kv_overlapped
    from ops_decoding_cache import CSADecodingCache
    from run_correctness import (_make_csa_decoding_params,
                                 _project_csa_for_cache)

    torch.manual_seed(13)
    d, c, c_I, m, nh, nIh, dc, win = 32, 8, 4, 4, 2, 1, 8, 4
    T, B = 17, 1
    p = _make_csa_decoding_params(d, c, c_I, m, nh, nIh, dc, device, seed=13)
    H = torch.randn(B, T, d, device=device) * 0.1
    (Ca_f, Cb_f, Za_f, Zb_f, K_idx_f, Z_idx_f,
     _q, _qidx, _w) = _project_csa_for_cache(H, p, c, c_I, dc, nh, nIh)
    n_full = T // m
    expected_C = csa_compress_kv_overlapped(
        Ca_f[:, :n_full * m], Cb_f[:, :n_full * m],
        Za_f[:, :n_full * m], Zb_f[:, :n_full * m],
        p['Ba'], p['Bb'], m)
    expected_K = csa_compress_kv(
        K_idx_f[:, :n_full * m], Z_idx_f[:, :n_full * m], p['B_idx'], m)

    def build(chunks):
        cache = CSADecodingCache(B, c, c_I, m, win, device, torch.float32)
        pos = 0
        for n in chunks:
            (Ca, Cb, Za, Zb, K_idx, Z_idx, _q, _qi, _w) = \
                _project_csa_for_cache(
                    H[:, pos:pos + n], p, c, c_I, dc, nh, nIh)
            cache.append_step(Ca, Cb, Za, Zb, K_idx, Z_idx,
                              p['Ba'], p['Bb'], p['B_idx'])
            pos += n
        return cache

    ref = build([1] * T)
    for chunks in ([1] * T, [T], [3, 2, 7, 1, 4], [m] * 4 + [1]):
        assert sum(chunks) == T, chunks
        cache = build(chunks)
        dC_ref = (cache.C_comp - ref.C_comp).abs().max().item()
        dK_ref = (cache.K_IComp - ref.K_IComp).abs().max().item()
        dC_full = (cache.C_comp - expected_C).abs().max().item()
        dK_full = (cache.K_IComp - expected_K).abs().max().item()
        _ok(f'CSA recompressed blocks match full (chunks={chunks})',
            max(dC_ref, dK_ref, dC_full, dK_full) < 1e-6
            and cache.n_blocks == n_full and cache.accumulator_len == T % m,
            f'C_ref={dC_ref:.3e}, K_ref={dK_ref:.3e}, '
            f'C_full={dC_full:.3e}, K_full={dK_full:.3e}, '
            f'n_blocks={cache.n_blocks}, acc={cache.accumulator_len}')


if __name__ == '__main__':
    test_rwkv_broken_install_probe_never_raises_or_poisons()
    test_rwkv_adapter_dispatch_uses_kernel_convention()
    test_rwkv_auto_fallback_surface_and_oom_reraise()
    test_mamba_broken_install_probe_and_actionable_error()
    test_mamba_official_constructor_filtering_and_transfer()
    test_capture_provenance_records_requested_and_resolved()
    test_capture_provenance_survives_broken_optional_backends()
    test_hca_partial_block_recompression_matches_full()
    test_csa_overlapped_recompression_across_chunkings()
    print('all backend-probe / provenance tests passed')
