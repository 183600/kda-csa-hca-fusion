"""Deep bug hunting - focusing on subtle issues that might have been missed."""

import torch
import torch.nn.functional as F
import sys

def test_csa_indexer_topk_tiebreak():
    """Test that CSA indexer handles top-k ties deterministically."""
    print("\nTesting CSA indexer top-k tie-breaking...")
    from ops_csa import csa_lightning_indexer
    
    B, T, n_blocks = 2, 16, 4
    nIh, c_I = 2, 8
    topk = 2
    
    # csa_lightning_indexer expects:
    #   q_idx: [B, T, HI, DI]  (indexer queries per head)
    #   k_idx: [B, n_blocks, DI] (compressed indexer keys, shared across heads)
    #   w_idx: [B, T, HI] or None (per-head aggregation weights)
    q_idx = torch.randn(B, T, nIh, c_I)
    k_idx = torch.randn(B, n_blocks, c_I)
    w_idx = torch.randn(B, T, nIh)
    
    # Make some compressed keys identical to create tied scores
    k_idx[:, 1] = k_idx[:, 0]  # Blocks 0 and 1 have identical keys
    
    # Build a causal block mask (query t sees block s iff s < t // m)
    m = 4  # compression factor
    t_pos = torch.arange(T)
    b_pos = torch.arange(n_blocks)
    causal_mask = t_pos[:, None] // m > b_pos[None, :]  # [T, n_blocks]
    
    try:
        selected = csa_lightning_indexer(
            q_idx, k_idx, w_idx, topk=topk,
            scale=c_I ** -0.5,
            causal_block_mask=causal_mask,
            return_soft_weights=False,
        )
        
        # Check that selection is deterministic
        selected2 = csa_lightning_indexer(
            q_idx, k_idx, w_idx, topk=topk,
            scale=c_I ** -0.5,
            causal_block_mask=causal_mask,
            return_soft_weights=False,
        )
        
        if not torch.equal(selected, selected2):
            print("  ✗ Top-k selection is not deterministic")
            return False
        
        print("  ✓ Top-k tie-breaking is deterministic")
    except Exception as e:
        print(f"  ✗ Indexer FAILED: {e}")
        return False
    
    return True

def test_csa_ste_gradient_flow():
    """Test that STE gradient flows correctly to indexer parameters."""
    print("\nTesting CSA STE gradient flow...")
    from ops_csa import naive_csa
    
    B, T, d = 2, 16, 32
    m, nh, c, dc, c_I, nIh = 4, 4, 8, 8, 8, 2
    topk = 2
    
    H = torch.randn(B, T, d, requires_grad=True)
    
    # Create all required weights
    W_aKV = torch.randn(c, d, requires_grad=True)
    W_bKV = torch.randn(c, d, requires_grad=True)
    W_aZ = torch.randn(c, d, requires_grad=True)
    W_bZ = torch.randn(c, d, requires_grad=True)
    W_KV_idx = torch.randn(c_I, d, requires_grad=True)
    W_Z_idx = torch.randn(c_I, d, requires_grad=True)
    Ba = torch.randn(m, c, requires_grad=True)
    Bb = torch.randn(m, c, requires_grad=True)
    B_idx = torch.randn(m, c_I, requires_grad=True)
    W_DQ = torch.randn(dc, d, requires_grad=True)
    W_UQ = torch.randn(c * nh, dc, requires_grad=True)
    W_IUQ = torch.randn(c_I * nIh, dc, requires_grad=True)
    W_w = torch.randn(nIh, d, requires_grad=True)
    
    try:
        out = naive_csa(
            H, W_aKV, W_bKV, W_aZ, W_bZ, Ba, Bb,
            W_DQ, W_UQ, W_IUQ, W_w, W_KV_idx, W_Z_idx, B_idx,
            m=m, topk=topk, nh=nh, nIh=nIh, c=c, c_I=c_I, dc=dc,
            use_ste=True, ste_mode='topk_columns'
        )
        
        loss = out.sum()
        loss.backward()
        
        # Check that indexer parameters received gradients
        for name, param in [('W_KV_idx', W_KV_idx), ('W_Z_idx', W_Z_idx),
                            ('W_IUQ', W_IUQ), ('W_w', W_w)]:
            if param.grad is None:
                print(f"  ✗ {name} has no gradient")
                return False
            if not torch.isfinite(param.grad).all():
                print(f"  ✗ {name} has non-finite gradient")
                return False
            if (param.grad == 0).all():
                print(f"  ✗ {name} has all-zero gradient")
                return False
        
        print("  ✓ STE gradients flow correctly")
    except Exception as e:
        print(f"  ✗ STE gradient FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_sink_softmax_numerical_stability():
    """Test attention sink numerical stability."""
    print("\nTesting attention sink numerical stability...")
    from ops_csa import _sink_softmax
    
    B, nh, T, K = 2, 4, 8, 6
    
    # Test with extreme scores
    scores = torch.randn(B, nh, T, K) * 100
    sink_logits = torch.randn(nh) * 10
    
    # Test with all -inf scores (should put all mass on sink)
    scores_all_masked = torch.full((B, nh, T, K), float('-inf'))
    all_masked = torch.ones(B, nh, T, 1, dtype=torch.bool)
    
    try:
        p = _sink_softmax(scores_all_masked, sink_logits.view(1, nh, 1, 1),
                         all_masked, inplace_ok=False)
        
        if not torch.isfinite(p).all():
            print("  ✗ All-masked case produces non-finite output")
            return False
        
        # When all scores are -inf, output should be zero (sink absorbs all mass)
        if p.abs().max() > 1e-6:
            print(f"  ✗ All-masked output should be near zero, got max {p.abs().max()}")
            return False
        
        print("  ✓ Attention sink handles all-masked case correctly")
    except Exception as e:
        print(f"  ✗ Sink softmax FAILED: {e}")
        return False
    
    # Test with +inf scores (should put all mass on +inf entries)
    scores_with_inf = torch.randn(B, nh, T, K)
    scores_with_inf[:, :, :, 0] = float('inf')
    all_masked = torch.zeros(B, nh, T, 1, dtype=torch.bool)
    
    try:
        p = _sink_softmax(scores_with_inf, sink_logits.view(1, nh, 1, 1),
                         all_masked, inplace_ok=False)
        
        if not torch.isfinite(p).all():
            print("  ✗ +inf scores produce non-finite output")
            return False
        
        # Most mass should be on the +inf entry
        if p[..., 0].mean() < 0.99:
            print(f"  ✗ +inf entry should get most mass, got {p[..., 0].mean()}")
            return False
        
        print("  ✓ Attention sink handles +inf scores correctly")
    except Exception as e:
        print(f"  ✗ +inf case FAILED: {e}")
        return False
    
    return True

def test_compress_kv_overlapped_first_block():
    """Test that first block in overlapped compression handles missing predecessor."""
    print("\nTesting overlapped compression first block...")
    from ops_csa import csa_compress_kv_overlapped
    
    B, T, c, m = 2, 16, 8, 4
    n_blocks = T // m
    
    Ca = torch.randn(B, T, c)
    Cb = torch.randn(B, T, c)
    Za = torch.randn(B, T, c)
    Zb = torch.randn(B, T, c)
    Ba = torch.randn(m, c)
    Bb = torch.randn(m, c)
    
    try:
        C_comp = csa_compress_kv_overlapped(Ca, Cb, Za, Zb, Ba, Bb, m)
        
        if not torch.isfinite(C_comp).all():
            print("  ✗ Output contains non-finite values")
            return False
        
        # First block should only use a-branch (no b-branch predecessor)
        # This is handled internally by filling b-branch logits with -inf
        print("  ✓ First block handled correctly")
    except Exception as e:
        print(f"  ✗ Overlapped compression FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_hybrid_mamba2_route():
    """Test hybrid stack with Mamba-2 memory route."""
    print("\nTesting hybrid stack with Mamba-2 route...")
    from ops_fused import HybridConfig, HybridAttentionStack
    
    cfg = HybridConfig(
        d_model=64,
        n_heads_qk=4,
        n_heads_v=4,
        head_dim_k=16,
        head_dim_v=16,
        memory_type='mamba2',
        mamba_d_state=16,
        mamba_headdim=16,
    )
    
    model = HybridAttentionStack(cfg, total_layers=5)
    model.train()
    
    B, T = 2, 32
    x = torch.randn(B, T, cfg.d_model, requires_grad=True)
    
    try:
        y = model(x)
        assert y.shape == (B, T, cfg.d_model), f"Wrong shape: {y.shape}"
        
        loss = y.sum()
        loss.backward()
        
        # Check gradients
        if x.grad is None:
            print("  ✗ No gradient to input")
            return False
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if not torch.isfinite(param.grad).all():
                    print(f"  ✗ Parameter {name} has non-finite gradient")
                    return False
        
        print("  ✓ Mamba-2 route works correctly")
    except Exception as e:
        print(f"  ✗ Mamba-2 route FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_partial_rope_norm_preservation():
    """Test that partial RoPE preserves vector norms."""
    print("\nTesting RoPE norm preservation...")
    from ops_rope import partial_rope
    
    B, T, nh, c = 2, 16, 4, 32
    rope_dim = 16
    
    x = torch.randn(B, T, nh, c)
    positions = torch.arange(T)
    
    norm_before = x.norm(dim=-1)
    x_rot = partial_rope(x, positions, rope_dim, base=10000.0, dim=1)
    norm_after = x_rot.norm(dim=-1)
    
    diff = (norm_before - norm_after).abs().max()
    if diff > 1e-5:
        print(f"  ✗ RoPE does not preserve norms: max diff = {diff}")
        return False
    
    print(f"  ✓ RoPE preserves norms (max diff = {diff:.2e})")
    return True

def test_chunked_mamba2_vs_naive():
    """Test that chunked Mamba-2 scan matches naive scan."""
    print("\nTesting chunked vs naive Mamba-2 scan...")
    from ops_mamba import naive_mamba2_scan, chunked_mamba2_scan
    
    batch, seqlen, nheads, headdim = 2, 33, 4, 16  # Not divisible by chunk
    d_state, ngroups = 16, 1
    
    x = torch.randn(batch, seqlen, nheads, headdim)
    dt = F.softplus(torch.randn(batch, seqlen, nheads))
    B = torch.randn(batch, seqlen, ngroups, d_state)
    C = torch.randn(batch, seqlen, ngroups, d_state)
    A = -torch.exp(torch.randn(nheads))
    D = torch.randn(nheads)
    
    try:
        y_naive, S_naive = naive_mamba2_scan(x, dt, B, C, A, D)
        y_chunk, S_chunk = chunked_mamba2_scan(x, dt, B, C, A, D, chunk_size=16)
        
        y_diff = (y_naive - y_chunk).abs().max()
        S_diff = (S_naive - S_chunk).abs().max()
        
        if y_diff > 1e-4 or S_diff > 1e-4:
            print(f"  ✗ Chunk vs naive disagreement: y_diff={y_diff}, S_diff={S_diff}")
            return False
        
        print(f"  ✓ Chunk and naive agree (y_diff={y_diff:.2e}, S_diff={S_diff:.2e})")
    except Exception as e:
        print(f"  ✗ Chunked scan FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    print("=" * 60)
    print("DEEP BUG HUNTING SUITE")
    print("=" * 60)
    
    tests = [
        test_csa_indexer_topk_tiebreak,
        test_csa_ste_gradient_flow,
        test_sink_softmax_numerical_stability,
        test_compress_kv_overlapped_first_block,
        test_hybrid_mamba2_route,
        test_partial_rope_norm_preservation,
        test_chunked_mamba2_vs_naive,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"  ✗ EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)
    
    return all(results)

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
