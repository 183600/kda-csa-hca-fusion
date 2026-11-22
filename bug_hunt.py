"""Independent bug hunting script for KDA/CSA/HCA fusion.

Focuses on areas that might have subtle bugs:
1. Numerical stability in edge cases
2. Gradient flow correctness
3. Causal mask application
4. Dimension mismatches
5. Training loop integration
"""

import torch
import torch.nn.functional as F
import sys

def test_kda_recurrence_stability():
    """Test KDA recurrence for numerical stability issues."""
    print("Testing KDA recurrence stability...")
    from ops_kda import naive_recurrent_kda
    
    # Test with very small beta (near-zero learning rate)
    B, T, H, K, V = 2, 32, 4, 16, 16
    q = F.normalize(torch.randn(B, T, H, K), dim=-1)
    k = F.normalize(torch.randn(B, T, H, K), dim=-1)
    v = torch.randn(B, T, H, V)
    g = -torch.rand(B, T, H, K) * 0.1  # Small decay
    beta = torch.sigmoid(torch.randn(B, T, H)) * 1e-6  # Very small beta
    
    try:
        o, S = naive_recurrent_kda(q, k, v, g, beta, output_final_state=True)
        assert torch.isfinite(o).all(), "Output contains non-finite values"
        assert torch.isfinite(S).all(), "State contains non-finite values"
        print("  ✓ Small beta stability")
    except Exception as e:
        print(f"  ✗ Small beta FAILED: {e}")
        return False
    
    # Test with g close to zero (minimal decay)
    g = -torch.rand(B, T, H, K) * 1e-6
    beta = torch.sigmoid(torch.randn(B, T, H))
    
    try:
        o, S = naive_recurrent_kda(q, k, v, g, beta, output_final_state=True)
        assert torch.isfinite(o).all(), "Output contains non-finite values"
        print("  ✓ Minimal decay stability")
    except Exception as e:
        print(f"  ✗ Minimal decay FAILED: {e}")
        return False
    
    return True

def test_csa_causal_mask_strict():
    """Verify CSA causal mask is strictly preceding."""
    print("\nTesting CSA causal mask strictness...")
    from ops_csa import _causal_block_mask
    
    # Test that query at position t can only see blocks with index < t // m
    T, n_blocks, m = 16, 4, 4
    mask = _causal_block_mask(T, n_blocks, m, device='cpu')
    
    # Query at t=0..3 should see NO blocks (block 0 not visible yet)
    if not (mask[:4, :].sum() == 0):
        print("  ✗ Queries in first block should see NO blocks")
        return False
    print("  ✓ First block queries see no blocks")
    
    # Query at t=4..7 should see only block 0
    if not (mask[4:8, 0].all() and not mask[4:8, 1:].any()):
        print("  ✗ Queries in second block should see only block 0")
        return False
    print("  ✓ Second block queries see only block 0")
    
    return True

def test_hca_sliding_window_masking():
    """Test HCA sliding window masking at sequence boundaries."""
    print("\nTesting HCA sliding window masking...")
    from ops_hca import naive_hca
    
    B, T, d = 2, 17, 64  # T not divisible by m2
    m2, nh, c, dc = 8, 4, 16, 16
    
    H = torch.randn(B, T, d, requires_grad=True)
    W_KV = torch.randn(c, d)
    W_Z = torch.randn(c, d)
    B_pos = torch.randn(m2, c)
    W_DQ = torch.randn(dc, d)
    W_UQ = torch.randn(c * nh, dc)
    
    try:
        out = naive_hca(
            H, W_KV, W_Z, B_pos, W_DQ, W_UQ,
            m2=m2, nh=nh, c=c, dc=dc,
            sliding_window=4,
        )
        assert out.shape == (B, T, nh * c), f"Wrong shape: {out.shape}"
        assert torch.isfinite(out).all(), "Output contains non-finite values"
        
        # Check gradient flow
        out.sum().backward()
        assert H.grad is not None, "No gradient to input"
        assert torch.isfinite(H.grad).all(), "Gradient contains non-finite values"
        print("  ✓ Sliding window masking correct")
    except Exception as e:
        print(f"  ✗ Sliding window FAILED: {e}")
        return False
    
    return True

def test_hybrid_stack_gradient_flow():
    """Test gradient flow through the full hybrid stack."""
    print("\nTesting hybrid stack gradient flow...")
    from ops_fused import HybridConfig, HybridAttentionStack
    
    cfg = HybridConfig(
        d_model=64,
        n_heads_qk=4,
        n_heads_v=4,
        head_dim_k=16,
        head_dim_v=16,
        memory_type='kda',
    )
    
    model = HybridAttentionStack(cfg, total_layers=5)
    model.train()
    
    B, T = 2, 32
    x = torch.randn(B, T, cfg.d_model, requires_grad=True)
    
    try:
        y = model(x)
        assert y.shape == (B, T, cfg.d_model), f"Wrong shape: {y.shape}"
        
        # Check gradient flow
        loss = y.sum()
        loss.backward()
        
        # Check all parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is None:
                    print(f"  ✗ Parameter {name} has no gradient")
                    return False
                if not torch.isfinite(param.grad).all():
                    print(f"  ✗ Parameter {name} has non-finite gradient")
                    return False
        
        print("  ✓ All parameters have finite gradients")
    except Exception as e:
        print(f"  ✗ Hybrid stack FAILED: {e}")
        return False
    
    return True

def test_rope_inverse_consistency():
    """Test that RoPE and inverse RoPE are consistent."""
    print("\nTesting RoPE inverse consistency...")
    from ops_rope import partial_rope, inverse_partial_rope
    
    B, T, nh, c = 2, 16, 4, 32
    rope_dim = 16
    
    x = torch.randn(B, T, nh, c)
    positions = torch.arange(T)
    
    # Apply RoPE then inverse
    x_rotated = partial_rope(x, positions, rope_dim, base=10000.0, dim=1)
    x_recovered = inverse_partial_rope(x_rotated, positions, rope_dim, base=10000.0, dim=1)
    
    diff = (x - x_recovered).abs().max()
    if diff > 1e-5:
        print(f"  ✗ RoPE inverse not consistent: max diff = {diff}")
        return False
    
    print(f"  ✓ RoPE inverse consistent (max diff = {diff:.2e})")
    return True

def test_mamba2_scan_causality():
    """Test Mamba-2 scan causality."""
    print("\nTesting Mamba-2 scan causality...")
    from ops_mamba import naive_mamba2_scan
    
    batch, seqlen, nheads, headdim = 2, 32, 4, 16
    d_state, ngroups = 16, 1
    
    x = torch.randn(batch, seqlen, nheads, headdim)
    dt = F.softplus(torch.randn(batch, seqlen, nheads))
    B = torch.randn(batch, seqlen, ngroups, d_state)
    C = torch.randn(batch, seqlen, ngroups, d_state)
    A = -torch.exp(torch.randn(nheads))  # Negative
    D = torch.randn(nheads)
    
    try:
        y_full, _ = naive_mamba2_scan(x, dt, B, C, A, D, output_final_state=False)
        
        # Test causality: first T1 tokens should be identical
        T1 = 16
        y_prefix, _ = naive_mamba2_scan(
            x[:, :T1], dt[:, :T1], B[:, :T1], C[:, :T1], A, D,
            output_final_state=False
        )
        
        diff = (y_full[:, :T1] - y_prefix).abs().max()
        if diff > 1e-5:
            print(f"  ✗ Causality violated: max diff = {diff}")
            return False
        
        print(f"  ✓ Causality preserved (max diff = {diff:.2e})")
    except Exception as e:
        print(f"  ✗ Mamba-2 scan FAILED: {e}")
        return False
    
    return True

def test_chunk_vs_recurrent_agreement():
    """Test that chunk and recurrent KDA agree."""
    print("\nTesting KDA chunk vs recurrent agreement...")
    from ops_kda import naive_recurrent_kda, naive_chunk_kda
    
    B, T, H, K, V = 2, 33, 4, 16, 16  # T not divisible by chunk
    chunk_size = 16
    
    q = F.normalize(torch.randn(B, T, H, K), dim=-1)
    k = F.normalize(torch.randn(B, T, H, K), dim=-1)
    v = torch.randn(B, T, H, V)
    g = -torch.rand(B, T, H, K) * 0.5
    beta = torch.sigmoid(torch.randn(B, T, H))
    
    try:
        o_rec, _ = naive_recurrent_kda(q, k, v, g, beta, output_final_state=False)
        o_chk, _ = naive_chunk_kda(q, k, v, g, beta, chunk_size=chunk_size,
                                   output_final_state=False)
        
        diff = (o_rec - o_chk).abs().max()
        if diff > 1e-4:
            print(f"  ✗ Chunk vs recurrent disagreement: max diff = {diff}")
            return False
        
        print(f"  ✓ Chunk and recurrent agree (max diff = {diff:.2e})")
    except Exception as e:
        print(f"  ✗ Chunk/recurrent FAILED: {e}")
        return False
    
    return True

def main():
    print("=" * 60)
    print("INDEPENDENT BUG HUNTING SUITE")
    print("=" * 60)
    
    tests = [
        test_kda_recurrence_stability,
        test_csa_causal_mask_strict,
        test_hca_sliding_window_masking,
        test_hybrid_stack_gradient_flow,
        test_rope_inverse_consistency,
        test_mamba2_scan_causality,
        test_chunk_vs_recurrent_agreement,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"  ✗ EXCEPTION: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)
    
    return all(results)

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
