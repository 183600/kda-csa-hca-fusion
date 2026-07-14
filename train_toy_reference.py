"""Legacy compatibility wrapper for the old toy LM script.

The original toy reference depended on modules that are not part of this
repository (``config``, ``model.hybrid_model``, ``dataset``) and therefore
failed at import/runtime. Keeping a broken training entry point is dangerous:
users may run it and report results from an absent or mismatched implementation.

Use the rigorous repository implementation in :mod:`train_lm_autodl` instead.
This wrapper preserves the filename for old commands while routing to the
supported training pipeline based on ``ops_fused.HybridKCHAttention``.
"""

from train_lm_autodl import main


if __name__ == "__main__":
    main()
