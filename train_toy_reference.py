"""Filename-compatibility wrapper for ``train_toy_reference``.

Routes to the supported training pipeline in :mod:`train_lm_autodl`,
which is built on ``ops_fused.HybridKCHAttention``.
"""

from train_lm_autodl import main


if __name__ == "__main__":
    main()
