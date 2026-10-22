"""Regression: bad-case examples must show the token just consumed.

``lm_token_diagnostics`` counts ``input_ids[:, t]`` as context for the
prediction at position ``t`` (an immediate repetition has distance 1), so
the decoded example must include that token too. Excluding it made a
``copyable_missed`` example look like the target had never appeared.
"""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')

from train_lm_autodl import analyze_lm_errors  # noqa: E402


class _FakeLM(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self._logits = logits
        self.hybrid = SimpleNamespace(reset_state=lambda: None)

    def forward(self, input_ids):
        return {'logits': self._logits}


def test_bad_case_example_includes_just_consumed_token():
    input_ids = torch.tensor([[5, 3, 3]])
    labels = torch.tensor([[3, 3, 7]])
    logits = torch.zeros(1, 3, 10)
    logits[:, :, 9] = 1.0
    logits[:, :, 3] = -5.0
    dataset = [{'input_ids': input_ids[0], 'labels': labels[0]}]
    report = analyze_lm_errors(
        _FakeLM(logits), dataset, batch_size=1, max_batches=1,
        max_examples=4, top_k=2, confidence=0.5, context_tokens=24)
    examples = report['examples']['copyable_missed']
    assert examples
    ex = examples[0]
    assert ex['target'] == 3
    assert ex['repeat_distance'] == 1
    assert ex['context_token_ids'][-1] == 3
    assert ex['context_token_ids'] == [5, 3]
