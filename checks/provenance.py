"""Correctness checks: reproducibility provenance scope."""

from __future__ import annotations

import tempfile

from ._shared import *  # noqa: F401,F403


def test_source_identity_scope(device='cpu'):
    """Reference-paper text is data, not executable code identity.

    ``source_tree_sha256`` is documented as a code/config digest, but the
    ``.txt`` suffix allowlist (needed for dependency manifests) used to pull
    the arXiv reference papers into it.  Editing a paper then looked like code
    drift.  The digest now excludes the arXiv filename form while ordinary
    ``.txt`` manifests stay in scope.
    """
    logger.info("Test: source identity excludes reference-paper text")
    from kaggle_setup import (_is_source_file, capture_source_identity,
                              source_tree_sha256)

    classify_ok = (
        not _is_source_file('2510.26692v2.txt')
        and not _is_source_file('2606.19348v1.txt')
        and not _is_source_file('references/2510.26692v2.txt')
        and _is_source_file('requirements-ci.txt')
        and _is_source_file('constraints.txt')
        and _is_source_file('notes.txt')
    )
    with tempfile.TemporaryDirectory(prefix='xuen-provenance-') as root:
        def write(name, text):
            with open(f'{root}/{name}', 'w', encoding='utf-8') as f:
                f.write(text)

        write('module.py', 'X = 1\n')
        write('requirements.txt', 'torch>=2\n')
        write('2510.26692v2.txt', 'paper v1\n')
        write('notes.txt', 'config notes\n')
        before = source_tree_sha256(root)
        identity = capture_source_identity(root)
        write('2510.26692v2.txt', 'paper v2\n')
        paper_excluded = source_tree_sha256(root) == before
        write('notes.txt', 'changed config notes\n')
        text_still_counted = source_tree_sha256(root) != before

    return [
        _ok('source identity classifies papers as data', classify_ok, ''),
        _ok('source file count excludes reference papers',
            identity['source_file_count'] == 3,
            f"count={identity['source_file_count']} (expected 3)"),
        _ok('editing a reference paper leaves the code digest unchanged',
            paper_excluded, ''),
        _ok('ordinary .txt manifests remain part of the code digest',
            text_still_counted, ''),
    ]


__all__ = [
    'test_source_identity_scope',
]
