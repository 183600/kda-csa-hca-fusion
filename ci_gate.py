#!/usr/bin/env python3
"""CI gate for the kda-csa-hca-fusion continuous loop.

Exit 0 = pass. Any other exit code = the calling workflow must discard the
phase's changes (git reset --hard).

Layers
------
1. Syntax: every .py file parses (always run, no dependencies).
2. Full pytest regression suite (``run_correctness.py`` + ``test_figures.py``,
   fast subset via ``-m "not slow"``). Skipped (with a warning) when
   torch / pytest are missing so a network hiccup can never brick the loop.

--syntax-only runs layer 1 only (used by the shutdown trap).
"""
from __future__ import annotations

import ast
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent

FAILURES: list[str] = []


def fail(msg: str) -> None:
    FAILURES.append(msg)
    print(f"[gate] FAIL: {msg}")


def ok(msg: str) -> None:
    print(f"[gate] ok: {msg}")


# --------------------------------------------------------------------------
# Layer 1: syntax
# --------------------------------------------------------------------------
def check_syntax() -> None:
    n = 0
    for py in sorted(ROOT.rglob("*.py")):
        if any(part in (".git", "__pycache__", ".venv", "venv",
                        "wt103_cache", "checkpoints_lm", ".opencode")
               for part in py.parts):
            continue
        try:
            ast.parse(py.read_text(encoding="utf-8"))
            n += 1
        except SyntaxError as e:
            fail(f"syntax error in {py.relative_to(ROOT)}: {e}")
        except (OSError, UnicodeDecodeError) as e:
            fail(f"unreadable file {py.relative_to(ROOT)}: {e}")
    if n:
        ok(f"syntax: {n} python files parse")


# --------------------------------------------------------------------------
# Layer 2: pytest regression suite (needs torch+einops+matplotlib+pytest)
# --------------------------------------------------------------------------
def check_pytest() -> None:
    try:
        import torch  # noqa: F401
        import pytest  # noqa: F401
    except Exception as e:
        print(f"[gate] warn: torch/pytest unavailable ({e}); pytest layer skipped")
        return
    r = subprocess.run(
        [sys.executable, "-m", "pytest",
         "-q", "-x", "-p", "no:cacheprovider", "-m", "not slow",
         "run_correctness.py", "test_figures.py"],
        cwd=ROOT, capture_output=True, text=True, timeout=1800,
    )
    tail = (r.stdout or r.stderr).strip().splitlines()
    summary = tail[-1] if tail else "(no output)"
    if r.returncode != 0:
        fail(f"pytest suite failed ({summary})")
        for line in tail[-15:]:
            print(f"    {line}")
    else:
        ok(f"pytest: {summary}")


def main() -> int:
    syntax_only = "--syntax-only" in sys.argv
    check_syntax()
    if not syntax_only:
        check_pytest()
    if FAILURES:
        print(f"\n[gate] REJECTED — {len(FAILURES)} failure(s)")
        return 1
    print("\n[gate] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
