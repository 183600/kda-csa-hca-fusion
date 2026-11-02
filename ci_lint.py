"""ci_lint.py — static hygiene pre-check for kda-csa-hca-fusion.

Complements :mod:`ci_gate` (syntax parse + pytest regression suite) with
source-hygiene rules that pytest cannot see:

* bare ``except:`` — silent swallows that corrupt research results
* orphan modules — top-level library modules imported nowhere and with no
  ``__main__`` guard (dead code / forgotten entry points)
* missing module docstrings — every library module carries a docstring

Each check is a single-responsibility function registered in a plain list,
so new checks can be added without touching the runner (Open-Closed
Principle); the runner itself never changes (closed for modification).

Exit codes: 0 clean, 1 findings reported, 2 usage/read error.
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent

# Top-level modules that are deliberate *gates* rather than library code.
GATE_MODULES = frozenset({"ci_gate", "dependency_contract"})


# --------------------------------------------------------------------------
# Check registry — one (name, callable) per responsibility.  Add a new
# (name, fn) tuple here to extend; the runner below never changes.
# --------------------------------------------------------------------------
Check = tuple[str, callable]


def _list_python_files(base: pathlib.Path) -> list[pathlib.Path]:
    return sorted(
        p for p in base.rglob("*.py")
        if not any(part in (".git", "__pycache__", ".venv", "venv")
                   for part in p.parts)
    )


def _top_level_modules(base: pathlib.Path) -> dict[str, pathlib.Path]:
    """library modules at the repo root (non-test, non-conftest, non-gate)."""
    out: dict[str, pathlib.Path] = {}
    for p in base.glob("*.py"):
        name = p.stem
        if name.startswith("test_") or name == "conftest" or name in GATE_MODULES:
            continue
        out[name] = p
    return out


def _imported_names(base: pathlib.Path) -> set[str]:
    """Every top-level module name referenced by an import anywhere in the tree."""
    refs: set[str] = set()
    for p in _list_python_files(base):
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    refs.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                refs.add(node.module.split(".")[0])
    return refs


# ---- individual checks (each one responsibility) --------------------------

def check_bare_except(path: pathlib.Path) -> list[tuple[str, int]]:
    """Bare ``except:`` silently swallows every exception."""
    findings: list[tuple[str, int]] = []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        return [(f"unreadable: {exc}", 0)]
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            findings.append(("bare `except:` swallows all exceptions",
                             node.lineno))
    return findings


def check_module_docstring(path: pathlib.Path) -> list[tuple[str, int]]:
    """Every library module must carry a module docstring."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        return [(f"unreadable: {exc}", 0)]
    if ast.get_docstring(tree) is None:
        return [("module is missing a docstring", 1)]
    return []


def check_orphan_module(path: pathlib.Path,
                        modules: dict[str, pathlib.Path],
                        imported: set[str]) -> list[tuple[str, int]]:
    """A top-level module imported nowhere and with no ``__main__`` guard."""
    name = path.stem
    if name in imported:
        return []
    try:
        src = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return [(f"unreadable: {exc}", 0)]
    if "__name__" in src and "__main__" in src:
        return []  # deliberate script entry point
    return [(f"module `{name}` is imported nowhere and has no `__main__` "
             "guard (dead code)", 1)]


# registry: (display name, check callable).  The orphan check needs extra
# context, so it is wired through a small closure below.
_CHECKS: list[Check] = [
    ("bare-except", check_bare_except),
    ("module-docstring", check_module_docstring),
]


def run(base: pathlib.Path = ROOT) -> list[tuple[str, pathlib.Path, str, int]]:
    """Run every check over every top-level library module.

    Returns a list of ``(check_name, path, message, lineno)`` findings.
    """
    modules = _top_level_modules(base)
    imported = _imported_names(base)
    findings: list[tuple[str, pathlib.Path, str, int]] = []
    for name, path in modules.items():
        for check_name, fn in _CHECKS:
            for message, lineno in fn(path):
                findings.append((check_name, path, message, lineno))
        for message, lineno in check_orphan_module(path, modules, imported):
            findings.append(("orphan-module", path, message, lineno))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Static hygiene pre-check for kda-csa-hca-fusion "
                    "(complements ci_gate.py).")
    parser.add_argument("--root", default=str(ROOT),
                        help="repository root (default: this file's directory)")
    parser.add_argument("--json", action="store_true",
                        help="emit findings as strict JSON")
    args = parser.parse_args(argv)

    base = pathlib.Path(args.root)
    if not base.is_dir():
        print(f"[ci-lint] not a directory: {base}", file=sys.stderr)
        return 2

    findings = run(base)
    if args.json:
        import json
        payload = [
            {"check": name, "file": str(p.relative_to(base)),
             "message": msg, "line": lineno}
            for name, p, msg, lineno in findings
        ]
        print(json.dumps(payload, indent=2, allow_nan=False))
    else:
        if not findings:
            print("[ci-lint] clean: no hygiene findings")
        else:
            print(f"[ci-lint] {len(findings)} hygiene finding(s):")
            for name, p, msg, lineno in findings:
                rel = p.relative_to(base)
                print(f"  [{name}] {rel}:{lineno} — {msg}")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
