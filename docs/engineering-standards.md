# Engineering standards: the declared environment contract

*Why a reproducible result needs more than a seeded experiment, and how the
repository keeps its dependency declarations and installable surface honest.*

## The problem

A run can be perfectly seeded and still not be reproducible if the
environment used to execute it cannot be reconstructed. Three declarations
describe that environment:

* `pyproject.toml` — canonical dependency bounds and the installable
  module/package manifest;
* `requirements.txt` — the mirror used by `pip install -r`;
* `requirements-ci.txt` — the exact pins used by the CI gate.

If one file is edited without the others, the command in a paper still looks
the same while the numerical stack changes. The result's provenance records
the versions that actually ran, but it cannot tell a reader whether the
declared environment would install those versions again. That is a
reviewability problem, not a missing field.

The same issue applies to the installable surface. A module that exists in
the working tree but is absent from `pyproject.toml` still imports when the
repository root happens to be on `sys.path`; after `pip install -e .` from
another directory it does not. A result produced by the former is hard to
replay with the latter.

## What is checked

`dependency_contract.check_contract()` performs five static checks and
returns a strict-JSON report:

| Check | Contract |
|---|---|
| `module_manifest` | every non-test top-level `*.py` file is listed in `[tool.setuptools].py-modules`; no listed module is missing from the tree |
| `package_manifest` | every top-level directory with `__init__.py` is listed in `[tool.setuptools].packages` |
| `requirements_mirror` | every runtime and `dev` requirement in `pyproject.toml` appears in `requirements.txt` with the same specifier and marker; `requirements.txt` adds no undeclared package |
| `ci_pins` | every exact pin in `requirements-ci.txt` names a declared package and satisfies both the `pyproject.toml` and `requirements.txt` bounds |
| `requires_python` | the running interpreter lies inside `project.requires-python` |

The check is metadata-only: it does not resolve dependencies over the
network, import torch, or execute an experiment. The parsers are `tomllib`
(Python >= 3.11) or `tomli` (installed with pytest on Python 3.10) and the
`packaging` distribution (a pytest/pip dependency).

## Commands

```bash
# Human-readable report; exit 0 = pass, 1 = drift, 2 = parsers unavailable.
python dependency_contract.py

# Machine-readable report.
python dependency_contract.py --json

# The focused regression tests (also collected by a bare `pytest`).
pytest -q test_engineering_contract.py

# The CI gate: syntax + environment contract + fast regression suite.
python ci_gate.py
```

`ci_gate.check_environment_contract()` runs the same check before the pytest
layer. If the metadata parsers are unavailable it warns and skips the layer
instead of failing the loop; the syntax layer still runs.

`run_all.run_all()` also calls the check once at campaign start and stores the
result under `dependency_contract` in the hashed `results/summary.json`
(`status: ok | drift | unavailable`). A campaign can still run with drift so a
pre-existing environment is never destroyed, but the artifact records that the
declarations disagreed.

## Exact versions and lock files

The declared files keep **bounds** so a preinstalled CUDA stack is not
downgraded by `pip install -e .`. They do not pin every transitive package.
For a bit-exact replay, generate an environment lock from the machine that
will run the campaign and keep it with the artifact:

```bash
python -m pip freeze --all > constraints.txt
```

`constraints.txt` is intentionally not committed (`.gitignore` ignores
`*.lock`; add the generated file explicitly if a project needs one). The
per-result provenance already records the exact installed versions, the
resolved backends, the source-tree digest and the command line; the lock file
adds the transitive closure that provenance does not enumerate.

`requirements-ci.txt` is the CI's exact pin set. It is deliberately small:
the gate only needs torch, einops, numpy, matplotlib and pytest. Optional
accelerated backends stay in the `fla` / `mamba` extras so a CPU-only
install never has to build them.

## Updating a dependency

1. Change the bound in `pyproject.toml` (canonical).
2. Mirror the same specifier and marker in `requirements.txt`.
3. If the CI pin changes, update `requirements-ci.txt` to an exact `==`
   version that satisfies the new bound.
4. Run `python dependency_contract.py` and `pytest -q test_engineering_contract.py`.
5. Run the full fast suite (`pytest -q -m "not slow"`) before treating the
   new stack as validated.

The repository now accepts pytest 9 (`>=7.0,<10.0`), matching the pinned CI
gate and the current test runner. The earlier `<9.0` bound was inconsistent
with `requirements-ci.txt`; the contract check prevents that class of drift
from recurring.

## What this does not prove

* The check is static. It proves the declarations agree, not that a package
  wheel exists for every platform or that optional GPU packages import.
* `requirements-ci.txt` pins the gate environment, not the experiment host.
  A Kaggle or AutoDL image may legitimately provide a newer torch build; the
  per-result provenance records which one actually ran.
* A complete module list makes the code importable, not numerically
  reproducible. Seed control, deterministic flags, data fingerprints and
  result integrity are separate contracts documented in
  [reproducibility.md](reproducibility.md) and
  [experiment-design.md](experiment-design.md).

## Code map

* `dependency_contract.py` — parser and checks (`check_contract`, CLI).
* `ci_gate.py` — layer 2 of the continuous-loop gate.
* `run_all.py` — snapshots the contract into the campaign summary.
* `test_engineering_contract.py` — repository-clean, synthetic-drift, CLI,
  campaign-snapshot, and pre-registration-runner wiring tests.
* `pyproject.toml`, `requirements.txt`, `requirements-ci.txt` — the three
  declarations.
