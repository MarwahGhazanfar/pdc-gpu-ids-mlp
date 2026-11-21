# Branch and Tag Strategy

## Branches

| Branch | Purpose | Rules |
|---|---|---|
| `main` | Stable, always builds and runs | Protected. Changes only by pull request reviewed by the other member. Receives tagged milestones only. |
| `develop` | Integration of completed features | Feature branches merge here first. |
| `feature/<topic>` | One focused piece of work | Short-lived; deleted after merge. |

Planned feature branches:
`feature/preprocess`, `feature/serial-mlp`, `feature/gradient-check`, `feature/openmp-reduction`,
`feature/cuda-gemm`, `feature/bench-scripts`, `feature/report`.

## Workflow

1. Branch from `develop`: `git checkout develop && git checkout -b feature/serial-mlp`
2. Commit small, working changes with clear messages.
3. Open a pull request into `develop`; the other team member reviews it.
4. When a milestone is stable and validated, open a pull request `develop` -> `main` and tag it.

## Commit messages

Format: `<area>: <what changed>` — for example `serial: add softmax and cross-entropy loss`.
Areas: `preprocess`, `serial`, `openmp`, `cuda`, `bench`, `docs`, `repo`.
Both members should have visible commits.

## Tags

| Tag | Meaning |
|---|---|
| `v0.0-scope` | Phase 0 scope pack and repository skeleton |
| `v0.1-serial` | Serial baseline validated (gradient checks, baseline metrics) |
| `v0.2-openmp` | OpenMP version validated and thread-scaling results |
| `v0.3-cuda` | CUDA version validated on Colab T4 |
| `v1.0-final` | Final code, results and report |

Tags are annotated: `git tag -a v0.1-serial -m "Serial baseline validated"`.

## Rules

- Never commit dataset files or large binaries (see `.gitignore`).
- `main` must build before anything is tagged.
- Every benchmark result in `results/` has a matching run manifest in `bench/`.
