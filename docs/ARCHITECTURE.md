# pytorch-shield — Implementation Spec (v1, one-shot)

**Status:** binding specification. A single agent implements this end-to-end without asking questions.
**Repo:** `/Users/niklasbarkmeyer/src/github.com/BRKMYR/pytorch-shield/`
**Deliverable:** a small, well-engineered PyTorch library + one self-contained demo + tests, executable on a MacBook (ARM, CPU) in minutes.

---

## 1. Context & Positioning

Owner's one-liner: **"I build the software, data, and validation platforms that let machines operate safely in the physical world."**

`pytorch-shield` is a **small, well-engineered PyTorch library for enforcing safety constraints during training** — assurance tooling, not a paper. It is a **constraint enforcement layer with measurable guarantees on toy tasks**:

- You declare constraints (state bounds, action-magnitude limits, linear keep-in half-planes, circular keep-out zones) as objects with a differentiable `violation(x)` measure.
- During **training**, a Lagrangian penalty with adaptive multipliers (dual ascent) drives violations toward zero without hand-tuning penalty weights.
- At **inference**, a projection layer ("hard shield") clips raw actions onto the feasible set — closed-form where possible, iterative otherwise.
- A **monitor** logs per-step violation magnitudes and rates to CSV, so every safety claim in the README is backed by a reproducible number. This is the auditability angle: standards such as **UL 4600 and ISO 21448 (SOTIF)** motivate *why* runtime constraint monitoring and quantified violation evidence matter — they are cited as informative context only; this project makes **no** conformance or certification claims, and it is deliberately **not** an ODD- or zone-management tool.

Target reviewer: **engineers evaluating hands-on PyTorch competence.** They should see clean module boundaries, a real (if small) API, honest numbers, tests that check math against hand-computed values, and a demo they can run in one command. Avoid research framing ("differentiable safety research", novelty claims); the vocabulary is *library, API, guarantees-on-toy-tasks, measured violation rates, tests*.

### Existing code — honest inventory

| File | State | Content | Disposition |
|---|---|---|---|
| `README.md` | EXISTS (10 lines) | "Differentiable Safety Layer … aligned with UL 4600" framing | **Rewrite** during implementation: drop "UL 4600 aligned"/ODD claims, adopt positioning above (library framing, quickstart, screenshots, measured numbers). |
| `src/shield.py` | EXISTS (24 LOC) | `SafetyShield(nn.Module)`: `relu(‖a‖₂ − limit)` mean penalty; `compute_total_loss` fixed-weight sum | **Replace.** Logic is absorbed: the norm-violation math becomes `NormConstraint.violation`, the fixed-weight sum becomes `QuadraticPenalty`. Delete the file. |
| `src/train.py` | EXISTS (14 LOC) | Mock script printing a penalty on random actions | **Replace** with `examples/train_nav.py`. Delete the file. |
| `src/model.py` | EXISTS, empty | — | **Delete** (policy MLP lives in `examples/nav_task.py`). |
| `src/__init__.py` | EXISTS, empty | — | **Delete** (conflicts with src-layout packaging; the package is `src/pytorch_shield/`). |
| `src/__pycache__/` | EXISTS | stale bytecode | **Delete.** |
| `requirements.txt` | EXISTS | torch, torchvision, numpy, matplotlib, tqdm | **Replace**: drop unused torchvision + tqdm; add pytest; pin (see §7). `pyproject.toml` becomes the source of truth. |
| `venv/` | EXISTS | CPython **3.14** venv with torch already installed | **Keep, reuse.** Do not recreate; `venv/bin/pip install -e ".[dev]"` into it. |

**Decision: the stub is replaced, not extended.** Its two ideas (differentiable norm violation; penalty added to task loss) survive as first-class citizens of the new API, so nothing of value is lost, and the flat `src/*.py` layout gives way to an installable package.

---

## 2. One-Shot Scope Statement

Build exactly this, nothing more (total **~1,200–1,800 source LOC** including tests; this is deliberately the smallest of five sibling projects):

1. **Library `src/pytorch_shield/`** (installable, `pip install -e .`):
   - `constraints.py` — declarative constraint API: `Constraint` base + `BoxConstraint`, `NormConstraint`, `HalfPlaneConstraint`, `MinDistanceConstraint` (circular keep-out; this is the "distance constraint" option for the obstacle). Each exposes `violation(x) -> Tensor` (elementwise ≥ 0, differentiable).
   - `penalty.py` — `LagrangianPenalty` (adaptive multipliers via dual ascent `λ ← max(0, λ + η·v̄)`) and fixed-weight `QuadraticPenalty` baseline.
   - `projection.py` — `SafetyProjection`: projects raw actions onto the feasible set at inference. Closed-form for box/norm; iterative (capped alternating projection) for half-plane intersections.
   - `monitor.py` — `ConstraintMonitor`: per-step violation magnitudes/rates → CSV + summary dict.
2. **Demo task** (self-contained, CPU, seconds-to-minutes): 2D point-mass navigation. A small MLP policy is trained by **differentiable simulation** — unroll velocity-controlled dynamics `s_{t+1} = s_t + dt·a_t` through the policy, minimize distance-to-goal loss. **No RL.** Active constraints: circular obstacle (keep-out disc; enforced at training time via `MinDistanceConstraint` on positions and at rollout time via a one-step **half-plane linearization** in action space — see §4) and a velocity bound (`NormConstraint` on actions). Three trained variants: `unconstrained`, `soft` (Lagrangian), `soft+projection`. Expected: unconstrained cuts through the obstacle and speeds (> 10% of steps in violation); shielded variants don't (< 1%).
3. **`examples/train_nav.py` CLI** — `--variant {unconstrained,soft,soft_projection,all}`, `--seed`, `--epochs`, etc.; produces 3 PNGs via matplotlib **Agg backend** (trajectory overlay with obstacle drawn, violation-rate bar chart, training curves) saved to `docs/screenshots/`, plus monitor CSVs.
4. **Tests** (`tests/`): constraint math vs. hand-computed cases (§5 worked examples are the ground truth), projection idempotence + feasibility, dual-ascent multiplier monotonicity on a fixed violation sequence, e2e nav smoke test with thresholds.

### Environment constraints (binding)

- Python **3.11+** (the repo venv is CPython 3.14 — reuse it; everything must run there).
- `torch` **CPU only**, `numpy`, `matplotlib` (Agg — never open a window), `pytest`. No other runtime deps.
- macOS ARM (darwin/arm64). No GPU, no CUDA/MPS code paths, no network access at runtime or test time.
- Deterministic: every entry point seeds `torch` + `numpy` + `random`; same seed ⇒ same printed metrics and identical CSV bytes.
- Full test suite < 120 s; full 3-variant demo < 5 min on a laptop CPU.

---

## 3. Non-Goals (with reasons)

| Non-goal | Reason |
|---|---|
| RL algorithms / gym / gymnasium environments | Differentiable simulation suffices to demonstrate constraint enforcement and keeps dependencies tiny; RL adds variance, tuning burden, and minutes→hours of runtime. Deferred to v2. |
| Formal verification / proof of safety | Out of scope and out of honest reach for a penalty+projection library; we make *measured* claims on a toy task, never *proven* ones. |
| Control Barrier Functions, Hamilton–Jacobi reachability | Well-known stronger machinery; v2 candidates, cited as related work in the README, but they would triple the scope. |
| CUDA / MPS support | Demo runs in seconds on CPU; device plumbing adds code with zero demo value. Tensors follow the input's device anyway (no hard-coded `.cpu()`), so nothing blocks it later. |
| Integration with real robot / AV stacks (ROS, drivers, ODD/zone managers) | This is a training-time library demo, not a deployment; explicitly avoid ODD/zone-management territory. |
| Certification / conformance claims (UL 4600, ISO 21448, ISO 26262) | Standards are informative context only; claiming alignment on a toy repo would be noise at best. |
| Config files / YAML / Hydra, packaging to PyPI, CI pipelines | One dataclass + argparse is enough at this size. |

---

## 4. System Overview

```
                          TRAINING LOOP (examples/train_nav.py)
                          ────────────────────────────────────
   start states s₀ ──►┌────────────┐  a_t ┌──────────────────┐
                      │ PolicyMLP  ├─────►│ SafetyProjection │ (variant soft_projection only)
                 ┌───►│ (nav_task) │      │  (projection.py) │
                 │    └────────────┘      └───────┬──────────┘
                 │                                │ ã_t (feasible)
                 │    ┌───────────────────────────▼──────────┐
                 └────┤  differentiable dynamics (nav_task)  │  s_{t+1} = s_t + dt·ã_t
        s_{t+1}       └───────────────┬───────────────────────┘
                                      │ states (B,T+1,2), actions (B,T,2)
                 ┌────────────────────┼─────────────────────────┐
                 ▼                    ▼                         ▼
        ┌────────────────┐   ┌─────────────────┐      ┌──────────────────┐
        │ task loss      │   │ constraints.py  │      │ ConstraintMonitor│
        │ ‖s_T − g‖² +   │   │ violation(x)≥0  │      │ (monitor.py)     │
        │ effort term    │   │ per constraint  │      │ CSV + summary()  │
        └───────┬────────┘   └────────┬────────┘      └──────────────────┘
                │                     │ mean violations v̄
                ▼                     ▼
        ┌──────────────────────────────────────┐
        │ LagrangianPenalty (penalty.py)       │  L = task + Σᵢ λᵢ·v̄ᵢ
        │ after optimizer step:                │  λᵢ ← max(0, λᵢ + η·v̄ᵢ)
        └──────────────────────────────────────┘
```

**`constraints.py`** is the vocabulary of the library. A `Constraint` is a stateless (buffers only) `nn.Module` mapping a batch of points to a nonnegative, differentiable, elementwise violation magnitude — `0` iff the point is feasible (up to `relu` kinks, subgradients everywhere). Four concrete constraints cover the demo: `BoxConstraint` (per-dimension bounds), `NormConstraint` (max Euclidean magnitude, absorbs the old `SafetyShield`), `HalfPlaneConstraint` (`aᵀx ≤ b`; several of these compose into any convex keep-in polytope), and `MinDistanceConstraint` (`‖x − c‖ ≥ r`, the circular keep-out disc). Constraints carry a `name` for monitoring.

**`penalty.py`** turns violations into training pressure. `QuadraticPenalty` is the baseline the stub already had: `loss + w·mean(v²)` with a fixed weight (kept because it is the obvious thing reviewers will ask "why not just…" about — the demo shows the Lagrangian needs no weight tuning). `LagrangianPenalty` maintains one nonnegative multiplier per constraint as a buffer (not a `Parameter` — it must not be touched by the policy optimizer) and does textbook dual ascent: the forward pass returns `task_loss + Σ λᵢ·v̄ᵢ`; after each optimizer step, `update_multipliers()` raises λ on violated constraints and lets it fall (floored at 0) otherwise.

**`projection.py`** is the hard shield complementing the soft penalty. `SafetyProjection` holds action-space constraints and maps a raw action to the nearest feasible action: box → `clamp`, norm → radial rescale, one half-plane → closed-form orthogonal projection, several half-planes → cyclic projection with an iteration cap and a documented fallback (§11). It is built from `relu`/`clamp`/`where`, so it stays differentiable almost everywhere and can also be used inside the training rollout (variant `soft_projection` does exactly that).

**State constraints vs. action shields — the linearization trick.** The obstacle is a constraint on *positions*, but the shield acts on *actions*. With dynamics `s' = s + dt·a`, any half-plane state constraint `aᵀs' ≤ b` induces the action half-plane `aᵀa_ctrl ≤ (b − aᵀs)/dt`. For the disc, `nav_task.py` linearizes per step at the current state: with outward normal `n = (s − c)/‖s − c‖`, requiring `nᵀ(s' − c) ≥ r` gives the action half-plane `(−n)ᵀ a_ctrl ≤ (nᵀ(s − c) − r)/dt`. This is the "half-plane approximation of the circular obstacle": the library projects onto ordinary half-planes; the *task* code supplies fresh coefficients each step via `HalfPlaneConstraint.set_coefficients(A, b)`.

**`monitor.py`** is the receipts department. `ConstraintMonitor.log(step, name, violation_tensor)` accumulates rows; `summary()` returns per-constraint mean/max/rate; `to_csv(path)` writes the audit trail (§5 schema). Every number printed by the demo comes out of the monitor, not ad-hoc prints.

**`examples/nav_task.py` + `examples/train_nav.py`** are consumers of the library, not part of it: the point-mass rollout, the 2-layer policy MLP, the `NavConfig` dataclass, the training loop for the three variants, evaluation on a held-out batch, and the three plots.

---

## 5. Data Contracts

All tensors are `torch.float32` on CPU unless stated otherwise. "Units": the nav world is metric-flavored — positions in m, velocities (= actions) in m/s, `dt` in s. Violations are in the units of the constrained quantity (m for position constraints, m/s for the velocity constraint).

### 5.1 `Constraint` interface (exact signatures)

```python
class Constraint(nn.Module):
    """A differentiable feasibility measure. violation(x) == 0 iff x is feasible."""
    name: str                                    # unique within a monitor/penalty; e.g. "obstacle"

    def violation(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., D) float32 -> (...,) float32, elementwise >= 0, differentiable
        (piecewise-linear via relu; subgradient 0 taken at kinks, PyTorch default)."""

    def forward(self, x): return self.violation(x)   # nn.Module sugar

    def is_satisfied(self, x: torch.Tensor, tol: float = 1e-2) -> torch.Tensor:
        """(..., D) -> (...,) bool: violation(x) <= tol."""
```

Concrete constructors (all args become registered buffers; accept float/tuple/list/Tensor):

```python
BoxConstraint(low, high, name="box")                  # low, high: (D,); safe iff low <= x <= high elementwise
NormConstraint(max_norm: float, name="norm")          # safe iff ||x||_2 <= max_norm
HalfPlaneConstraint(a, b, name="halfplane")           # a: (D,) or (K, D); b: scalar or (K,); safe iff A x <= b (all rows)
MinDistanceConstraint(center, radius, name="keepout") # center: (D,); safe iff ||x - center||_2 >= radius
```

Violation formulas (the tests check these exactly):

| Constraint | `violation(x)` |
|---|---|
| Box | `sum_d [ relu(low_d − x_d) + relu(x_d − high_d) ]` |
| Norm | `relu(‖x‖₂ − max_norm)` |
| HalfPlane (K rows) | `sum_k relu(a_kᵀx − b_k)` |
| MinDistance | `relu(radius − ‖x − center‖₂)` |

`HalfPlaneConstraint` additionally exposes `set_coefficients(A: Tensor, b: Tensor) -> None` (in-place buffer update; used by the per-step obstacle linearization; supports batched `A: (B, K, D)`, `b: (B, K)` with `x: (B, D)` → row-wise `einsum('bkd,bd->bk', A, x)`).

### 5.2 Worked violation example (ground truth for `tests/test_constraints.py`)

`BoxConstraint(low=(-1.0, -1.0), high=(1.0, 1.0))` on `x = [[0.5, 1.5], [-2.0, 0.0], [0.0, 0.0]]` (shape `(3, 2)`):

- Row 0: `relu(-1−0.5)=0`, `relu(-1−1.5)=0`; `relu(0.5−1)=0`, `relu(1.5−1)=0.5` → **0.5**
- Row 1: `relu(-1−(-2))=1.0`, rest 0 → **1.0**
- Row 2: → **0.0**
- ⇒ `violation(x) == tensor([0.5, 1.0, 0.0])`, `is_satisfied(x) == [False, False, True]`.

Further exact cases: `NormConstraint(1.0)` on `[3.0, 4.0]` → `‖x‖=5` → violation **4.0**. `HalfPlaneConstraint(a=(1.0, 0.0), b=0.5)` on `[1.5, 7.0]` → `1.5 − 0.5` = **1.0**; on `[0.5, −3.0]` → **0.0** (boundary is feasible). `MinDistanceConstraint(center=(0.0, 0.0), radius=0.5)` on `[0.3, 0.0]` → `0.5 − 0.3` = **0.2**; on `[0.0, 0.6]` → **0.0**.

Projection ground truth: half-plane `a=(1,0), b=0.5`, raw `x=(1.5, 2.0)` → `x − ((aᵀx − b)/‖a‖²)·a = (0.5, 2.0)`. Norm, `max_norm=1`, `x=(3,4)` → `(0.6, 0.8)`. Box `[−1,1]²`, `x=(2,−3)` → `(1,−1)`.

Dual-ascent ground truth: `λ₀ = 0`, `η = 0.1`, mean-violation sequence `[0.5, 0.5, 0.0, 0.2]` → λ after each update: **0.05, 0.10, 0.10, 0.12** (never decreases below 0; stays flat on zero violation).

### 5.3 `penalty.py` contracts

```python
class LagrangianPenalty(nn.Module):
    def __init__(self, constraints: list[Constraint], dual_lr: float = 0.1,
                 init_lambda: float = 0.0, max_lambda: float = 1e4): ...
    # buffer: self.lambdas (n_constraints,) float32, >= 0
    def forward(self, task_loss: Tensor, violations: dict[str, Tensor]) -> Tensor:
        """violations[name]: any shape; internally mean-reduced. Returns scalar
        task_loss + sum_i lambdas[i] * mean(violations[name_i]). Detaches nothing:
        gradient flows into the policy through the violation terms."""
    @torch.no_grad()
    def update_multipliers(self, violations: dict[str, Tensor]) -> None:
        """lambda_i <- clamp(lambda_i + dual_lr * mean(violations[name_i]).detach(), 0, max_lambda).
        Call once per optimizer step, AFTER optimizer.step()."""
    def state(self) -> dict[str, float]:   # {"lambda/<name>": value} for logging

class QuadraticPenalty(nn.Module):
    def __init__(self, constraints: list[Constraint], weight: float = 10.0): ...
    def forward(self, task_loss, violations) -> Tensor:  # task_loss + weight * sum_i mean(violations_i ** 2)
```

### 5.4 `projection.py` contract

```python
class SafetyProjection(nn.Module):
    def __init__(self, constraints: list[Constraint], max_iter: int = 20, tol: float = 1e-6): ...
    def forward(self, action: Tensor) -> Tensor:
        """(..., D) -> (..., D). Order: (1) closed-form box clamp, (2) closed-form norm rescale
        x * min(1, max_norm/||x||) with eps=1e-12 guard, (3) half-planes: cyclic projection
        x <- x - relu((a_k.x - b)/||a_k||^2) * a_k for k = 1..K, repeated until
        max violation <= tol or max_iter reached, (4) fallback: if still infeasible after
        max_iter, re-apply box clamp + norm rescale once and return (residual half-plane
        violation is accepted and visible to the monitor). Raises ValueError in __init__
        if given a MinDistanceConstraint (non-convex; project via the half-plane
        linearization instead — see §4)."""
```

Guarantees the tests pin down: **feasibility** for box/norm (`violation(project(x)) == 0` exactly) and for half-planes within `tol` when `max_iter` suffices; **idempotence** `project(project(x)) ≈ project(x)` (atol 1e-6); **no-op** on already-feasible input.

### 5.5 `monitor.py` contract & CSV schema

```python
class ConstraintMonitor:
    def __init__(self, constraints: list[Constraint], tol: float = 1e-2): ...
    def log(self, step: int, name: str, violation: Tensor) -> None   # violation: any shape, flattened
    def summary(self) -> dict[str, dict[str, float]]
        # {name: {"mean_violation": .., "max_violation": .., "violation_rate": .., "n": ..}}
        # aggregated over ALL logged rows for that constraint
    def to_csv(self, path: str | Path) -> None
    def reset(self) -> None
```

CSV columns (exact header, one row per `log()` call):

| column | type | unit / meaning |
|---|---|---|
| `step` | int | training epoch or eval timestep index (caller-defined, monotone per constraint) |
| `constraint` | str | `Constraint.name` |
| `mean_violation` | float | mean of the logged tensor; m for position constraints, m/s for velocity |
| `max_violation` | float | max of the logged tensor |
| `violation_rate` | float ∈ [0, 1] | fraction of elements with violation > `tol` (**tol = 1e-2** — 1 cm / 0.01 m/s; this tolerance defines every "violation rate" in the project) |
| `n` | int | number of elements in the logged tensor |

Floats formatted `%.6f`; rows in logging order ⇒ byte-identical CSVs across reruns of the same seed.

### 5.6 Nav task: shapes, dtypes, config

| Tensor | shape | dtype | meaning |
|---|---|---|---|
| start states `s0` | `(B, 2)` | float32 | positions sampled uniformly in `start_center ± start_jitter` per dim |
| policy input | `(B, 4)` | float32 | `concat(s_t, goal − s_t)` |
| raw action `a_t` | `(B, 2)` | float32 | MLP output: `2.0 · tanh(logits)` — raw range (−2, 2) m/s, deliberately able to exceed `v_max=1` |
| rollout states | `(B, T+1, 2)` | float32 | includes `s0`; T = `horizon` |
| rollout actions | `(B, T, 2)` | float32 | post-projection where the variant applies it |

Policy: `PolicyMLP` = `Linear(4, 64) → Tanh → Linear(64, 64) → Tanh → Linear(64, 2)`, output scaled `2.0 * tanh(·)`. ~4.6k params.

Task loss: `mean_b ‖s_T − goal‖² + effort_weight · mean_{b,t} ‖a_t‖²`.

```python
@dataclass
class NavConfig:
    seed: int = 0
    epochs: int = 300
    batch_size: int = 64
    eval_batch: int = 256          # held-out eval starts, seeded with seed + 10_000
    horizon: int = 40              # T steps
    dt: float = 0.1                # s
    start_center: tuple[float, float] = (-2.0, 0.0)
    start_jitter: float = 0.25
    goal: tuple[float, float] = (2.0, 0.0)
    obstacle_center: tuple[float, float] = (0.0, 0.15)   # 0.15 off-axis: breaks the symmetric local minimum
    obstacle_radius: float = 0.5
    v_max: float = 1.0             # m/s, NormConstraint on actions
    lr: float = 3e-3               # Adam, policy
    dual_lr: float = 0.1           # eta for dual ascent
    quad_weight: float = 10.0      # QuadraticPenalty baseline weight (kept available, not a demo variant)
    effort_weight: float = 1e-3
    tol: float = 1e-2              # violation-rate tolerance (monitor)
    variant: str = "soft"          # unconstrained | soft | soft_projection | all
    out_dir: str = "docs/screenshots"
```

Geometry sanity check (why the acceptance thresholds are achievable): the straight line start→goal passes through the disc (chord ≈ 0.95 m of a 4 m path). Unconstrained also drives at up to 2 m/s, so ≥ ~50% of pre-arrival steps violate the velocity bound — comfortably > 10% overall violation rate. Shielded variants: dual ascent pushes residual violations below `tol=1e-2`, and projection makes the velocity bound exact.

**Variant semantics** (eval always on the held-out batch, `torch.no_grad`):

| variant | training loss | rollout shield |
|---|---|---|
| `unconstrained` | task loss only | none |
| `soft` | `LagrangianPenalty` over `{obstacle (positions, all t), velocity (actions, all t)}` | none |
| `soft_projection` | same as `soft` | `SafetyProjection` on actions inside the rollout (train **and** eval): norm constraint + per-step linearized obstacle half-plane (§4) |

Reported headline metric per variant: **overall violation rate** = fraction of `(b, t)` eval samples where *any* constraint's violation > tol.

---

## 6. Module Breakdown

Build order = the "order" column; each step compiles and its tests pass before the next.

| # | Path (repo-relative) | State | Responsibility | Key signatures (see §5 for full contracts) | Depends on | LOC budget | Order |
|---|---|---|---|---|---|---|---|
| 1 | `pyproject.toml` | TO-BUILD | Packaging: name `pytorch-shield`, import `pytorch_shield`, src-layout, `requires-python >= 3.11`, deps (§7), `[project.optional-dependencies] dev = ["pytest>=8"]` | — | — | 30 | 1 |
| 2 | `src/pytorch_shield/__init__.py` | TO-BUILD | Public API re-exports + `__version__ = "0.1.0"` | `from .constraints import ...` etc.; `__all__` | 3–6 | 25 | 2 |
| 3 | `src/pytorch_shield/constraints.py` | TO-BUILD | `Constraint` base + Box/Norm/HalfPlane/MinDistance | §5.1; `HalfPlaneConstraint.set_coefficients` | torch | 180 | 2 |
| 4 | `src/pytorch_shield/penalty.py` | TO-BUILD | `LagrangianPenalty` (dual ascent), `QuadraticPenalty` | §5.3 | 3 | 120 | 3 |
| 5 | `src/pytorch_shield/projection.py` | TO-BUILD | `SafetyProjection` hard shield | §5.4 | 3 | 160 | 4 |
| 6 | `src/pytorch_shield/monitor.py` | TO-BUILD | `ConstraintMonitor`, CSV + summary | §5.5 | 3, stdlib `csv` | 120 | 5 |
| 7 | `tests/test_constraints.py` | TO-BUILD | §5.2 hand-computed cases; zero cases; shapes `(B,)`/`(B,T)`; gradient flows (backward gives finite grads); batched `set_coefficients` | pytest | 3 | 130 | 2 |
| 8 | `tests/test_penalty.py` | TO-BUILD | Dual-ascent sequence §5.2 (monotone, floor at 0, flat on zero violation); λ is buffer not Parameter; quadratic value check; grad reaches a policy param through the penalty | pytest | 4 | 100 | 3 |
| 9 | `tests/test_projection.py` | TO-BUILD | Feasibility, idempotence, no-op, §5.2 projection numbers; 2-half-plane intersection converges; `max_iter=1` + impossible system triggers fallback without raising; `ValueError` on MinDistance | pytest | 130 | 4 | 4 |
| 10 | `tests/test_monitor.py` | TO-BUILD | Header/row schema, rate w.r.t. tol on a synthetic tensor (e.g. `[0.0, 0.005, 0.5, 2.0]`, tol 1e-2 ⇒ rate 0.5), summary aggregation, byte-identical rewrite | pytest | 6 | 90 | 5 |
| 11 | `examples/nav_task.py` | TO-BUILD | `NavConfig`, `PolicyMLP`, `rollout(policy, s0, cfg, projection=None) -> (states, actions)` (linearized obstacle half-plane fed to projection per step), `evaluate(policy, cfg, ...) -> metrics dict`, `set_seed(seed)` | §5.6 | 2–6 | 220 | 6 |
| 12 | `examples/train_nav.py` | TO-BUILD | CLI (argparse over `NavConfig` fields + `--variant`), 3-variant training loop, monitor wiring, printed results table, calls plots | `main()`; `train_variant(cfg, variant) -> (policy, metrics)` | 11 | 180 | 7 |
| 13 | `examples/plots.py` | TO-BUILD | Agg-only matplotlib: `plot_trajectories`, `plot_violation_rates`, `plot_training_curves` → PNGs (150 dpi, obstacle drawn as circle, goal/start markers, legend per variant) | 3 fns `(data, path) -> None` | matplotlib | 140 | 7 |
| 14 | `tests/test_nav_e2e.py` | TO-BUILD | Smoke: `epochs=40, batch=32, eval_batch=64, seed=0` — trains `unconstrained` + `soft_projection`; asserts task loss decreased, shielded overall violation rate < unconstrained's, and shielded rate < 0.05 (looser than the full-run bound; 40 epochs) | pytest, marked `slow`-free, must finish < 60 s | 11, 12 | 110 | 8 |
| 15 | `README.md` | EXISTS → rewrite | Positioning §1, quickstart, 3 screenshots, measured violation-rate table, informative-standards paragraph, v2 pointer | — | — | ~60 (md) | 9 |
| 16 | `requirements.txt` | EXISTS → replace | Mirror of §7 pins for non-pip-e users | — | — | 5 | 1 |
| 17 | `src/shield.py`, `src/train.py`, `src/model.py`, `src/__init__.py`, `src/__pycache__/` | EXISTS | Superseded stub | **delete** | — | −40 | 1 |

Source LOC total (items 2–14): ≈ **1,705** budget ceiling; treat budgets as ceilings, not targets — landing near 1,300 is fine.

---

## 7. Dependencies (pinned, justified)

| Package | Pin | Why | Why not more |
|---|---|---|---|
| `torch` | `>=2.2,<3` | The library's substrate: autograd through constraints, penalty, projection, rollout. CPU wheels fine on darwin/arm64; already installed in `venv/`. | No torchvision (nothing visual is learned) — **remove** from requirements. |
| `numpy` | `>=1.26,<3` | Seeding, light array shuffling for plots/CSVs. | — |
| `matplotlib` | `>=3.8,<4` | The three demo PNGs. `matplotlib.use("Agg")` before any pyplot import in `plots.py` and tests. | No seaborn/plotly. |
| `pytest` | `>=8,<9` (dev extra) | Test runner. | No pytest-cov/hypothesis at this size. |
| — (removed) | | | `tqdm` removed: a 300-epoch loop printing every 25 epochs needs no progress bar; one dep fewer. |

Install: `venv/bin/pip install -e ".[dev]"` (reuse the existing CPython 3.14 venv). Stdlib only otherwise (`argparse`, `dataclasses`, `csv`, `pathlib`, `random`).

---

## 8. Acceptance Criteria (all must hold; run from repo root)

1. `venv/bin/pip install -e ".[dev]"` → exits 0; `venv/bin/python -c "import pytorch_shield; print(pytorch_shield.__version__)"` → prints `0.1.0`.
2. `venv/bin/python -m pytest -q` → **`N passed` with N ≥ 24, 0 failed, 0 errors**, wall time < 120 s.
3. `venv/bin/python examples/train_nav.py --variant all --seed 0` → exits 0 in < 5 min (CPU) and:
   - writes exactly these files: `docs/screenshots/trajectories.png`, `docs/screenshots/violation_rates.png`, `docs/screenshots/training_curves.png`, plus `docs/screenshots/violations_unconstrained.csv`, `violations_soft.csv`, `violations_soft_projection.csv` (schema §5.5);
   - prints a results table with one row per variant containing `overall_violation_rate`, `obstacle_rate`, `velocity_rate`, `final_dist_to_goal`;
   - printed **`unconstrained` overall_violation_rate > 0.10** and **both shielded variants < 0.01** (tol = 1e-2 per §5.5);
   - both shielded variants reach the goal: `final_dist_to_goal < 0.3` m (safety must not come from refusing to move);
   - `soft_projection` `velocity_rate == 0.0` exactly (hard shield on a closed-form constraint).
4. Determinism: run the §8.3 command a second time → the printed results table is character-identical and all three CSVs are byte-identical (`cmp` the copies). PNG bytes are exempt (matplotlib metadata); their *content* must be reproduced.
5. `venv/bin/python examples/train_nav.py --variant soft --seed 1 --epochs 50` → exits 0 in < 60 s (short-run/CLI-override path works; no threshold requirements at 50 epochs).
6. `venv/bin/python -c "from pytorch_shield import BoxConstraint; import torch; print(BoxConstraint((-1,-1),(1,1)).violation(torch.tensor([[0.5,1.5],[-2.0,0.0],[0.0,0.0]])))"` → prints `tensor([0.5000, 1.0000, 0.0000])` (§5.2 ground truth holds in the shipped package).
7. Repo hygiene: `src/shield.py`, `src/train.py`, `src/model.py`, `src/__init__.py` no longer exist; `grep -ri "ODD" README.md src/ examples/` → no matches (UL 4600 / ISO 21448 may appear in README only, phrased as informative context, no alignment/conformance claims).

---

## 9. Demo Script (60 seconds)

> **[0–10 s]** "This is pytorch-shield — a small PyTorch library that enforces safety constraints while you train. Constraints are declared as objects with a differentiable violation measure; a Lagrangian layer auto-tunes the penalty weights; a projection layer hard-clips actions at inference; and a monitor writes the violation evidence to CSV."
>
> **[10–25 s]** Run `python examples/train_nav.py --variant all --seed 0`. "One command, CPU, about two minutes — I'll show the pre-computed outputs. A point-mass policy is trained by differentiable simulation to reach a goal past a keep-out disc, under a speed limit."
>
> **[25–40 s]** Open `docs/screenshots/trajectories.png`. "Unconstrained — red — drives straight through the obstacle at double the speed limit. The soft-shielded and projected variants — same task loss — go around it and respect the limit."
>
> **[40–52 s]** Open `violation_rates.png` + the printed table. "Measured, not asserted: unconstrained violates on over 10% of steps; both shielded variants are under 1%, and the projected variant has exactly zero speed violations. Every number traces back to a CSV the monitor wrote."
>
> **[52–60 s]** "The constraint math is unit-tested against hand-computed values, the whole run is seed-deterministic, and the same four classes drop into any model that emits actions. That's the point: assurance tooling you can read in an afternoon."

Screenshots to capture for the README (all produced by the demo itself): `trajectories.png` (hero image), `violation_rates.png`, `training_curves.png`.

---

## 10. Test Plan

Framework: plain `pytest`, no fixtures beyond `tmp_path`; every test seeds explicitly. Target ≥ 24 tests.

| File | Tests (assertion-level) |
|---|---|
| `test_constraints.py` (~8) | (1) Box §5.2 triple exactly (atol 1e-6); (2) Norm 3-4-5 case = 4.0 and feasible point = 0; (3) HalfPlane violated/boundary cases; (4) MinDistance inside/outside cases; (5) all violations ≥ 0 on `randn(256, 2)` for every class; (6) shape preservation `(B,)` and `(B, T)` inputs; (7) `violation.sum().backward()` produces finite, nonzero grads for an infeasible leaf input on each class; (8) `set_coefficients` with batched `(B,K,D)` matches a Python-loop reference. |
| `test_penalty.py` (~5) | (1) dual-ascent λ trace `[0.05, 0.10, 0.10, 0.12]` from §5.2; (2) λ never negative when fed zero/near-zero violations from a feasible policy; (3) `lambdas` is a buffer: absent from `parameters()`; (4) `QuadraticPenalty` value = `task + w·mean(v²)` on known numbers; (5) gradient from `LagrangianPenalty(task, viol)` reaches the upstream parameter that produced `viol`. |
| `test_projection.py` (~6) | (1) §5.2 closed-form numbers (box/norm/half-plane); (2) feasibility: for `randn(512, 2)·3`, `violation(project(x)) ≤ 1e-6` for box+norm and each half-plane system tested; (3) idempotence `project(project(x)) ≈ project(x)` atol 1e-6; (4) no-op on feasible input (exact equality); (5) two-half-plane wedge: converges within `max_iter=20` to a point satisfying both; (6) contradictory system (`x ≤ −1` and `x ≥ 1` as `−x ≤ −1`) with cap: returns finite output, no exception, residual violation > 0 (fallback path executed); plus `ValueError` on constructing with `MinDistanceConstraint`. |
| `test_monitor.py` (~4) | (1) CSV header exactly `step,constraint,mean_violation,max_violation,violation_rate,n`; (2) rate on `[0.0, 0.005, 0.5, 2.0]` with tol 1e-2 is 0.5, `n=4`, mean/max correct; (3) `summary()` aggregates two `log()` calls correctly; (4) writing twice yields byte-identical files. |
| `test_nav_e2e.py` (~3) | Config: `epochs=40, batch_size=32, eval_batch=64, horizon=40, seed=0`. (1) `unconstrained`: final task loss < initial task loss (it learns); (2) `soft_projection`: overall eval violation rate < `unconstrained`'s AND < 0.05, and `velocity_rate == 0.0`; (3) determinism: two `train_variant` calls with the same cfg give identical eval metric dicts. Entire file < 60 s. |

Not tested (consciously): plot pixel content (only that PNG files exist and are non-empty, covered by §8.3), long-run 300-epoch thresholds (covered by acceptance run, not CI-ish pytest).

---

## 11. Risks & Fallbacks

| Risk | Symptom | Mitigation baked into the spec | Fallback if it still occurs |
|---|---|---|---|
| Lagrangian nonconvergence / oscillating λ | violation rate plateaus > 1%, loss oscillates | Defaults chosen conservatively: Adam `lr=3e-3`, `dual_lr=0.1`, 300 epochs, `max_lambda=1e4`, violations mean-reduced (scale-stable) | Halve `lr` to `1.5e-3` and/or `dual_lr` to `0.05`; raise epochs to 600; warm-start `init_lambda=1.0`. Change **defaults in `NavConfig`**, don't hand-tune per run — determinism criterion §8.4 must keep holding. |
| Projection cycling on half-plane intersections | iterative loop hits `max_iter` routinely | Cyclic projection with `max_iter=20`, `tol=1e-6`; demo uses ≤ 1 active half-plane + norm per step, so 20 is generous | Fallback is specified behavior, not an error: one final box-clamp + norm-rescale, return, and let the monitor show the residual (§5.4). Raise `max_iter` to 50 only if e2e thresholds fail because of it. |
| Symmetric local minimum (policy stalls at the disc) | soft variants never reach goal, `final_dist_to_goal` ≥ 0.3 | Obstacle center offset to `(0.0, 0.15)` breaks the symmetry; start-position jitter ±0.25; effort weight tiny (1e-3) | Increase offset to 0.25; increase `start_jitter` to 0.4; if needed, seed-dependent tie-break noise `±0.05` added to `s0` y-coordinate (seeded ⇒ still deterministic). |
| Unconstrained variant violates on < 10% of steps | acceptance §8.3 threshold missed | Raw action scale 2.0 vs `v_max=1.0` ⇒ speeding for most of the transit; horizon 40 ≈ minimal-time path length, so transit dominates the horizon | Reduce `horizon` to 30 (less post-arrival hovering diluting the rate) — verify shielded variants still reach the goal (30 steps × 1 m/s = 3 m < 4 m path ⇒ if broken, keep 40 and lower `v_max` to 0.8 instead, rescaling the geometry check). |
| Shielded soft variant residual rate ∈ [1%, 5%] | soft passes "better than unconstrained" but misses < 1% | tol = 1e-2 absorbs converged Lagrangian residuals; dual ascent runs the full 300 epochs | Apply the λ warm-start fallback above; if only `soft` (not `soft_projection`) misses, that is a finding — do **not** silently relax §8.3; fix training (more epochs) until both pass. |
| CPython 3.14 venv incompatibility with a pin | pip resolve fails in `venv/` | Pins are ranges, not exact (§7); torch already installed and importable in this venv | Relax the upper bound of the offending pin; never downgrade the venv or create a second interpreter. |
| PNG nondeterminism trips people up | §8.4 confusion | Determinism defined over printed table + CSV bytes, PNGs explicitly exempt | `plots.py` may set `metadata={"Software": None}` in `savefig` to reduce diff noise; not required. |

---

## 12. Deferred to v2 (explicitly out of this one-shot)

1. **CBF layer** — a `ControlBarrierConstraint` enforcing `ḣ(x) ≥ −α·h(x)` through a differentiable QP (e.g. via `qpth`/custom KKT backward); the principled upgrade of the per-step half-plane linearization.
2. **RL integration** — wrap `LagrangianPenalty` as a reward/loss modifier for an actor-critic on `gymnasium` classic-control tasks; compare against the differentiable-sim results.
3. **Benchmark vs. Safety-Gym-style suites** — port the monitor to standard constrained-RL metrics (cost-rate curves) and publish a comparison table.
4. **HJ-reachability-informed constraints** — precomputed value-function keep-out sets as `Constraint` objects (cite as related work in v1 README).
5. Device support (MPS/CUDA), `torch.compile` pass, PyPI release, CI workflow.

---

*End of spec. Implement top-to-bottom in §6 order; §5 numbers are the test ground truth; §8 is the definition of done.*
