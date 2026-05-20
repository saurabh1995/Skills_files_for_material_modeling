---
name: elastoplasticity-integration
description: Generate return mapping algorithms for rate-independent elastoplastic constitutive
  models with isotropic and/or kinematic hardening. Use when users provide yield function,
  hardening law, and flow rule and need a JAX-compatible implicit integration scheme
  (return mapping) with Newton solver. Trigger for von Mises plasticity, Drucker-Prager,
  Mohr-Coulomb, or any custom yield surface with associative/non-associative flow rules,
  isotropic hardening, kinematic hardening, or mixed hardening.
---

# Implicit Elastoplastic Integration — Skill

---

## HOW TO USE THIS SKILL

**When given equations → always do both:**
1. Show the hand-calculation derivation (Steps A1–A6) so the user can verify the physics
2. Generate the complete working Python code following `von-mises-template.md`

**When given an existing model to extend:**
1. Identify which new hardening variables are added
2. Derive their implicit updates using the Step A4/A5 procedure
3. Add new variables to `internal_state_variables` and `operand`
4. Add their implicit updates inside `R_plastic`
5. Add their final state updates after Newton solve
6. Add write-back to `state` dict

**Reading order for reference files:**
- `von-mises-template.md` → mandatory first read before writing any code
- `general-template.md` → for non-von-Mises yield surfaces
- `hardening-laws.md` → hardening law reference
- `yield-surfaces.md` → yield surface reference

---

## PART A — DERIVATION WORKFLOW (Hand Calculation)

Follow these steps for ANY new model before writing code.
This converts the continuous constitutive equations into a scalar residual R(Δλ) = 0.

> **Rate-independent vs viscoplastic:** In rate-independent plasticity there is no
> overstress or time-dependent flow rule. The consistency condition f = 0 is enforced
> exactly at the end of every plastic step. Steps A1–A5 are mathematically identical to
> the viscoplastic derivation. Step A6 is the key difference — no phi, no time.

---

### STEP A1 — Write All Continuous Governing Equations

List every equation explicitly. Identify:
- The **additive strain split**: ε = εe + εp
- The **elastic law**: σ = C:(ε - εp)
- The **yield function**: f(σ, p, X) ≤ 0
- The **flow rule**: how ε̇p is defined (associative or non-associative)
- Every **hardening evolution equation**: one entry per internal variable
- The **consistency condition**: f = 0 during plastic flow (rate-independent, not viscoplastic)

**Example — von Mises with Voce isotropic hardening:**
```
Additive split:    ε = εe + εp
Elastic law:       σ = C:(ε - εp)
Yield function:    f = σ_eq - R(p)       where σ_eq = √(3/2 s:s)
                   R(p) = σ₀ + Q·(1 - exp(-b·p))
Flow rule:         ε̇p = λ̇ · N           N = (3/2)·s/σ_eq  (associative)
Hardening:         ṗ = λ̇               (equivalent plastic strain rate)
Consistency:       f = 0  during plastic loading
```

**Example — von Mises with Armstrong-Frederick kinematic hardening:**
```
Additive split:    ε = εe + εp
Elastic law:       σ = C:(ε - εp)
Yield function:    f = σ_eq(σ - X) - σ₀  where σ_eq(σ-X) = √(3/2 (s-X_dev):(s-X_dev))
Flow rule:         ε̇p = λ̇ · N           N = (3/2)·(s-X_dev)/σ_eq(σ-X)
Backstress ODE:    Ẋ = a·λ̇·N - c·X·λ̇   (Armstrong-Frederick)
Consistency:       f = 0  during plastic loading
```

---

### STEP A2 — Elastic Predictor and Yield Check

Freeze all plastic quantities at the previous step (backward Euler elastic predictor):
```
Trial stress:     σ_tr  = σ_n + C:Δε         (Δε = ε_t - ε_{t-1})
Trial backstress: X_tr  = X_{t-1}             (unchanged in predictor)
Trial p:          p_tr  = p_{t-1}             (unchanged in predictor)

Yield check:
  f_tr = σ_eq(σ_tr) - R(p_n)                 (no backstress)
  f_tr = σ_eq(σ_tr - X_n) - σ₀              (with backstress, no iso hardening)
  f_tr = σ_eq(σ_tr - X_n) - R(p_n) - σ₀    (with backstress + iso hardening)

if f_tr ≤ 0  → ELASTIC: accept trial state, done
if f_tr > 0  → PLASTIC: proceed to plastic corrector
```

> **Key difference from viscoplastic:** f_tr > 0 is inadmissible in rate-independent
> plasticity. The corrector must return the stress exactly to the yield surface.
> There is no "overstress" state that can persist — f_{n+1} = 0 exactly.

---

### STEP A3 — Express Stress in Terms of Δλ

Plastic strain increment from the (associative) flow rule, backward Euler:
```
Δεp = Δλ · N_tr     where Δλ = λ̇·Δt ≥ 0  (plastic multiplier increment)
```

Backward Euler stress update:
```
σ_{n+1} = σ_tr - C:(Δλ·N_tr)
```

**KEY SIMPLIFICATION when N_tr is purely deviatoric (tr(N_tr) = 0) — von Mises case:**
```
C = (κ - 2μ/3)(I⊗I) + 2μ·𝕀

C:N_tr = (κ - 2μ/3)·tr(N_tr)·I + 2μ·N_tr = 0 + 2μ·N_tr   (because tr(N_tr) = 0)

For von Mises flow:  Δεp = Δλ·(3/2)·n_tr  where n_tr = s_tr/σ_eq,tr

∴ C:(Δλ · (3/2) · n_tr) = 3μ·Δλ·n_tr

∴ σ_{n+1} = σ_tr - 3μ·Δλ·n_tr     ← correction involves μ only, not κ
```

This gives the fundamental scalar relationship for von Mises:
```
σ_eq,{n+1} = σ_eq,tr - 3μ·Δλ     ← equivalent stress decreases linearly with Δλ
```

> This simplification holds only when tr(N_tr) = 0 (associative von Mises, with or
> without backstress). For Drucker-Prager or non-associative flow (tr(N_tr) ≠ 0),
> use the full `C @ (deps - dlam * n_tr)` in the stress update.

In code, always write:
```python
sig_new = sig_old + C @ (deps - 1.5 * dlam * n_tr)     # von Mises (3/2 factor)
sig_new = sig_old + C @ (deps - dlam * n_tr)            # general / non-associative
```

---

### STEP A4 — Implicit Update for Each Scalar Hardening Variable

For each scalar internal variable q with evolution equation `q̇ = f(q)·λ̇`:

Apply backward Euler: `q_{n+1} - q_n = f(q_{n+1})·Δλ`, then solve algebraically.

> **Note:** This derivation is mathematically identical to the viscoplastic case.
> The driving variable Δλ is the same symbol. The only difference is how Δλ is found —
> by the consistency condition f = 0 (here) rather than a time-dependent flow rule.
> The rational form derivation below is unchanged.

**Case: equivalent plastic strain `ṗ = λ̇` (linear ODE, trivial)**
```
p_{n+1} - p_n = Δλ
∴ p_{n+1} = p_n + Δλ    ← no algebra needed
```

**Case: linear isotropic hardening `Ṙ = H·λ̇`**
```
R_{n+1} = R_n + H·Δλ    ← direct, no nonlinearity
```

**Case: Voce saturation hardening `Ṙ = b·(Q - R)·λ̇`**
```
R_{n+1} - R_n = b·(Q - R_{n+1})·Δλ
R_{n+1} + b·R_{n+1}·Δλ = R_n + b·Q·Δλ
R_{n+1}·(1 + b·Δλ) = R_n + b·Q·Δλ

∴ R_{n+1} = (R_n + b·Q·Δλ) / (1 + b·Δλ)    ← rational fraction, always stable
```
As Δλ→∞: R_{n+1}→Q (saturates). As Δλ→0: R_{n+1}→R_n. ✓

**General rational form — for any ODE `q̇ = (A - B·q)·λ̇`:**
```
q_{n+1} = (q_n + A·Δλ) / (1 + B·Δλ)
```
For Voce: A = b·Q, B = b. For linear (B = 0): q_{n+1} = q_n + A·Δλ.

> **Why backward Euler?** Forward Euler (`q_{n+1} = q_n + f(q_n)·Δλ`) is explicit and
> goes unstable for large Δλ. The backward Euler rational fraction is unconditionally
> stable: q stays bounded for any step size. Always use it for nonlinear hardening.

---

### STEP A5 — Implicit Update for Each Tensor Hardening Variable

For each 6-component Voigt tensor Q with ODE `Q̇ = A·λ̇·N - B·Q·λ̇`:

Apply backward Euler with N fixed at the trial state (semi-implicit):
```
Q_{n+1} - Q_n = A·Δλ·N_tr - B·Q_{n+1}·Δλ
Q_{n+1}·(1 + B·Δλ) = Q_n + A·Δλ·N_tr

∴ Q_{n+1} = (Q_n + A·Δλ·N_tr) / (1 + B·Δλ)
```

**Armstrong-Frederick backstress `Ẋ = a·λ̇·N - c·X·λ̇`:**
```
X_{n+1} = (X_n + a·Δλ·N_tr) / (1 + c·Δλ)
```
As Δλ→∞: X_{n+1}→(a/c)·N_tr (saturates). ✓

> **Semi-implicit** means N_tr = n_tr is fixed at the trial state throughout the Newton
> solve. This keeps the residual scalar in Δλ. If N were updated inside the Newton loop,
> the problem would become a 6-component tensor equation requiring a full 6×6 solve.

**Multiple backstresses (Chaboche multi-surface):**
```
X1_{n+1} = (X1_n + a1·Δλ·N_tr) / (1 + c1·Δλ)
X2_{n+1} = (X2_n + a2·Δλ·N_tr) / (1 + c2·Δλ)
X_total = X1_{n+1} + X2_{n+1}    ← use X_total in yield function
```

---

### STEP A6 — Form the Scalar Residual R(Δλ) = 0

**Rate-independent elastoplasticity:** enforce the consistency condition f_{n+1} = 0.

Substitute σ_{n+1}(Δλ) from A3 and all internal variable updates from A4/A5 into
the yield function and set it to zero:

**Von Mises, isotropic hardening only:**
```
f_{n+1} = σ_eq,{n+1} - R(p_{n+1}) = 0

Substituting A3:  σ_eq,{n+1} = σ_eq,tr - 3μ·Δλ
Substituting A4:  p_{n+1}    = p_n + Δλ  →  R(p_{n+1}) = yield_stress(p_n + Δλ)

∴ R(Δλ) = (σ_eq,tr - 3μ·Δλ) - yield_stress(p_n + Δλ) = 0

Code:  return sigma_eq_new - sigma_y_new
```

**Von Mises, mixed iso + kinematic hardening:**
```
f_{n+1} = σ_eq(σ_{n+1} - X_{n+1}) - σ₀ - R_{n+1} = 0

Substituting A3:  σ_{n+1} = σ_tr - 3μ·Δλ·n_tr
Substituting A4:  R_{n+1} = (R_n + b·Q·Δλ) / (1 + b·Δλ)
Substituting A5:  X_{n+1} = (X_n + a·Δλ·n_tr) / (1 + c·Δλ)

∴ R(Δλ) = σ_eq(σ_{n+1} - X_{n+1}) - σ₀ - R_{n+1} = 0
```

> **Critical difference from viscoplastic:** There is NO phi (overstress function) and
> NO time step Δt in the residual. The residual IS the yield function set to zero.
> In viscoplasticity: R(Δλ) = Δλ - η·Δt·φ(f) or J₂ - k - R - K·(Δλ/Δt)^(1/n).
> In rate-independent plasticity: R(Δλ) = f_{n+1}(Δλ) = 0. Nothing else.

**Linear hardening — Δλ has a closed form (Newton converges in 1 step):**
```
σ_eq,tr - 3μ·Δλ - σ₀ - H·(p_n + Δλ) = 0
σ_eq,tr - σ₀ - H·p_n = (3μ + H)·Δλ

∴ Δλ = (σ_eq,tr - σ₀ - H·p_n) / (3μ + H)    ← exact in 1 Newton step
```

---

### STEP A7 — Summary Table

| Step | Task | Output for von Mises + Voce |
|------|------|-----------------------------|
| A1 | List all governing equations | ε=εe+εp, σ=C:εe, f=σ_eq-R(p), ṗ=λ̇ |
| A2 | Elastic predictor + yield check | σ_tr, f_tr = σ_eq,tr - R(p_n) |
| A3 | Stress in terms of Δλ | σ_{n+1} = σ_tr - 3μΔλ n_tr; σ_eq,{n+1} = σ_eq,tr - 3μΔλ |
| A4 | Implicit scalar hardening update | p_{n+1} = p_n + Δλ; R_{n+1} = (R_n + bQΔλ)/(1+bΔλ) |
| A5 | Implicit tensor hardening update | X_{n+1} = (X_n + aΔλN)/(1+cΔλ)  [if kinematic present] |
| A6 | Consistency condition → R(Δλ)=0 | σ_eq,{n+1} - yield_stress(p_{n+1}) = 0 |
| A7 | Newton solve → update all state | Converged Δλ, then σ, εp, εe, p [, X, R] |

---

## PART B — CODE GENERATION WORKFLOW

After completing the derivation (Part A), generate code as follows.

### B1 — Map equations to code structure

| Equation component | Code location |
|---|---|
| Material parameters | `@dataclass MyModel_Params` |
| Hardening law R(p) | `yield_stress(self, alpha)` method |
| Shear modulus μ | `_shear_modulus(self)` method |
| Internal state variables | `internal_state_variables` dict |
| Flow direction N_tr (trial) | `n_tr` computed before `operand` |
| Elastic predictor | `sig_tr = sig_old + C @ deps` |
| Yield check | `yield_criterion = sigma_eq_tr - sigma_y_old` |
| Step A3 (stress in residual) | first line of `R_plastic(dlam)` |
| Steps A4/A5 (hardening in residual) | subsequent lines of `R_plastic(dlam)` |
| Step A6 (residual = yield function = 0) | return line of `R_plastic(dlam)` |
| Final state update | after `newton.solve(0.0)`, recompute with converged dlam |

### B2 — Determine which template to start from

| Model characteristics | Start from template |
|---|---|
| Von Mises, any isotropic hardening | `von-mises-template.md` |
| Von Mises + kinematic (Armstrong-Frederick) | `von-mises-template.md` + add X to state |
| Von Mises + mixed iso + kinematic | `von-mises-template.md` + add X and R to state |
| Two backstresses (Chaboche) | `von-mises-template.md` + add X1, X2 to state |
| Non-von-Mises yield surface | `general-template.md` |

### B3 — Adding a new internal variable (general recipe)

For every new variable `q` (scalar) or `Q` (tensor):

```python
# 1. Declare in internal_state_variables:
"q":  1,   # scalar
"Q":  6,   # Voigt tensor

# 2. Unpack in constitutive_update:
q_old = state["q"][0]        # scalar — always [0] to get Python/JAX scalar
Q_old = state["Q"]           # tensor — no [0], already shape (6,)

# 3. Add to operand tuple (both branches must receive it):
operand = (..., q_old, Q_old, ...)

# 4. Pass through unchanged in elastic_update (and add to return tuple):
q_new = q_old
Q_new = Q_old

# 5. Add implicit update inside R_plastic(dlam) — from Steps A4/A5:
q_new = (q_old + A * dlam) / (1.0 + B * dlam)            # scalar, Step A4
Q_new = (Q_old + A * dlam * n_tr) / (1.0 + B * dlam)     # tensor, Step A5

# 6. Recompute after newton.solve with converged dlam (same formulas, fresh):
q_new = (q_old + A * dlam) / (1.0 + B * dlam)
Q_new = (Q_old + A * dlam * n_tr) / (1.0 + B * dlam)

# 7. Write back to state dict:
state["q"] = jnp.array([q_new])    # scalar → always wrap in array
state["Q"] = Q_new                  # tensor → already array, no wrapping
```

Both `elastic_update` and `plastic_update` must return a tuple with **identical element
count and identical shapes**. Count every return value after adding a variable.

---

## PART C — CRITICAL RULES (violations cause wrong results or JAX errors)

### C1 — Elastic stiffness C
```python
# CORRECT:
C = self.elastic_model.C

# WRONG — never compute manually:
lam = nu*E/((1+nu)*(1-2*nu));  mu = E/(2*(1+nu))
C = jnp.array([[lam+2*mu, lam, ...]])
```

### C2 — Shear modulus μ
```python
# CORRECT — always use the class method:
mu = self._shear_modulus()

# WRONG — computing inline (creates scope issues in R_plastic closure):
mu = self.params.E / (2.0 * (1.0 + self.params.nu))  # in constitutive_update body
```
The method `_shear_modulus` is defined using `self.params`, so it is available everywhere
via `self` and does not create closure problems.

### C3 — Newton placement: inside plastic_update only
```python
# CORRECT — Newton is inside plastic_update only:
def plastic_update(operand):
    def R_plastic(dlam): ...
    newton = JAXNewton()
    newton.set_residual(R_plastic)
    dlam, _ = newton.solve(0.0)
    ...

# WRONG — Newton outside lax.cond (runs unconditionally, elastic step gets wrong dlam):
newton = JAXNewton()
newton.set_residual(residual_with_lax_cond_inside)
dlam, _ = newton.solve(0.0)
sig = sig_tr - 2*mu * deps_p(dlam, yield_criterion)
```

### C4 — No lax.cond inside R_plastic
```python
# CORRECT — R_plastic is only ever called in the plastic branch; no conditional needed:
def R_plastic(dlam):
    sig_new      = sig_tr - 3.0 * mu * dlam * n_tr
    sigma_eq_new = self._equivalent_stress(sig_new)
    sigma_y_new  = self.yield_stress(p_old + dlam)
    return sigma_eq_new - sigma_y_new

# WRONG — nested lax.cond inside residual:
def residual(dlam):
    return jax.lax.cond(yield_criterion < 0.0, r_elastic, r_plastic, dlam)
```

### C5 — jax.lax.cond branch shape matching
Both `elastic_update` and `plastic_update` must return identical element count and shapes:
```python
# elastic: return sig_new, p_new, eps_e_new, eps_p_new, dlam, yield_criterion
# plastic: return sig_new, p_new, eps_e_new, eps_p_new, dlam, yield_criterion
# dlam in elastic branch: Python float 0.0 — NOT jnp.array([0.0])
# jnp.array([...]) wrapping only at state write-back, after lax.cond
```

### C6 — Scalar state variables
```python
# Unpack (always [0]):
p_old = state["p"][0]
R_old = state["R"][0]

# Write back (always wrap):
state["p"] = jnp.array([p_new])
state["R"] = jnp.array([R_new])
```

### C7 — Never reuse values from inside R_plastic after Newton
```python
# WRONG — closure variable is stale (last Newton iteration, not converged):
dlam, _ = newton.solve(0.0)
sig_new = sig_computed_inside_R_plastic

# CORRECT — recompute fresh with converged dlam:
dlam, _ = newton.solve(0.0)
sig_new = sig_old + C @ (deps - 1.5 * dlam * n_tr)
```

### C8 — Flow direction n_tr fixed at trial state (semi-implicit)
```python
# CORRECT — n_tr computed once before R_plastic, captured by closure:
n_tr = s_tr / jnp.clip(sigma_eq_tr, a_min=1e-8)
def R_plastic(dlam):
    sig_new = sig_tr - 3.0 * mu * dlam * n_tr    # n_tr fixed throughout

# WRONG — n_tr recomputed inside Newton loop (turns scalar problem into tensor problem):
def R_plastic(dlam):
    sig_new = ...
    n_tr = self._deviatoric(sig_new) / sigma_eq_new    # wrong
```

### C9 — Newton initial guess
```python
newton.solve(0.0)    # always start from zero — never from a previous dlam value
```

### C10 — lax.cond condition and branch order
```python
# CORRECT:
is_plastic = yield_criterion >= 0.0
jax.lax.cond(is_plastic, plastic_update, elastic_update, operand)

# WRONG — swapped branches:
jax.lax.cond(is_plastic, elastic_update, plastic_update, operand)

# WRONG — strict > misses the exact yield point:
is_plastic = yield_criterion > 0.0
```

---

## PART D — COMPLETE IMPLICIT UPDATE FORMULAS

Quick lookup — copy these into `R_plastic` and the final state update block after Newton.
For rate-independent elastoplasticity there is NO phi and NO Δt in the residual.

### Stress
```python
sig_new = sig_tr - 3.0 * mu * dlam * n_tr              # von Mises, inside R_plastic
sig_new = sig_old + C @ (deps - 1.5 * dlam * n_tr)     # equivalent; use for final recompute
sig_new = sig_old + C @ (deps - dlam * n_tr)            # Drucker-Prager / non-associative
```

### Equivalent plastic strain (trivial, linear)
```python
p_new = p_old + dlam
```

### Plastic strain tensor
```python
eps_p_new = eps_p_old + 1.5 * dlam * n_tr    # von Mises (3/2 factor)
eps_p_new = eps_p_old + dlam * n_tr           # general / Drucker-Prager
```

### Elastic strain tensor
```python
eps_e_new = eps - eps_p_new    # additive split: εe = ε - εp
```

### Linear isotropic hardening
```python
R_new = R_old + H * dlam
```

### Voce saturation hardening (nonlinear — backward Euler rational form)
```python
R_new = (R_old + b * Q * dlam) / (1.0 + b * dlam)
# b = saturation rate, Q = saturation stress increment
```

### Power law hardening R(p) = σ₀ + K·pⁿ
```python
# No separate ODE needed — R is evaluated directly as a function of p:
p_new = p_old + dlam
# yield_stress(p_new) handles R automatically via the method
```

### Armstrong-Frederick backstress (single)
```python
X_new = (X_old + a * dlam * n_tr) / (1.0 + c * dlam)
# a = kinematic modulus, c = dynamic recovery rate
```

### Two Armstrong-Frederick backstresses (Chaboche multi-surface)
```python
X1_new = (X1_old + a1 * dlam * n_tr) / (1.0 + c1 * dlam)
X2_new = (X2_old + a2 * dlam * n_tr) / (1.0 + c2 * dlam)
X_total = X1_new + X2_new    # use X_total in yield function and residual
```

### Residuals (Step A6) — always the yield function set to zero, never phi

```python
# Von Mises, isotropic hardening only (pure iso, n_tr from trial stress):
def R_plastic(dlam):
    sig_new      = sig_tr - 3.0 * mu * dlam * n_tr         # Step A3
    sigma_eq_new = self._equivalent_stress(sig_new)
    sigma_y_new  = self.yield_stress(p_old + dlam)          # Step A4
    return sigma_eq_new - sigma_y_new                        # Step A6: f = 0

# Von Mises, Voce iso hardening (R as separate state variable):
def R_plastic(dlam):
    sig_new = sig_tr - 3.0 * mu * dlam * n_tr               # Step A3
    R_new   = (R_old + b * Q * dlam) / (1.0 + b * dlam)    # Step A4: Voce
    sigma_eq_new = self._equivalent_stress(sig_new)
    return sigma_eq_new - (sigma_0 + R_new)                  # Step A6: f = 0

# Von Mises, mixed iso + Armstrong-Frederick kinematic:
def R_plastic(dlam):
    sig_new = sig_tr - 3.0 * mu * dlam * n_tr               # Step A3
    R_new   = (R_old + b * Q * dlam) / (1.0 + b * dlam)    # Step A4
    X_new   = (X_old + a * dlam * n_tr) / (1.0 + c * dlam) # Step A5
    xi_new  = self._deviatoric(sig_new - X_new)
    sigma_eq_new = self._equivalent_stress(xi_new)
    return sigma_eq_new - sigma_0 - R_new                    # Step A6: f = 0
```

---

## PART E — MODEL IDENTIFICATION QUICK GUIDE

When the user gives equations, identify model type in 30 seconds:

| What you see in the equations | Model type | Template |
|---|---|---|
| `f = σ_eq - σ₀`, no hardening ODE | Perfect plasticity | `von-mises-template.md`, constant `yield_stress` |
| `f = σ_eq - (σ₀ + H·p)` | Von Mises + linear iso | `von-mises-template.md` |
| `f = σ_eq - (σ₀ + Q(1-exp(-bp)))` | Von Mises + Voce iso | `von-mises-template.md` |
| `f = σ_eq - (σ₀ + K·pⁿ)` | Von Mises + power law | `von-mises-template.md` |
| `f = σ_eq(σ-X) - σ₀` + `Ẋ = a·λ̇·N - c·X·λ̇` | Von Mises + kinematic (A-F) | `von-mises-template.md` + add X |
| Same + iso hardening ODE for R | Von Mises + mixed iso+kinematic | `von-mises-template.md` + add X, R |
| Two backstress ODEs X1, X2 | Chaboche multi-surface | `von-mises-template.md` + add X1, X2 |
| `f = √J₂ + α·I₁ - k` (pressure-dependent) | Drucker-Prager | `general-template.md` |
| Any custom f(σ) | General surface | `general-template.md` |

> **Rate-independent vs viscoplastic check:** If the equations contain a time-dependent
> term in the flow rule (`η`, `Δt`, `(f/K)^n`, `exp(f/...)`) → it is viscoplastic; use
> the viscoplastic SKILL.md. Rate-independent models have NO time in the flow rule and
> enforce f = 0 exactly at every step.

---

## PART F — PITFALLS AND FIXES

| Symptom | Root cause | Fix |
|---------|-----------|-----|
| Newton doesn't converge | Residual has wrong sign or wrong form | Return `sigma_eq_new - sigma_y_new`; check A6 derivation |
| Newton converges in 0 steps (dlam = 0 always) | Newton runs outside `lax.cond` | Move Newton fully inside `plastic_update` |
| `dlam` always 0 even inside plastic branch | Yield check uses `>` not `>=` | Use `yield_criterion >= 0.0` |
| Stress drifts off yield surface over steps | Explicit Euler used for nonlinear ODE | Use backward Euler rational form `(old + A*dlam)/(1 + B*dlam)` |
| R or X grows without bound | Explicit ODE update used | Switch to rational form |
| `jax.lax.cond` shape error | Return tuple count or shapes differ between branches | Count every element in both branch return tuples |
| `state["p"]` wrong value | Missing `[0]` on scalar unpack | `p_old = state["p"][0]` |
| `eps_p` unchanged after plastic step | Not updated in `plastic_update` | `eps_p_new = eps_p_old + 1.5 * dlam * n_tr` |
| Stress wrong but `dlam` correct | Stale value reused from inside `R_plastic` | Recompute `sig_new` fresh after `newton.solve` |
| μ not available in residual | μ not stored as method | Use `mu = self._shear_modulus()` at top of `constitutive_update` |
| C wrong | Manual Lamé construction | `C = self.elastic_model.C` only |
| NaN gradient at zero stress | Raw `jnp.sqrt` at zero | `jnp.clip(sigma_eq_tr, a_min=1e-8)` before division in `n_tr` |
| Voigt shear factor wrong | Missing factor-of-2 on shear terms | `s_dot_s = s0²+s1²+s2² + 2*(s3²+s4²+s5²)` |
| Wrong branch executed | `lax.cond` branches swapped | TRUE branch = `plastic_update`, FALSE = `elastic_update` |
| Voce update unstable for large Δλ | Using `R_old + b*Q*dlam` (explicit) | Use `(R_old + b*Q*dlam)/(1 + b*dlam)` (implicit) |
| Residual contains Δt or phi | Mixed up with viscoplastic derivation | Rate-independent residual is ONLY f_{n+1} = 0; no Δt, no phi |

---

## PART G — REFERENCE FILE MAP

| File | Purpose | When to read |
|---|---|---|
| `von-mises-template.md` | Complete annotated template + critical rules + pitfalls | Before writing any code |
| `general-template.md` | Template for non-von-Mises surfaces; note on C@n_tr vs 3μ shortcut | When yield surface is not von Mises |
| `hardening-laws.md` | R(p) function forms: linear, Voce, power law, Swift, combined | When choosing or implementing hardening |
| `yield-surfaces.md` | Yield surface forms: von Mises, Tresca, Drucker-Prager, Mohr-Coulomb | When yield surface is not von Mises |

---

## PART H — NOTATION REFERENCE

| Symbol | Meaning | Code variable |
|--------|---------|---------------|
| Δλ | Plastic multiplier increment (= λ̇·Δt, solved via consistency f=0) | `dlam` |
| N_tr, n_tr | Flow direction at trial state, (3/2)·s_tr/σ_eq,tr | `n_tr` |
| σ_tr | Trial stress | `sig_tr` |
| σ_eq,tr | Von Mises equivalent stress of trial stress | `sigma_eq_tr` |
| μ | Shear modulus E/(2(1+ν)) | `self._shear_modulus()` |
| f_tr | Trial yield function value | `yield_criterion` |
| p | Equivalent plastic strain (scalar) | `p`, unpack with `[0]` |
| R | Isotropic hardening variable (scalar) | `R`, unpack with `[0]` |
| X | Backstress tensor (Voigt 6-vector) | `X`, no `[0]` needed |
| σ₀ | Initial yield stress | `params.sigma_0` |
| Q | Voce saturation stress increment | `params.Q` |
| b | Voce saturation rate | `params.b` |
| H | Linear hardening modulus | `params.H` |
| a | A-F kinematic modulus | `params.a` |
| c | A-F dynamic recovery rate | `params.c` |
| ()' | Deviatoric part | `_deviatoric(sig)` |
| σ_eq(·) | Von Mises equivalent stress | `_equivalent_stress(sig)` |
| C | Elastic stiffness tensor (6×6 Voigt) | `self.elastic_model.C` |
| ε, εe, εp | Total, elastic, plastic strain (Voigt 6-vector) | `eps`, `eps_e`, `eps_p` |
| f_{n+1} = 0 | Consistency condition (rate-independent) | return line of `R_plastic` |