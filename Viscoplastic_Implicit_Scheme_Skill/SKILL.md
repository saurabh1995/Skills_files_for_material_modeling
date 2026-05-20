---
name: implicit-integrator
description: >
  Use this skill for ANY of the following:
  (1) Generating fully implicit integration scheme code for a viscoplastic constitutive model
      given differential equations (flow rule, hardening ODEs, yield/overstress function).
  (2) Deriving the implicit scheme by hand — converting continuous rate equations into a
      scalar residual R(Δλ) = 0 in terms of the plastic multiplier.
  (3) Extending or modifying an existing model (adding damage, extra backstress, new hardening).
  Covers: Perzyna, Duvaut-Lions, Consistency, Lemaitre-Chaboche, and any user-defined
  viscoplastic model. Output is always FEniCSx-JAX compatible (dolfinx_materials framework).
---

# Implicit Viscoplastic Integration — Skill

---

## HOW TO USE THIS SKILL

**When given equations → always do both:**
1. Show the hand-calculation derivation (Steps A1–A6) so the user can verify the physics
2. Generate the complete working Python code following `code-template.md`

**When given an existing model to extend:**
1. Identify which new ODEs are added
2. Derive their implicit updates (Step A4/A5 procedure)
3. Add new variables to `internal_state_variables` and `operand`
4. Add their implicit updates inside `R_plastic`
5. Add their final state updates after Newton solve
6. Add write-back to `state` dict

**Reading order for reference files:**
- `code-template.md` → mandatory first read before writing any code
- `quick-reference.md` → model identification, residual snippets, debugging
- `box-algorithms.md` → mathematical algorithm boxes for each model type

---

## PART A — DERIVATION WORKFLOW (Hand Calculation)

Follow these steps for ANY new model. This is the procedure from
`LC_implicit_scheme_hand_calculation.pdf` — generalised to work for any equations.

---

### STEP A1 — Write All Continuous Governing Equations

List every rate equation explicitly. Identify:
- The **flow rule**: how Ėₚ (or Ēdot_p) is defined
- The **overstress / yield function**: what drives plastic flow
- Every **hardening ODE**: one entry per internal variable
- The **stress equation**: Hooke's law S = C:(E - Ep)

**Example — Lemaitre-Chaboche:**
```
Additive split:    Ė = Ėe + Ėp
Flow rule:         Ėp = (3/2) · Ēdot_p · N          where N = dev(S-X)/J₂(S-X)
Overstress law:    Ēdot_p = ⟨(J₂(S-X) - R - k) / K⟩ⁿ
Backstress ODE:    Ẋ = a·Ēdot_p·N - s·X·Ēdot_p       (Armstrong-Frederick)
Iso. hardening:    Ṙ = b₁·(b₂ - R)·Ēdot_p             (Voce saturation)
Hooke's law:       S = C:(E - Ep)
```

**Example — Perzyna pure plasticity:**
```
Flow rule:         Ėp = η·φ(f)·∂f/∂σ
Overstress:        φ = (f/σ_y)^N  for f > 0,  else 0
Yield function:    f = J₂(σ) - σ_y
No hardening ODEs.
```

---

### STEP A2 — Elastic Predictor and Yield Check

Always freeze plastic strain and backstress for the predictor step:
```
Trial stress:   Ŝ = S_n + C:ΔE         (ΔE = E_t - E_{t-1})
Trial backstress: X̂ = X_{t-1}           (unchanged in predictor)

Yield check:    f_y = J₂(Ŝ - X̂) - R_{t-1} - k    (LC form)
             or f_y = J₂(Ŝ) - σ_y                  (Perzyna, no backstress)

if f_y ≤ 0  → ELASTIC: accept trial state
if f_y > 0  → PLASTIC: proceed to corrector
```

---

### STEP A3 — Express Stress in Terms of Δλ

Plastic strain increment from flow rule:
```
ΔEp = Ep_t - Ep_{t-1} = (3/2)·Δλ·N        (for von Mises / LC)
                       = Δλ·∂f/∂σ          (for Perzyna, no 3/2 factor)
```

Stress correction (subtract plastic contribution from trial stress):
```
S_t = Ŝ - C:ΔEp
```

**KEY SIMPLIFICATION when N is deviatoric (tr(N) = 0):**
```
C = (κ - 2μ/3)(I⊗I) + 2μ·𝕀

C:N = (κ - 2μ/3)·tr(N)·I + 2μ·N = 0 + 2μ·N = 2μN

∴ C:((3/2)·Δλ·N) = 3μ·Δλ·N

∴ S_t = Ŝ - 3μ·Δλ·N     ← stress correction only involves shear modulus μ
```

> This simplification holds for any associative von Mises model where N = dev(.)/J₂(.).
> For Drucker-Prager or non-associative flow (tr(N) ≠ 0), use `C @ (deps - dlam*n)` directly.
>
> In code we always write `sig_old + C @ (deps - 1.5*dlam*N)` which is general and equivalent.

---

### STEP A4 — Implicit Update for Each Scalar Hardening ODE

For each scalar internal variable q with ODE `q̇ = f(q)·Ēdot_p`:

Apply backward Euler: `q_t - q_{t-1} = f(q_t)·Δλ`, then solve algebraically for `q_t`.

**Case: Voce isotropic hardening `Ṙ = b₁(b₂ - R)·Ēdot_p`**
```
R_t - R_{t-1} = b₁(b₂ - R_t)·Δλ
R_t + b₁·R_t·Δλ = R_{t-1} + b₁·b₂·Δλ
R_t·(1 + b₁·Δλ) = R_{t-1} + b₁·b₂·Δλ

∴ R_t = (R_{t-1} + b₁·b₂·Δλ) / (1 + b₁·Δλ)    ← rational fraction, always stable
```
As Δλ→∞: R_t→b₂ (saturates). As Δλ→0: R_t→R_{t-1} (no change). ✓

**Case: Linear isotropic hardening `ṗ = Ēdot_p` (equivalent plastic strain)**
```
p_t = p_{t-1} + Δλ    ← trivial, no ODE needed
```

**Case: Power-law hardening `Ṙ = H·Ēdot_p`** (linear hardening in disguise)
```
R_t = R_{t-1} + H·Δλ    ← explicit update sufficient (linear ODE)
```

**General rational form recipe** — for any ODE `q̇ = (A - B·q)·Ēdot_p`:
```
q_t = (q_{t-1} + A·Δλ) / (1 + B·Δλ)
```
Where A = b₁·b₂ and B = b₁ for Voce. If B=0 (no decay): q_t = q_{t-1} + A·Δλ.

---

### STEP A5 — Implicit Update for Each Tensor Hardening ODE

For each 6-component Voigt tensor variable Q with ODE `Q̇ = A·Ēdot_p·N - B·Q·Ēdot_p`:

Apply backward Euler with N fixed at previous step (semi-implicit):
```
Q_t - Q_{t-1} = A·Δλ·N - B·Q_t·Δλ
Q_t·(1 + B·Δλ) = Q_{t-1} + A·Δλ·N

∴ Q_t = (Q_{t-1} + A·Δλ·N) / (1 + B·Δλ)
```

**Armstrong-Frederick backstress `Ẋ = a·Ēdot_p·N - s·X·Ēdot_p`:**
```
X_t = (X_{t-1} + a·Δλ·N) / (1 + s·Δλ)
```

> Semi-implicit means N = dev(S_{t-1} - X_{t-1})/J₂(S_{t-1} - X_{t-1}) is fixed throughout
> the Newton solve. This keeps the problem SCALAR in Δλ. If N were updated at each Newton
> iteration, the system becomes a tensor equation requiring a full 6×6 Newton solve.

**Multiple backstresses** (e.g., Chaboche 2-surface):
```
X1_t = (X1_{t-1} + a1·Δλ·N) / (1 + c1·Δλ)
X2_t = (X2_{t-1} + a2·Δλ·N) / (1 + c2·Δλ)
X_total = X1_t + X2_t    ← use X_total in yield function
```

---

### STEP A6 — Form the Scalar Residual R(Δλ) = 0

Substitute all expressions from A3, A4, A5 into the flow rule.
Rearrange so that one side is zero. This is the Newton residual.

**For Lemaitre-Chaboche:**
```
Flow rule (plastic domain, drop ⟨⟩):
  Ēdot_p = ((J₂(S-X) - R - k) / K)ⁿ

Multiply by Δt:
  Δλ = Δt·((J₂(S_t - X_t) - R_t - k) / K)ⁿ

Take (1/n)-th power of both sides:
  K·(Δλ/Δt)^(1/n) = J₂(S_t - X_t) - R_t - k

∴ R(Δλ) = J₂(S_t - X_t) - k - R_t - K·(Δλ/Δt)^(1/n) = 0

where S_t, R_t, X_t are all functions of Δλ from Steps A3–A5.
```

**For Perzyna:**
```
Flow rule:  Ėp = η·(f/σ_y)^N·n    →   Δλ = η·Δt·(f(S_t)/σ_y)^N

∴ R(Δλ) = Δλ - η·Δt·(f(S_t)/σ_y)^N = 0

where S_t = S_n + C:(Δε - Δλ·n)  from Step A3.
```

**For Consistency model:**
```
Enforce: f(S_t, κ_t, κ̇_t) = 0

κ_t = κ_{t-1} + Δλ        (Step A4)
κ̇_t = Δλ/Δt              (backward Euler rate)

∴ R(Δλ) = J₂(S_t) - σ_y - H·κ_t - m·(Δλ/Δt) = 0
```

---

### STEP A7 — Summary Table

| Step | Task | Output for LC model |
|------|------|---------------------|
| A1 | List all rate equations | Ėp, Ēdot_p, Ẋ, Ṙ |
| A2 | Elastic predictor + yield check | Ŝ, fᵧ = J₂(Ŝ-X_n) - R_n - k |
| A3 | Stress update in terms of Δλ | S_t = Ŝ - 3μΔλN (or via full C) |
| A4 | Implicit scalar hardening ODEs | R_t = (R_n + b₁b₂Δλ)/(1+b₁Δλ) |
| A5 | Implicit tensor hardening ODEs | X_t = (X_n + aΔλN)/(1+sΔλ) |
| A6 | Substitute into flow rule → R(Δλ)=0 | J₂(S_t-X_t) - k - R_t - K(Δλ/Δt)^(1/n) |
| A7 | Newton solve → update all state | Converged Δλ, then S, Ep, Ee, X, R, p |

---

## PART B — CODE GENERATION WORKFLOW

After completing derivation (Part A), generate code as follows.

### B1 — Map equations to code structure

| Equation component | Code location |
|---|---|
| Material parameters | `@dataclass MyModel_Params(JAXMaterial)` |
| Internal state variables | `internal_state_variables` dict |
| Flow direction N | `_flow_direction(sig, X)` or `df_dsigma(sig)` |
| Elastic predictor | `sig_trial = sig_old + C @ deps` |
| Yield check | `fy = J₂(...) - k - R_old` |
| Step A3 (stress in R) | first line inside `R_plastic(dlam)` |
| Steps A4/A5 (hardening in R) | subsequent lines inside `R_plastic(dlam)` |
| Step A6 (residual) | return line of `R_plastic(dlam)` |
| Final state update | after `newton.solve(0.0)`, recompute with converged dlam |

### B2 — Determine which template to start from

Read `code-template.md` first. Select starting template:

| Model characteristics | Start from template |
|---|---|
| No hardening, constant yield | Template 1 (Perzyna pure) |
| Isotropic hardening only (R or p) | Template 2 (Perzyna + iso hardening) |
| Backstress X + isotropic R | Template 3 (Lemaitre-Chaboche) |
| Any new model | Template 3 as base, modify per equations |

### B3 — Adding a new internal variable (general recipe)

For every new variable `q` (scalar) or `Q` (tensor):

```python
# 1. Declare in internal_state_variables:
"q":  1,   # scalar
"Q":  6,   # Voigt tensor

# 2. Unpack in constitutive_update:
q_old = state["q"][0]        # scalar
Q_old = state["Q"]           # tensor (already shape (6,))

# 3. Add to operand tuple (both branches must receive it)
operand = (..., q_old, Q_old, ...)

# 4. Pass through unchanged in elastic_update:
q_new = q_old
Q_new = Q_old

# 5. Add implicit update inside R_plastic(dlam):
q_new = (q_old + A*dlam) / (1 + B*dlam)      # from Step A4
Q_new = (Q_old + A*dlam*N) / (1 + B*dlam)    # from Step A5

# 6. Recompute after Newton (outside R_plastic, same formulas):
q_new = (q_old + A*dlam) / (1 + B*dlam)
Q_new = (Q_old + A*dlam*N) / (1 + B*dlam)

# 7. Write back to state dict:
state["q"] = jnp.array([q_new])    # scalar: wrap in array
state["Q"] = Q_new                  # tensor: already array
```

---

## PART C — CRITICAL RULES (violations cause wrong results or JAX errors)

### C1 — Elastic stiffness C
```python
# CORRECT — always and only:
C = self.elastic_model.C

# WRONG — never compute manually:
lam = nu*E/((1+nu)*(1-2*nu));  mu = E/(2*(1+nu))
C = jnp.array([[lam+2*mu, lam, ...]])
```

### C2 — jax.lax.cond branch shape matching
Both `elastic_update` and `plastic_update` must return a tuple with:
- **identical number of elements**
- **identical array shapes for each element**

```python
# elastic:  return sig_new, eps_e_new, eps_p_new, X_new, R_new, p_new, fy, dlam
# plastic:  return sig_new, eps_e_new, eps_p_new, X_new, R_new, p_new, fy, dlam
# scalar dlam stays scalar (0.0) in elastic branch — NOT jnp.array([0.0])
# Only wrap in jnp.array([...]) when writing to state dict
```

### C3 — Scalar state variables
```python
# Unpack from state (always [0] to get scalar):
R_old = state["R"][0]
p_old = state["p"][0]

# Write back to state (always wrap):
state["R"] = jnp.array([R_new])
state["p"] = jnp.array([p_new])
```

### C4 — Newton residual sign convention
```python
# Lemaitre-Chaboche form (yield residual):
return J2_eff - (k + R_new) - phi     # = 0 at solution

# Perzyna form (flow rule residual):
return dlam - eta * dt * phi           # = 0 at solution
# NOT: return eta*dt*phi - dlam   ← wrong sign, Newton diverges
```

### C5 — Never reuse values from inside R_plastic after Newton
```python
# WRONG — reusing closure variable:
dlam, _ = newton.solve(0.0)
sig_new = sig_computed_inside_R_plastic   # stale, from last Newton iteration

# CORRECT — recompute with converged dlam:
dlam, _ = newton.solve(0.0)
sig_new = sig_old + C @ (deps - 1.5 * dlam * N)    # fresh computation
```

### C6 — Semi-implicit flow direction
```python
# CORRECT — N fixed at previous step, outside R_plastic:
N = self._flow_direction(sig_old, X_old)
def R_plastic(dlam):
    sig_new = sig_old + C @ (deps - 1.5 * dlam * N)   # N is captured, not updated
    ...

# WRONG — N recomputed inside Newton loop:
def R_plastic(dlam):
    sig_new = sig_old + C @ (deps - 1.5 * dlam * N)
    N = self._flow_direction(sig_new, X_new)   # creates tensor-level coupling
```

### C7 — Initial guess for Newton
```python
newton.solve(0.0)    # always start from zero — never from dlam_old
```

---

## PART D — COMPLETE IMPLICIT UPDATE FORMULAS

Quick lookup — copy these into `R_plastic` and the final state update block.

### Stress
```python
sig_new = sig_old + C @ (deps - 1.5 * dlam * N)    # von Mises / LC
sig_new = sig_old + C @ (deps - dlam * n)           # Perzyna (n = df/dsig)
```

### Plastic strain
```python
eps_p_new = eps_p_old + 1.5 * dlam * N    # von Mises / LC
eps_p_new = eps_p_old + dlam * n          # Perzyna
```

### Elastic strain
```python
eps_e_new = eps_e_old + (deps - 1.5 * dlam * N)    # von Mises / LC
```

### Equivalent plastic strain (trivial linear)
```python
p_new = p_old + dlam
```

### Voce isotropic hardening (nonlinear ODE)
```python
R_new = (R_old + b * R_sat * dlam) / (1.0 + b * dlam)
# parameters: b = saturation rate, R_sat = saturation value
```

### Linear isotropic hardening
```python
R_new = R_old + H * dlam
```

### Armstrong-Frederick backstress (single)
```python
X_new = (X_old + a * dlam * N) / (1.0 + c * dlam)
# parameters: a = modulus, c = dynamic recovery
```

### Two Armstrong-Frederick backstresses (Chaboche multi-surface)
```python
X1_new = (X1_old + a1 * dlam * N) / (1.0 + c1 * dlam)
X2_new = (X2_old + a2 * dlam * N) / (1.0 + c2 * dlam)
X_total = X1_new + X2_new    # use X_total in yield check and residual
```

### Lemaitre damage (explicit-in-D, semi-implicit)
```python
# D̊ = (Y/S)^s · p̊   where Y = S²_eq/(2E·(1-D)²) is strain energy release rate
Y = sig_eq**2 / (2.0 * params.E * (1.0 - D_old)**2)
D_new = D_old + (Y / params.S_damage)**params.s_exp * dlam
# Note: use D_old (semi-implicit) to keep problem scalar in dlam
```

### Viscoplastic overstress terms (use inside R_plastic)
```python
# Lemaitre-Chaboche:
phi = jnp.where(dlam > 0.0, (params.k * dlam / dt)**(1.0/params.n_visc), 0.0)

# Perzyna power law:
phi = jnp.where(f > 0.0, (f / params.sig_y)**params.N, 0.0)

# Perzyna exponential:
phi = jnp.where(f > 0.0, jnp.exp(f / params.sig_0) - 1.0, 0.0)

# Consistency model (enforce f = 0):
# No phi — residual IS the yield function: return J2_eff - sig_y - H*kappa - m*(dlam/dt)
```

---

## PART E — MODEL IDENTIFICATION QUICK GUIDE

When user gives equations, identify model type in 30 seconds:

| What you see in the equations | Model type | Template to use |
|---|---|---|
| `ε̇ᵖ = γ·φ(f)·∂f/∂σ`, no X, no R ODE | Perzyna pure | Template 1 |
| Same + `κ̇ = \|ε̇ᵖ\|` + yield grows with κ | Perzyna + iso hardening | Template 2 |
| `Ē̇ᵖ = ⟨(J₂(S-X)-R-k)/K⟩ⁿ` + Ẋ ODE + Ṙ ODE | Lemaitre-Chaboche | Template 3 |
| `f(σ,κ,κ̇) = 0` (yield function has κ̇) | Consistency | box-algorithms.md Box 4 |
| `ε̇ᵖ = (1/τ)(σ - σ̄)` | Duvaut-Lions | box-algorithms.md Box 3 |
| Any of above + `Ḋ = ...` damage ODE | LC/Perzyna + damage | Template 3 + D extension |
| Two backstress terms X1, X2 | Chaboche multi-surface | Template 3, add X2 |

---

## PART F — PITFALLS AND FIXES

| Symptom | Root cause | Fix |
|---|---|---|
| Newton doesn't converge | Wrong residual sign | Perzyna: `dlam - eta*dt*phi`. LC: `J2_eff - k - R_new - phi` |
| Stress explodes after plasticity | Flow direction wrong | Verify N = dev(S-X)/J₂(S-X), check factor 3/2 |
| R or X grows without bound | Explicit Euler used for nonlinear ODE | Use rational form `(old + A*dlam)/(1 + B*dlam)` |
| JAX shape error in lax.cond | Branch tuple mismatch | Count outputs of both branches — must match exactly |
| `state["R"][0]` wrong value | Forgot `[0]` to unpack scalar | Always unpack scalars with `[0]` |
| dlam always 0, no plastic flow | Yield check wrong sign or wrong variable | `fy = J₂(sig_trial - X_old) - k - R_old` |
| dlam converges to wrong value | N computed inside Newton | Move N = `_flow_direction(sig_old, X_old)` outside R_plastic |
| C gives wrong stiffness | Manual C computation | Use `C = self.elastic_model.C` only |
| AD gradient NaN at zero stress | Raw `jnp.sqrt` in J₂ | Use `_J2` with `stop_gradient` trick |
| Voigt shear terms wrong | Missing factor-of-2 | `_J2` uses `2*(s12²+s23²+s13²)` not `s12²+...` |

---

## PART G — REFERENCE FILE MAP

| File | Purpose | When to read |
|---|---|---|
| `code-template.md` | 3 complete working templates + adaptation guide | Before writing any code |
| `quick-reference.md` | Model decision tree, residual snippets, parameter ranges, debug table | During coding, fast lookup |
| `box-algorithms.md` | Mathematical algorithm boxes for all 4 model types + LC full derivation | When deriving or verifying math |
| `Lemiature_Chaboche_implicit_model.py` | Complete working LC implementation | Reference for LC-type models |
| `LC_implicit_scheme_hand_calculation.pdf` | Hand-written derivation of LC scheme | Reference for derivation procedure |

---

## PART H — NOTATION REFERENCE

| Symbol | Meaning | Code variable |
|--------|---------|---------------|
| Δλ, ΔĒᵖ | Plastic multiplier increment | `dlam` |
| N | Flow direction dev(S-X)/J₂(S-X) | `N` or `n_dir` |
| Ŝ | Trial stress | `sig_trial` |
| μ | Shear modulus E/(2(1+ν)) | computed from `elastic_model` |
| κ (bulk) | Bulk modulus E/(3(1-2ν)) | computed from `elastic_model` |
| ⟨x⟩ | McCauley bracket = max(x,0) | `jnp.where(x > 0, x, 0)` |
| ()' | Deviatoric part | `_deviatoric(sig)` |
| J₂(·) | sqrt(3/2 · dev(·):dev(·)) | `_J2(_deviatoric(sig))` |
| S | Stress tensor (Voigt 6-vector) | `sig` |
| X | Backstress tensor (Voigt 6-vector) | `X` |
| R | Isotropic hardening variable (scalar) | `R` |
| p | Equivalent plastic strain (scalar) | `p` |