# quick-reference.md — Fast Lookup for Implicit Viscoplastic Models

---

## 1 — MODEL IDENTIFICATION TREE

```
Given new equations — work through this tree:

Does the flow rule involve backstress X?
│
├─ YES (J₂(S-X) in flow rule)
│   │
│   └─ Does it have both Ṙ (isotropic) and Ẋ (kinematic) ODEs?
│       ├─ YES → LEMAITRE-CHABOCHE  (Template 3 in code-template.md)
│       └─ NO, only Ẋ → KINEMATIC ONLY  (Template 3, remove R parts)
│
└─ NO (no backstress, only σ in flow rule)
    │
    └─ Is the yield surface rate-dependent?  f = f(σ, κ, κ̇)?
        │
        ├─ YES, f contains κ̇ → CONSISTENCY MODEL  (box-algorithms.md Box 4)
        │
        └─ NO
            │
            └─ Is flow rule: ε̇ᵖ = (σ - σ̄)/η  or  "relaxation time"?
                ├─ YES → DUVAUT-LIONS  (box-algorithms.md Box 3)
                └─ NO, flow rule: ε̇ᵖ = η·φ(f)·∂f/∂σ → PERZYNA
                    │
                    └─ Does yield surface grow with plastic strain?
                        ├─ NO  → Template 1 (pure plasticity)
                        └─ YES → Template 2 (isotropic hardening)
```

---

## 2 — RESIDUAL SNIPPETS BY MODEL TYPE

Copy-paste into `R_plastic(dlam)`. Replace with correct parameter names.

### Lemaitre-Chaboche (full)
```python
def R_plastic(dlam):
    # Step A3
    sig_new = sig_old + C @ (deps - 1.5 * dlam * N)         # N fixed outside
    # Step A4 — Voce isotropic
    R_new   = (R_old + params.b * params.R1 * dlam) / (1.0 + params.b * dlam)
    # Step A5 — Armstrong-Frederick backstress
    X_new   = (X_old + params.a * dlam * N) / (1.0 + params.c * dlam)
    # Step A6 — overstress + residual
    phi     = jnp.where(dlam > 0.0, (params.k * dlam / dt)**(1.0/params.n_visc), 0.0)
    J2_eff  = self._J2(self._deviatoric(sig_new - X_new))
    return J2_eff - (params.k + R_new) - phi
```

### Lemaitre-Chaboche, two backstresses (Chaboche multi-surface)
```python
def R_plastic(dlam):
    sig_new = sig_old + C @ (deps - 1.5 * dlam * N)
    R_new   = (R_old  + params.b  * params.R1 * dlam) / (1.0 + params.b  * dlam)
    X1_new  = (X1_old + params.a1 * dlam * N)         / (1.0 + params.c1 * dlam)
    X2_new  = (X2_old + params.a2 * dlam * N)         / (1.0 + params.c2 * dlam)
    X_total = X1_new + X2_new
    phi     = jnp.where(dlam > 0.0, (params.k * dlam / dt)**(1.0/params.n_visc), 0.0)
    J2_eff  = self._J2(self._deviatoric(sig_new - X_total))
    return J2_eff - (params.k + R_new) - phi
```

### Perzyna, constant yield
```python
def R_plastic(dlam):
    sig_new = sig_old + C @ (deps - dlam * n)                # n fixed outside
    J2      = self._J2(self._deviatoric(sig_new))
    f       = J2 - params.sig_y
    phi     = jnp.where(f > 0.0, (f / params.sig_y)**params.N, 0.0)
    return dlam - params.eta * dt * phi
```

### Perzyna, linear isotropic hardening
```python
def R_plastic(dlam):
    sig_new = sig_old + C @ (deps - dlam * n)
    p_new   = p_old + dlam                                    # trivial linear update
    J2      = self._J2(self._deviatoric(sig_new))
    f       = J2 - (params.sig_y + params.H * p_new)
    phi     = jnp.where(f > 0.0, (f / params.sig_y)**params.N, 0.0)
    return dlam - params.eta * dt * phi
```

### Perzyna, Voce nonlinear isotropic hardening
```python
def R_plastic(dlam):
    sig_new = sig_old + C @ (deps - dlam * n)
    # Voce: Ṙ = b·(R_sat - R)·p̊  →  R_t = (R_n + b·R_sat·Δλ)/(1 + b·Δλ)
    R_new   = (R_old + params.b * params.R_sat * dlam) / (1.0 + params.b * dlam)
    J2      = self._J2(self._deviatoric(sig_new))
    f       = J2 - (params.sig_y + R_new)
    phi     = jnp.where(f > 0.0, (f / params.sig_y)**params.N, 0.0)
    return dlam - params.eta * dt * phi
```

### Consistency model (rate-dependent yield surface)
```python
def R_plastic(dlam):
    sig_new   = sig_old + C @ (deps - dlam * n)
    kappa_new = kappa_old + dlam
    kappa_dot = dlam / dt
    J2        = self._J2(self._deviatoric(sig_new))
    return J2 - (params.sig_y + params.H * kappa_new + params.m * kappa_dot)
```

---

## 3 — IMPLICIT HARDENING UPDATE FORMULAS

Quick lookup — use BOTH inside R_plastic AND in final state update after Newton.

| ODE (continuous) | Implicit update | Stability |
|---|---|---|
| `ṗ = Ēdot_p` (linear) | `p_new = p_old + dlam` | always stable |
| `Ṙ = H·Ēdot_p` (linear) | `R_new = R_old + H*dlam` | always stable |
| `Ṙ = b(R_sat-R)·Ēdot_p` (Voce) | `R_new = (R_old + b*R_sat*dlam)/(1+b*dlam)` | stable, saturates to R_sat |
| `Ẋ = a·Ēdot_p·N - c·X·Ēdot_p` (A-F) | `X_new = (X_old + a*dlam*N)/(1+c*dlam)` | stable, saturates to (a/c)·N |
| `Ẋ = a·Ēdot_p·N` (linear kinematic) | `X_new = X_old + a*dlam*N` | stable |
| `Ḋ = (Y/S)^s·Ēdot_p` (Lemaitre damage) | `D_new = D_old + (Y/S)**s * dlam` | semi-implicit in D |

---

## 4 — OVERSTRESS / FLOW FUNCTIONS

Change only the `phi` line inside `R_plastic`.

```python
# Lemaitre-Chaboche (current model):
phi = jnp.where(dlam > 0.0, (params.k * dlam / dt)**(1.0/params.n_visc), 0.0)

# Perzyna power law:
phi = jnp.where(f > 0.0, (f / params.sig_y)**params.N, 0.0)

# Perzyna linear (N=1):
phi = jnp.where(f > 0.0, f / params.sig_y, 0.0)

# Perzyna exponential:
phi = jnp.where(f > 0.0, jnp.exp(f / params.sig_0) - 1.0, 0.0)

# Duvaut-Lions (no phi — different residual structure, see box-algorithms.md Box 3)
```

---

## 5 — YIELD FUNCTION FORMS

Change the `J2_eff - (k + R_new) - phi` line in R_plastic.

```python
# Von Mises, no hardening:
f = self._J2(self._deviatoric(sig_new)) - params.sig_y

# Von Mises, isotropic hardening only:
f = self._J2(self._deviatoric(sig_new)) - (params.sig_y + R_new)

# Von Mises, kinematic hardening only:
f = self._J2(self._deviatoric(sig_new - X_new)) - params.sig_y

# Von Mises, iso + kinematic (LC):
f = self._J2(self._deviatoric(sig_new - X_new)) - (params.k + R_new)

# Drucker-Prager (pressure-dependent):
p_new  = -(sig_new[0] + sig_new[1] + sig_new[2]) / 3.0
J2_new = self._J2(self._deviatoric(sig_new))
f      = J2_new + params.alpha * p_new - params.k
# NOTE: flow direction is NOT purely deviatoric for Drucker-Prager
# => C:N simplification does NOT apply, use full C @ (deps - dlam*n)
```

---

## 6 — JAX PATTERNS

### Conditional branching
```python
# operand must contain ALL variables both branches need
operand = (eps_old, sig_old, eps_p_old, eps_e_old, X_old, R_old, p_old, deps, sig_trial, fy, dt)

def elastic_update(operand):
    ...
    return sig_new, eps_p_new, eps_e_new, X_new, R_new, p_new, fy, dlam   # dlam=0.0

def plastic_update(operand):
    ...
    return sig_new, eps_p_new, eps_e_new, X_new, R_new, p_new, fy, dlam

# BOTH returns: same count, same shapes
result = jax.lax.cond(fy > 0.0, plastic_update, elastic_update, operand)
```

### Newton solver
```python
newton = JAXNewton()
newton.set_residual(R_plastic)     # R_plastic must take scalar, return scalar
dlam, _ = newton.solve(0.0)        # ALWAYS start from 0
```

### Regularized J2 — ALWAYS use this form
```python
def _J2(self, sig_dev):
    s  = sig_dev
    ss = (s[0]*s[0] + s[1]*s[1] + s[2]*s[2]
          + 2.0*(s[3]*s[3] + s[4]*s[4] + s[5]*s[5]))
    val     = jnp.maximum(1.5 * ss, 0.0)
    J2_phys = jnp.sqrt(val)
    J2_reg  = jnp.sqrt(val + 1e-16)
    return jax.lax.stop_gradient(J2_phys - J2_reg) + J2_reg
```

---

## 7 — DEBUG CHECKLIST

| Symptom | Most likely cause | Fix |
|---|---|---|
| Newton doesn't converge | Wrong residual sign | Perzyna: `dlam - eta*dt*phi`. LC: `J2_eff - k - R - phi` |
| dlam always 0 | Yield check wrong | Check `fy = J₂(sig_trial - X_old) - R_old - k` (not R_new) |
| Stress explodes in plastic | Flow direction error | Verify N or n formula, check (3/2) factor for LC |
| R or X unbounded growth | Explicit ODE update used | Switch to rational form `(old + A*dlam)/(1+B*dlam)` |
| `jax.lax.cond` shape error | Branch tuple mismatch | Print `jnp.shape` of each return element in both branches |
| Wrong R value in state | Missing `[0]` on unpack | `R_old = state["R"][0]` not `state["R"]` |
| J2 gradient NaN | Raw sqrt at zero | Use `_J2` with `stop_gradient`, not `jnp.sqrt(1.5*ss)` |
| Voigt shear factor error | Missing 2× on shear terms | `ss = s0²+s1²+s2² + 2*(s3²+s4²+s5²)` |
| C matrix wrong | Manual Lamé params | `C = self.elastic_model.C` only |
| Wrong final state after Newton | Reusing closure vars | Recompute all state after `newton.solve` using same formulas |

---

## 8 — MATERIAL PARAMETER RANGES

| Parameter | Metals | Polymers |
|---|---|---|
| E (Young's modulus) | 10³–10⁵ MPa | 10¹–10³ MPa |
| ν (Poisson's ratio) | 0.25–0.35 | 0.35–0.49 |
| σᵧ / k (yield stress) | 10²–10³ MPa | 10–100 MPa |
| K_visc (LC viscosity) | 1–50 MPa | 0.1–10 MPa |
| n_visc (LC exponent) | 2–10 | 1–4 |
| η (Perzyna fluidity) | 10⁻⁶–10⁻² MPa⁻¹s⁻¹ | 10⁻⁴–10⁻¹ |
| N (Perzyna exponent) | 1–5 | 1–3 |
| H (linear hardening) | −E/10 to E/10 | varies |
| b (Voce rate) | 5–100 | 1–20 |
| a (A-F modulus) | 10³–10⁵ MPa | 10²–10⁴ |
| c / s (A-F recovery) | 10²–10³ | 10–10² |

**Validated LC values (Tandale & Stoffel 2024):**

| | E (MPa) | ν | n | a | c | K (MPa) | k (MPa) |
|---|---|---|---|---|---|---|---|
| Aluminum | 67 400 | 0.32 | 2.79 | 3100 | 110 | 3.42 | 110 |
| Copper | 113 066 | 0.32 | 8.15 | 98 939 | 1533 | 11.45 | 180 |

---

## 9 — TIME STEP GUIDELINES

| Regime | Δt guideline |
|---|---|
| Quasi-static, viscous metals | 10⁻⁴ – 1.0 s |
| Rate-sensitive (check convergence) | Δt << K/E |
| Dynamic (explicit FEM needed) | Δt < h/c_wave |
| Very stiff viscoplastic | Small Δt, verify Δλ << 1 |

---

## 10 — FILE STRUCTURE

```
my_material.py
├── Imports (jax, jnp, dataclass, JAXNewton, JAXMaterial, tangent_AD)
├── @dataclass MyModel_Params(JAXMaterial)    ← parameters only
└── class MyModel(JAXMaterial)
    ├── __init__(elastic_model, params)
    ├── gradient_names → ("Strain",)
    ├── flux_names → ("Stress",)
    ├── internal_state_variables → {name: size}
    ├── _deviatoric(sig)
    ├── _J2(sig_dev)
    ├── _flow_direction(sig, X)   ← if backstress present
    ├── df_dsigma(sig)            ← if Perzyna type
    └── @tangent_AD constitutive_update(eps, state, dt)
        ├── unpack state
        ├── C = self.elastic_model.C
        ├── elastic predictor + yield check
        ├── operand = (...)
        ├── def elastic_update(operand): ...
        ├── def plastic_update(operand):
        │   ├── N = _flow_direction(sig_old, X_old)  [outside R_plastic]
        │   ├── def R_plastic(dlam):
        │   │   ├── sig_new = sig_old + C@(deps - 1.5*dlam*N)  [A3]
        │   │   ├── R_new = (R_old + b*R1*dlam)/(1+b*dlam)     [A4]
        │   │   ├── X_new = (X_old + a*dlam*N)/(1+c*dlam)      [A5]
        │   │   ├── phi = K*(dlam/dt)^(1/n)                    [A6]
        │   │   └── return J2_eff - (k + R_new) - phi          [A6]
        │   ├── newton.solve(0.0)
        │   └── recompute all state with converged dlam
        ├── jax.lax.cond(fy > 0.0, plastic_update, elastic_update, operand)
        └── write ALL state → state dict → return sig_new, state
```