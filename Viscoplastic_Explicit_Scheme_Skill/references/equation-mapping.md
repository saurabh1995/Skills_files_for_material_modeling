# Equation to Code Mapping Guide

This reference shows how to translate common constitutive equations into JAX-compatible Python code for the explicit integration scheme.

## ⚠️ Read First: The Corrected Two-Phase Pattern

All plastic update implementations in this file follow the **corrected explicit sequence**:
- **Phase 1**: Advance state variables using rates stored from the *previous* step.
- **Phase 2**: Re-evaluate the yield function and all rates at the *new* state, and store them for the next step.

Never compute the plastic strain increment directly from `fy_trial`. See `code-template.md` for full details.

---

## Fundamental Equations

### 1. Strain Decomposition

**Equation:**
```
ε = εe + εp
```

**Code (inside `_plastic_update`, Phase 1):**
```python
# Phase 1: advance using old rates
delta_eps_p = eps_p_dot_old * dt
eps_p_new   = eps_p_old + delta_eps_p
eps_e_new   = eps_e_old + (deps - delta_eps_p)
```

---

### 2. Stress-Strain Law

**Equation (without damage):**
```
σ = C : (ε - εp)
```

**Code (Phase 1, after updating εp):**
```python
sig_new = C @ (eps - eps_p_new)
```

**Equation (with damage):**
```
σ̇ = (1-D) C : ε̇e
```

**Code (Phase 1, with damage):**
```python
D_new   = D_old + D_dot_old * dt
delta_eps_p = eps_p_dot_old * dt
delta_eps_e = deps - delta_eps_p
sig_new = sig_old + (1.0 - D_new) * (C @ delta_eps_e)
```

---

### 3. Elastic Predictor

**Code (before branching — uses OLD εp):**
```python
sig_trial = C @ (eps - eps_p_old)
# With damage: sig_trial = sig_old + (1.0 - D_old)*(C @ deps)
```

---

### 4. Yield Function

**Equation (Chaboche — backstress + hardening, no damage):**
```
f = J₂(σ - X) - (R + σy)
```

**Code (at trial state, for gate check):**
```python
sig_trial_dev  = self._deviatoric(sig_trial)
n_trial        = sig_trial_dev - alpha1_old  # subtract all backstresses
sigma_eq_trial = self._equivalent_norm(n_trial)
fy_trial       = sigma_eq_trial - (params.sigma_y + R_old)
```

**Code (Phase 2 — at new state):**
```python
sig_new_dev  = self._deviatoric(sig_new)
n_new        = sig_new_dev - alpha1_new    # subtract updated backstresses
sigma_eq_new = self._equivalent_norm(n_new)
fy_new       = sigma_eq_new - (params.sigma_y + R_new)
```

**With damage (at new state):**
```python
sigma_eq_new = self._equivalent_norm(n_new) / (1.0 - D_new)
fy_new       = sigma_eq_new - (params.sigma_y + R_new)
```

---

### 5. Viscoplastic Consistency (Perzyna)

**Equation:**
```
λ̇ = 〈f/η〉ⁿ
```

**Code (Phase 2 only — evaluated at new state):**
```python
x              = fy_new / params.eta
bracket        = 0.5 * (x + jnp.abs(x))   # McCauley bracket = max(x, 0)
lambda_dot_new = jnp.power(bracket, params.n_visc)
```

**WRONG (never do this — uses trial state):**
```python
# WRONG: do not compute lambda_dot from fy_trial
bracket = 0.5 * (fy_trial/params.eta + jnp.abs(fy_trial/params.eta))
dlambda = dt * jnp.power(bracket, params.n_visc)  # ← WRONG
```

---

### 6. Flow Rule

**Equation (associative):**
```
ε̇p = λ̇ × (3/2) × (σ' - X') / J₂(σ - X)
```

**Code (Phase 2 — uses `n_new` and `sigma_eq_new` from Phase 2):**
```python
inv_sigma_eq  = jnp.where(sigma_eq_new > 0.0, 1.0/sigma_eq_new, 0.0)
flow_dir      = n_new * inv_sigma_eq           # normalized flow direction
eps_p_dot_new = lambda_dot_new * 1.5 * flow_dir
```

**WRONG (uses trial flow direction):**
```python
# WRONG: do not use n_trial as flow direction for the increment
delta_eps_p = 1.5 * dlambda * n_trial * inv_sigma_eq_trial  # ← WRONG
```

---

## Hardening Evolution

### 7. Isotropic Hardening

**Equation (linear):**
```
Ṙ = H λ̇
```

**Phase 1 (advance R):**
```python
R_new = R_old + R_dot_old * dt
```

**Phase 2 (recompute rate at new state):**
```python
R_dot_new = params.H * lambda_dot_new
```

**Equation (exponential saturation):**
```
Ṙ = b(R₁ - R) λ̇
```

**Phase 1:**
```python
R_new = R_old + R_dot_old * dt
```

**Phase 2:**
```python
R_dot_new = params.b * (params.R1 - R_new) * lambda_dot_new
```

---

### 8. Kinematic Hardening (Armstrong-Frederick)

**Equation:**
```
α̇ᵢ = (2/3) Cᵢ ε̇p - γᵢ αᵢ λ̇
```

**Phase 1 (advance backstress):**
```python
alpha1_new = alpha1_old + alpha1_dot_old * dt
alpha2_new = alpha2_old + alpha2_dot_old * dt   # if two backstresses
```

**Phase 2 (recompute rate at new state):**
```python
alpha1_dot_new = (
    (2.0/3.0) * params.C1 * eps_p_dot_new
    - params.gamma1 * alpha1_new * lambda_dot_new
)
alpha2_dot_new = (
    (2.0/3.0) * params.C2 * eps_p_dot_new
    - params.gamma2 * alpha2_new * lambda_dot_new
)
```

**Note**: `C1` and `C2` can be different (that is correct for two-backstress models). Using the same `C1` for both backstresses is wrong unless the user's equations specify that.

---

## Damage Evolution

### 9. Lemaitre Damage

**Equation:**
```
Ḋ = [Dc/(εR - εD)] × [(2/3)(1+ν)σ²eq + 3(1-2ν)σ²H] × λ̇
```

**Phase 1 (advance D):**
```python
D_new = jnp.clip(D_old + D_dot_old * dt, 0.0, params.Dc)
```

**Phase 2 (recompute damage rate at new state):**
```python
sigma_eq_s    = self._equiv_stress(sig_new)
sigma_H_s     = self._hydrostatic(sig_new)
nu            = params.nu
bracket_D     = (
    (2.0/3.0) * (1.0 + nu) * sigma_eq_s**2
    + 3.0 * (1.0 - 2.0*nu) * sigma_H_s**2
)
factor_D      = params.Dc / (params.eps_R - params.eps_D + 1e-12)
D_dot_new     = factor_D * bracket_D * lambda_dot_new
```

---

## Helper Functions

### J2 Equivalent Norm

```python
def _equivalent_norm(self, vec):
    """sqrt(3/2 * vec:vec), regularized for gradients."""
    val     = 1.5 * self._norm_voigt(vec)
    val_pos = jnp.maximum(val, 0.0)
    eps_reg = 1e-16
    phys    = jnp.sqrt(val_pos)
    reg     = jnp.sqrt(val_pos + eps_reg)
    return jax.lax.stop_gradient(phys - reg) + reg
```

### Deviatoric Stress (Voigt)

```python
def _deviatoric(self, sig):
    p = (sig[0] + sig[1] + sig[2]) / 3.0
    return jnp.array([
        sig[0]-p, sig[1]-p, sig[2]-p,
        sig[3], sig[4], sig[5],
    ], dtype=sig.dtype)
```

### Hydrostatic Stress

```python
def _hydrostatic(self, sig):
    return (sig[0] + sig[1] + sig[2]) / 3.0
```

### Equivalent (von Mises) Stress

```python
def _equiv_stress(self, sig):
    return self._equivalent_norm(self._deviatoric(sig))
```

---

## Equation Patterns

### Pattern 1: Two-Phase for Any Rate Variable

For any `Q̇ = f(Q, λ̇, ε̇p, ...)`:

**Phase 1** (inside `_plastic_update`):
```python
Q_new = Q_old + Q_dot_old * dt
```

**Phase 2** (after computing `sig_new`, `lambda_dot_new`, `eps_p_dot_new`):
```python
Q_dot_new = f(Q_new, lambda_dot_new, eps_p_dot_new, ...)
```

**State storage** (outside both branches):
```python
state["Q"]     = jnp.array([Q_new])
state["Q_dot"] = jnp.array([Q_dot_new])
```

### Pattern 2: Saturation-Type Evolution

`Q̇ = a(Q∞ - Q) λ̇`

```python
# Phase 1
Q_new = Q_old + Q_dot_old * dt
# Phase 2
Q_dot_new = params.a * (params.Q_inf - Q_new) * lambda_dot_new
```

### Pattern 3: Recovery-Type (Armstrong-Frederick)

`Q̇ = a ε̇p - b Q λ̇`

```python
# Phase 1
Q_new = Q_old + Q_dot_old * dt
# Phase 2
Q_dot_new = params.a * eps_p_dot_new - params.b * Q_new * lambda_dot_new
```

---

## Common Model Combinations

### Model 1: Single-Backstress Viscoplastic (Chaboche)

| Equation               | Phase 1 advances  | Phase 2 recomputes      |
|------------------------|-------------------|-------------------------|
| Strain decomp.         | εp, εe            | —                       |
| Stress                 | σ                 | —                       |
| Yield + viscoplastic   | —                 | f, λ̇, ε̇p              |
| Isotropic hardening    | R                 | Ṙ                       |
| Kinematic hardening    | α                 | α̇                       |

### Model 2: Two-Backstress Viscoplastic

Same as Model 1, with `alpha1`, `alpha2` each having their own rate (`C1,γ1` and `C2,γ2`).

### Model 3: Lemaitre-Chaboche (with damage)

Adds `D` (advanced in Phase 1) and `D_dot` (recomputed in Phase 2). Stress update uses `(1-D_new)`.

---

## Time Integration Note

All equations use **forward Euler** (explicit). The corrected two-phase sequence ensures that the rates consumed in Phase 1 were accurately computed at the *state at the end of the previous step*, making it a proper explicit scheme.
