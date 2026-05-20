---
name: explicit-integration-scheme
description: Generate explicit time integration schemes for constitutive models. Use when users provide constitutive equations (stress-strain relations, damage evolution, hardening laws, flow rules) and need a FEniCSx-JAX-compatible explicit integration algorithm in the style of Lemaitre-Chaboche or Perzyna viscoplasticity with elastic predictor-plastic corrector structure. Trigger when user asks to create explicit scheme, integration algorithm, or material model implementation from equations.
---

# Explicit Integration Scheme Generator

Generate explicit time integration schemes for constitutive material models, following the radial return mapping algorithm pattern.

## Core Workflow

When the user provides constitutive equations, follow this sequence:
1. **Identify equation types** in the provided model:
   - Strain decomposition (elastic + inelastic)
   - Stress-strain relationship (damaged/undamaged)
   - Yield function
   - Flow rule (plastic/viscoplastic strain rate)
   - Hardening evolution (isotropic and/or kinematic)
   - Damage evolution

2. **Generate elastic predictor step** - Trial stress assuming elastic behavior

3. **Generate yield check** - Evaluate yield function with trial stress to gate plasticity

4. **Generate plastic corrector step** - If yielding occurs, use the TWO-PHASE pattern (see below)

5. **Format as FEniCSx-JAX-compatible Python code** following the reference template structure strictly

---

## ⚠️ CRITICAL: The Corrected Explicit Sequence (ALWAYS USE THIS)

**This is the single most important rule in this skill. The plastic update MUST follow a two-phase pattern. Violation of this causes wrong results that are hard to detect.**

### Why two phases?

The explicit scheme is a forward-Euler method. At time step `i`, the state update uses **rates computed at the end of step `i-1`** (i.e., already stored in `state`). After advancing the state, the rates are **re-evaluated at the new state `i`** and stored for use in step `i+1`.

This means the plastic branch has two distinct phases:

### Phase 1 — Advance state using OLD rates (from previous step)
Use the rates already stored in `state` (`lambda_dot_old`, `eps_p_dot_old`, `R_dot_old`, `X_dot_old`, etc.) to advance all state variables:

```
Δλ       = lambda_dot_old * dt
Δεp      = eps_p_dot_old * dt
εp_new   = εp_old + Δεp
εe_new   = εe_old + (Δε - Δεp)
R_new    = R_old + R_dot_old * dt
α_new    = α_old + α_dot_old * dt
σ_new    = C : (ε - εp_new)          ← recompute from updated εp
```

### Phase 2 — Re-evaluate ALL rates at the NEW state
Recompute the yield function, viscoplastic multiplier, flow direction, and all evolution rates using the updated variables `(σ_new, εp_new, R_new, α_new, ...)`:

```
n_new        = dev(σ_new) - α_new
σ_eq_new     = sqrt(3/2 n_new:n_new)
f_new        = σ_eq_new - (σ_y + R_new)
λ_dot_new    = <f_new / η>^n
eps_p_dot_new = λ_dot_new * (3/2) * n_new / σ_eq_new
R_dot_new    = H * λ_dot_new
α_dot_new    = (2/3)*C1*eps_p_dot_new - γ*α_new*λ_dot_new
```

These newly computed rates are stored in `state` and will be used as the "old rates" in the NEXT time step.

### What the WRONG (standard) sequence looks like — NEVER DO THIS
The wrong approach computes Δλ and Δεp directly from the trial yield function `fy_trial` using `bracket = 0.5*(fy_trial/η + |fy_trial/η|)`, applies those increments to get `εp_new`, and then stores those same increments as the rates. This is wrong because it conflates the trial state with the updated state, and does not store properly-evaluated rates for the next step.

**Key diagnostic**: If your `_plastic_update` contains `jnp.power(bracket, n_visc)` where `bracket` is computed from `fy_trial` (the yield function evaluated at the trial stress), you are using the WRONG sequence.

**Key diagnostic for correct**: In `_plastic_update`, `dlambda = lambda_dot_old * dt` should appear first, advancing state. The `jnp.power(...)` call should appear AFTER `sig_new` has been computed, using `fy` evaluated at `sig_new`.

---

## Code Structure Pattern

```python
@tangent_AD
def constitutive_update(self, eps, state, dt):
    params = self.params
    C = self.elastic_model.C

    # 1. EXTRACT OLD STATE AND OLD RATES
    eps_p_old     = state["eps_p"]
    eps_e_old     = state["eps_e"]
    R_old         = state["R"][0]
    alpha1_old    = state["alpha1"]    # etc. for all backstresses
    # Old rates (computed at end of previous step, used to advance now)
    lambda_dot_old = state["lambda_dot"][0]
    eps_p_dot_old  = state["eps_p_dot"]
    R_dot_old      = state["R_dot"][0]
    alpha1_dot_old = state["alpha1_dot"]

    # 2. ELASTIC PREDICTOR — trial stress using OLD plastic strain
    sig_trial = C @ (eps - eps_p_old)   # (no damage) or with (1-D) factor

    # 3. YIELD CHECK on trial stress
    sig_trial_dev = self._deviatoric(sig_trial)
    n_trial = sig_trial_dev - (alpha1_old + ...)
    sigma_eq_trial = self._equivalent_norm(n_trial)
    fy_trial = sigma_eq_trial - (params.sigma_y + R_old)

    # 4. PACK OPERAND (include old rates)
    operand = (eps, eps_p_old, eps_e_old, R_old, alpha1_old, ...,
               lambda_dot_old, eps_p_dot_old, R_dot_old, alpha1_dot_old, ...,
               sig_trial, fy_trial, dt)

    # 5. ELASTIC BRANCH — no change to plastic variables or rates
    def _elastic_update(operand):
        (...) = operand
        sig_new   = sig_trial
        eps_p_new = eps_p_old
        eps_e_new = eps - eps_p_new
        R_new     = R_old
        alpha1_new = alpha1_old
        # All rates remain zero / old
        lambda_dot_new = 0.0
        eps_p_dot_new  = jnp.zeros(6, dtype=eps.dtype)
        R_dot_new      = 0.0
        alpha1_dot_new = jnp.zeros(6, dtype=eps.dtype)
        dlambda        = 0.0
        fy_out         = fy_trial
        is_plastic_out = jnp.array(0.0, dtype=eps.dtype)
        return (sig_new, eps_p_new, eps_e_new, R_new, alpha1_new, ...,
                lambda_dot_new, eps_p_dot_new, R_dot_new, alpha1_dot_new, ...,
                dlambda, fy_out, is_plastic_out)

    # 6. PLASTIC BRANCH — TWO-PHASE CORRECTED SEQUENCE
    def _plastic_update(operand):
        (...) = operand

        # ── PHASE 1: Advance state using OLD rates ──────────────────────────
        dlambda      = lambda_dot_old * dt           # increment from old rate
        delta_eps_p  = eps_p_dot_old * dt            # plastic strain increment
        eps_p_new    = eps_p_old + delta_eps_p
        delta_eps_e  = deps - delta_eps_p            # deps = eps - eps_old
        eps_e_new    = eps_e_old + delta_eps_e
        R_new        = R_old + R_dot_old * dt
        alpha1_new   = alpha1_old + alpha1_dot_old * dt

        sig_new      = C @ (eps - eps_p_new)         # stress at updated εp

        # ── PHASE 2: Re-evaluate rates at NEW state ──────────────────────────
        sig_new_dev   = self._deviatoric(sig_new)
        n_new         = sig_new_dev - (alpha1_new + ...)
        sigma_eq_new  = self._equivalent_norm(n_new)
        fy_new        = sigma_eq_new - (params.sigma_y + R_new)

        x             = fy_new / params.eta
        bracket       = 0.5 * (x + jnp.abs(x))
        lambda_dot_new = jnp.power(bracket, params.n_visc)

        inv_sigma_eq  = jnp.where(sigma_eq_new > 0.0, 1.0 / sigma_eq_new, 0.0)
        flow_dir      = n_new * inv_sigma_eq
        eps_p_dot_new = lambda_dot_new * 1.5 * flow_dir

        R_dot_new     = params.H * lambda_dot_new
        alpha1_dot_new = ((2.0/3.0)*params.C1*eps_p_dot_new
                          - params.gamma1*alpha1_new*lambda_dot_new)

        is_plastic_out = jnp.array(1.0, dtype=eps.dtype)
        return (sig_new, eps_p_new, eps_e_new, R_new, alpha1_new, ...,
                lambda_dot_new, eps_p_dot_new, R_dot_new, alpha1_dot_new, ...,
                dlambda, fy_new, is_plastic_out)

    # 7. CONDITIONAL BRANCHING
    (...) = jax.lax.cond(fy_trial > 0.0, _plastic_update, _elastic_update, operand)

    # 8. UPDATE STATE — include ALL rate variables
    state["Strain"]        = eps
    state["Stress"]        = sig_new
    state["eps_p"]         = eps_p_new
    state["eps_e"]         = eps_e_new
    state["R"]             = jnp.array([R_new])
    state["alpha1"]        = alpha1_new
    state["lambda_dot"]    = jnp.array([lambda_dot_new])
    state["dlambda"]       = jnp.array([dlambda])
    state["eps_p_dot"]     = eps_p_dot_new
    state["R_dot"]         = jnp.array([R_dot_new])
    state["alpha1_dot"]    = alpha1_dot_new
    state["fy"]            = jnp.array([fy_new])
    state["is_plastic"]    = jnp.array([is_plastic_out])

    return sig_new, state
```

---

## State Variables — What Must Be Stored

The state must store **both the integrated variables AND the rates** so that the next step can advance using them. For every evolution equation `Q̇ = f(...)`, store `Q` AND `Q_dot`.

Standard set for a two-backstress viscoplastic model without damage:

```python
@property
def internal_state_variables(self):
    return {
        "eps_p":      6,   # plastic strain tensor
        "eps_e":      6,   # elastic strain tensor
        "R":          1,   # isotropic hardening variable
        "alpha1":     6,   # first backstress tensor
        "alpha2":     6,   # second backstress tensor
        # Rates — MANDATORY for corrected sequence
        "lambda_dot": 1,   # viscoplastic multiplier rate
        "dlambda":    1,   # viscoplastic multiplier increment (for diagnostics)
        "eps_p_dot":  6,   # plastic strain rate tensor
        "R_dot":      1,   # isotropic hardening rate
        "alpha1_dot": 6,   # first backstress rate
        "alpha2_dot": 6,   # second backstress rate
        # Diagnostics
        "fy":         1,   # yield function value
        "is_plastic": 1,   # plastic flag
    }
```

**CRITICAL**: If a model has additional evolution equations (damage `Ḋ`, additional backstresses, etc.), each must have its corresponding `_dot` state variable AND that `_dot` variable must be read at the start of `constitutive_update` as an "old rate" and updated at the end of the plastic branch Phase 2.

---

## JAX Compatibility Requirements

- **No Python conditionals**: Use `jax.lax.cond` for if/else
- **Use jnp not np**: All operations must use `jax.numpy` (imported as `jnp`)
- **No in-place mutations**: Create new arrays instead of modifying existing ones
- **Decorator required**: Mark with `@tangent_AD` for automatic differentiation
- **Pure functions**: Inner functions (`_elastic_update`, `_plastic_update`) must be pure
- **Elastic stiffness**: Always use `self.elastic_model.C` — never recompute from Lamé parameters

---

## Elastic Stiffness Tensor

**Step 1: Class Initialization**
```python
def __init__(self, elastic_model, params: ModelParams):
    super().__init__()
    self.elastic_model = elastic_model
    self.params = params
    self.dt = 0.0
```

**Step 2: Use in constitutive_update**
```python
C = self.elastic_model.C          # Always this — never recompute
sig_trial = C @ (eps - eps_p_old) # Use directly
```

---

## Numerical Stability Techniques

```python
# 1. Regularized equivalent norm (avoids sqrt(0) in gradients)
def _equivalent_norm(self, vec):
    val = 1.5 * self._norm_voigt(vec)
    val_pos = jnp.maximum(val, 0.0)
    eps_reg = 1e-16
    sigma_phys = jnp.sqrt(val_pos)
    sigma_reg  = jnp.sqrt(val_pos + eps_reg)
    return jax.lax.stop_gradient(sigma_phys - sigma_reg) + sigma_reg

# 2. Safe division
inv_sigma_eq = jnp.where(sigma_eq > 0.0, 1.0 / sigma_eq, 0.0)

# 3. McCauley bracket
bracket = 0.5 * (x + jnp.abs(x))   # = max(x, 0)

# 4. Clipping (for damage)
D_new = jnp.clip(D_new, 0.0, params.Dc)
```

---

## Variable Naming Conventions

| Variable          | Meaning                             | State Key       |
|-------------------|-------------------------------------|-----------------|
| `sig_trial`       | Trial stress (elastic predictor)    | —               |
| `fy_trial`        | Yield function at trial state       | —               |
| `lambda_dot_old`  | Viscoplastic rate from prev step    | `"lambda_dot"`  |
| `eps_p_dot_old`   | Plastic strain rate from prev step  | `"eps_p_dot"`   |
| `R_dot_old`       | Isotropic hardening rate prev step  | `"R_dot"`       |
| `alpha_dot_old`   | Backstress rate from prev step      | `"alpha1_dot"`  |
| `dlambda`         | Viscoplastic increment this step    | `"dlambda"`     |
| `lambda_dot_new`  | Rate re-evaluated at new state      | stored back     |
| `eps_p_dot_new`   | Plastic strain rate at new state    | stored back     |

---

## Input Specification Format

Request users to provide equations in this structured format:

1. **Strain decomposition**
2. **Stress-strain law**
3. **Yield function**
4. **Flow rule**
5. **Viscoplastic multiplier**
6. **Isotropic hardening** (if any)
7. **Kinematic hardening** (if any)
8. **Damage evolution** (if any)

---

## Quality Checklist

Before delivering the code, verify:

- [ ] Corrected two-phase sequence used in `_plastic_update`
- [ ] Phase 1: state advanced using OLD rates from `state[...]`
- [ ] Phase 2: ALL rates re-evaluated at NEW state and returned
- [ ] All rate variables declared in `internal_state_variables`
- [ ] All rate variables read from state at top of `constitutive_update`
- [ ] All rate variables updated in state dictionary at end
- [ ] JAX compatibility (jnp, jax.lax.cond, no Python if/else)
- [ ] `self.elastic_model.C` used for stiffness tensor
- [ ] Numerical stability (regularized sqrt, safe division, clipping)
- [ ] `@tangent_AD` decorator present
- [ ] Both branches return identical tuple structure

---

## Resources

1. **PRIMARY: `references/code-template.md`** — Complete annotated template. Always follow exactly.
2. **SECONDARY: `references/equation-mapping.md`** — Maps constitutive equations to code patterns.
3. **SUPPORTING: `references/example-models.md`** — Full working examples for reference.
