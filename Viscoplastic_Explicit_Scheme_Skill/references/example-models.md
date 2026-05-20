# Example Material Models

Complete implementations of classic constitutive models using the **corrected two-phase explicit integration scheme**.

> **All examples follow the corrected sequence:**
> - Phase 1: advance state using OLD rates from previous step
> - Phase 2: re-evaluate ALL rates at the new state for storage

---

## Example 1: von Mises Plasticity with Linear Isotropic Hardening

**Equations:**
- Yield: f = J₂(σ) - (σy + R)
- Viscoplastic: λ̇ = 〈f/η〉ⁿ
- Flow: ε̇p = λ̇ (3/2) σ'/J₂
- Hardening: Ṙ = H λ̇

**State variables:**
```python
return {
    "eps_p": 6, "eps_e": 6, "R": 1,
    "lambda_dot": 1, "dlambda": 1,
    "eps_p_dot": 6, "R_dot": 1,
    "fy": 1, "is_plastic": 1,
}
```

**Key implementation in `_plastic_update`:**
```python
def _plastic_update(operand):
    (...) = operand

    # Phase 1: advance state from old rates
    dlambda      = lambda_dot_old * dt
    delta_eps_p  = eps_p_dot_old * dt
    eps_p_new    = eps_p_old + delta_eps_p
    eps_e_new    = eps_e_old + (deps - delta_eps_p)
    R_new        = R_old + R_dot_old * dt
    sig_new      = C @ (eps - eps_p_new)

    # Phase 2: recompute rates at new state
    sig_dev_new    = self._deviatoric(sig_new)
    sigma_eq_new   = self._equivalent_norm(sig_dev_new)
    fy_new         = sigma_eq_new - (params.sigma_y + R_new)

    x              = fy_new / params.eta
    bracket        = 0.5 * (x + jnp.abs(x))
    lambda_dot_new = jnp.power(bracket, params.n_visc)

    inv_sigma_eq   = jnp.where(sigma_eq_new > 0.0, 1.0/sigma_eq_new, 0.0)
    flow_dir       = sig_dev_new * inv_sigma_eq
    eps_p_dot_new  = lambda_dot_new * 1.5 * flow_dir
    R_dot_new      = params.H * lambda_dot_new

    return (sig_new, eps_p_new, eps_e_new, R_new,
            lambda_dot_new, dlambda, eps_p_dot_new, R_dot_new,
            fy_new, jnp.array(1.0, dtype=eps.dtype))
```

---

## Example 2: Perzyna Viscoplasticity with Saturation Hardening

**Equations:**
- Yield: f = J₂(σ) - (σy + R)
- Rate: λ̇ = 〈f/η〉ⁿ
- Flow: ε̇p = (3/2) λ̇ σ'/J₂
- Hardening: Ṙ = b(R₁ - R) λ̇

**Key implementation — only Phase 2 differs from Example 1:**
```python
    # Phase 2: saturation hardening rate
    R_dot_new = params.b * (params.R1 - R_new) * lambda_dot_new
```

---

## Example 3: Single-Backstress Armstrong-Frederick

**Equations:**
- Yield: f = J₂(σ - X) - (σy + R)
- Rate: λ̇ = 〈f/η〉ⁿ
- Flow: ε̇p = (3/2) λ̇ (σ' - X') / J₂(σ-X)
- Isotropic: Ṙ = b(R₁ - R) λ̇
- Kinematic: Ẋ = (2/3) a ε̇p - c X λ̇

**State variables:**
```python
return {
    "eps_p": 6, "eps_e": 6, "R": 1, "alpha1": 6,
    "lambda_dot": 1, "dlambda": 1,
    "eps_p_dot": 6, "R_dot": 1, "alpha1_dot": 6,
    "fy": 1, "is_plastic": 1,
}
```

**Key implementation in `_plastic_update`:**
```python
def _plastic_update(operand):
    (...) = operand

    # Phase 1: advance all state variables from old rates
    dlambda       = lambda_dot_old * dt
    delta_eps_p   = eps_p_dot_old * dt
    eps_p_new     = eps_p_old + delta_eps_p
    eps_e_new     = eps_e_old + (deps - delta_eps_p)
    R_new         = R_old + R_dot_old * dt
    alpha1_new    = alpha1_old + alpha1_dot_old * dt
    sig_new       = C @ (eps - eps_p_new)

    # Phase 2: re-evaluate all rates at new state
    sig_eff_new   = sig_new - alpha1_new
    n_new         = self._deviatoric(sig_eff_new)
    sigma_eq_new  = self._equivalent_norm(n_new)
    fy_new        = sigma_eq_new - (params.sigma_y + R_new)

    x              = fy_new / params.eta
    bracket        = 0.5 * (x + jnp.abs(x))
    lambda_dot_new = jnp.power(bracket, params.n_visc)

    inv_sigma_eq   = jnp.where(sigma_eq_new > 0.0, 1.0/sigma_eq_new, 0.0)
    flow_dir       = n_new * inv_sigma_eq
    eps_p_dot_new  = lambda_dot_new * 1.5 * flow_dir
    R_dot_new      = params.b * (params.R1 - R_new) * lambda_dot_new
    alpha1_dot_new = (
        (2.0/3.0) * params.a * eps_p_dot_new
        - params.c * alpha1_new * lambda_dot_new
    )

    return (sig_new, eps_p_new, eps_e_new, R_new, alpha1_new,
            lambda_dot_new, dlambda, eps_p_dot_new, R_dot_new, alpha1_dot_new,
            fy_new, jnp.array(1.0, dtype=eps.dtype))
```

---

## Example 4: Two-Backstress Viscoplastic (the reference corrected model)

**Equations:**
- Yield: f = J₂(σ - α1 - α2) - (σy + R)
- Rate: λ̇ = 〈f/η〉ⁿ
- Flow: ε̇p = (3/2) λ̇ n / J₂(n)  where n = dev(σ) - (α1 + α2)
- Isotropic: Ṙ = H λ̇
- Kinematic: α̇ᵢ = (2/3) Cᵢ ε̇p - γᵢ αᵢ λ̇,  i=1,2

**State variables:**
```python
return {
    "eps_p": 6, "eps_e": 6, "R": 1, "alpha1": 6, "alpha2": 6,
    "lambda_dot": 1, "dlambda": 1,
    "eps_p_dot": 6, "R_dot": 1, "alpha1_dot": 6, "alpha2_dot": 6,
    "fy": 1, "is_plastic": 1,
}
```

**Full `_plastic_update`:**
```python
def _plastic_update(operand):
    (
        eps, eps_old, deps,
        eps_p_old, eps_e_old, R_old, alpha1_old, alpha2_old,
        lambda_dot_old, eps_p_dot_old, R_dot_old,
        alpha1_dot_old, alpha2_dot_old,
        sig_trial, n_trial, sigma_eq_trial, fy_trial, dt,
    ) = operand

    # ── Phase 1: advance state using old rates ───────────────────────────────
    dlambda      = lambda_dot_old * dt
    delta_eps_p  = eps_p_dot_old * dt
    eps_p_new    = eps_p_old + delta_eps_p
    eps_e_new    = eps_e_old + (deps - delta_eps_p)
    R_new        = R_old + R_dot_old * dt
    alpha1_new   = alpha1_old + alpha1_dot_old * dt
    alpha2_new   = alpha2_old + alpha2_dot_old * dt
    sig_new      = C @ (eps - eps_p_new)

    # ── Phase 2: re-evaluate rates at new state ──────────────────────────────
    sig_dev_new   = self._deviatoric(sig_new)
    n_new         = sig_dev_new - (alpha1_new + alpha2_new)
    sigma_eq_new  = self._equivalent_norm(n_new)
    fy_new        = sigma_eq_new - (params.sigma_y + R_new)

    x              = fy_new / params.eta
    bracket        = 0.5 * (x + jnp.abs(x))
    lambda_dot_new = jnp.power(bracket, params.n_visc)

    inv_sigma_eq   = jnp.where(sigma_eq_new > 0.0, 1.0/sigma_eq_new, 0.0)
    flow_dir       = n_new * inv_sigma_eq
    eps_p_dot_new  = lambda_dot_new * 1.5 * flow_dir

    R_dot_new      = params.H * lambda_dot_new
    alpha1_dot_new = (
        (2.0/3.0) * params.C1 * eps_p_dot_new
        - params.gamma1 * alpha1_new * lambda_dot_new
    )
    alpha2_dot_new = (
        (2.0/3.0) * params.C2 * eps_p_dot_new
        - params.gamma2 * alpha2_new * lambda_dot_new
    )

    return (
        sig_new, eps_p_new, eps_e_new, R_new, alpha1_new, alpha2_new,
        lambda_dot_new, dlambda, eps_p_dot_new, R_dot_new,
        alpha1_dot_new, alpha2_dot_new,
        fy_new, jnp.array(1.0, dtype=eps.dtype),
    )
```

---

## Example 5: Lemaitre Damage with Chaboche Kinematic Hardening

**Equations:**
- Yield: f = J₂(σ - X)/(1-D) - (σy + R)
- Rate: λ̇ = 〈f/η〉ⁿ
- Flow: ε̇p = (3/2) λ̇ (σ'-X') / J₂(σ-X)
- Isotropic: Ṙ = b(R₁-R) λ̇
- Kinematic: Ẋ = (2/3) a ε̇p - c X λ̇
- Damage: Ḋ = [Dc/(εR-εD)] × [(2/3)(1+ν)σ²eq + 3(1-2ν)σ²H] × λ̇

**State variables:**
```python
return {
    "eps_p": 6, "eps_e": 6, "R": 1, "X": 6, "D": 1,
    "lambda_dot": 1, "dlambda": 1,
    "eps_p_dot": 6, "R_dot": 1, "X_dot": 6, "D_dot": 1,
    "fy": 1, "is_plastic": 1,
}
```

**`_plastic_update` key points:**
```python
    # Phase 1
    dlambda      = lambda_dot_old * dt
    delta_eps_p  = eps_p_dot_old * dt
    D_new        = jnp.clip(D_old + D_dot_old * dt, 0.0, params.Dc)
    eps_p_new    = eps_p_old + delta_eps_p
    delta_eps_e  = deps - delta_eps_p
    eps_e_new    = eps_e_old + delta_eps_e
    R_new        = R_old + R_dot_old * dt
    X_new        = X_old + X_dot_old * dt
    # Stress with evolving damage
    sig_new      = sig_old + (1.0 - D_new) * (C @ delta_eps_e)

    # Phase 2
    sig_eff_new  = sig_new - X_new
    n_new        = self._deviatoric(sig_eff_new)
    J2_new       = self._equivalent_norm(n_new)
    sigma_eff_new = J2_new / (1.0 - D_new)
    fy_new       = sigma_eff_new - (params.sigma_y + R_new)

    x              = fy_new / params.eta
    bracket        = 0.5 * (x + jnp.abs(x))
    lambda_dot_new = jnp.power(bracket, params.n_visc)

    inv_J2         = jnp.where(J2_new > 0.0, 1.0/J2_new, 0.0)
    flow_dir       = n_new * inv_J2
    eps_p_dot_new  = lambda_dot_new * 1.5 * flow_dir
    R_dot_new      = params.b * (params.R1 - R_new) * lambda_dot_new
    X_dot_new      = (2.0/3.0)*params.a*eps_p_dot_new - params.c*X_new*lambda_dot_new

    sigma_eq_s     = self._equiv_stress(sig_new)
    sigma_H_s      = self._hydrostatic(sig_new)
    nu             = params.nu
    bracket_D      = ((2.0/3.0)*(1.0+nu)*sigma_eq_s**2
                      + 3.0*(1.0-2.0*nu)*sigma_H_s**2)
    factor_D       = params.Dc / (params.eps_R - params.eps_D + 1e-12)
    D_dot_new      = factor_D * bracket_D * lambda_dot_new
```

---

## Comparison Table

| Model                  | Backstresses | Isotropic | Damage | Example |
|------------------------|:------------:|:---------:|:------:|:-------:|
| von Mises (linear)     | 0            | Linear    | No     | 1       |
| Perzyna (saturation)   | 0            | Exp. Sat. | No     | 2       |
| Armstrong-Frederick    | 1            | Exp. Sat. | No     | 3       |
| Two-backstress         | 2            | Linear    | No     | 4       |
| Lemaitre-Chaboche      | 1            | Exp. Sat. | Yes    | 5       |

---

## What Changes Between Models

| Feature added        | Phase 1 change             | Phase 2 change                  | New state vars       |
|----------------------|----------------------------|---------------------------------|----------------------|
| Saturation hardening | `R_new = R_old + R_dot*dt` | `R_dot = b*(R1-R_new)*λ_dot`   | `R_dot`              |
| Kinematic (1 back)   | `α_new = α_old + α_dot*dt` | `α_dot = (2/3)C*ε_p_dot - c*α*λ_dot` | `alpha1`, `alpha1_dot` |
| Second backstress    | Add `alpha2_new = ...`     | Add `alpha2_dot_new = ...`      | `alpha2`, `alpha2_dot` |
| Damage               | `D_new = D_old + D_dot*dt` | `D_dot = factor*bracket_D*λ_dot` | `D`, `D_dot`         |
