# Complete Code Template for Explicit Integration Scheme

This file provides a fully annotated template for implementing explicit integration schemes for viscoplastic constitutive models.

---

## ⚠️ MANDATORY: The Corrected Two-Phase Explicit Sequence

The plastic update branch MUST follow this exact two-phase structure. Any deviation produces physically wrong results.

### Phase 1 — Advance state using OLD rates (already stored in `state`)
At each step `i`, the state is advanced using rates that were computed and stored at the **end of step `i-1`**:

```
dlambda       = lambda_dot_old * dt
delta_eps_p   = eps_p_dot_old  * dt
eps_p_new     = eps_p_old + delta_eps_p
eps_e_new     = eps_e_old + (deps - delta_eps_p)
R_new         = R_old + R_dot_old * dt
alpha_new     = alpha_old + alpha_dot_old * dt
sig_new       = C @ (eps - eps_p_new)     ← computed from updated εp
```

### Phase 2 — Re-evaluate ALL rates at the NEW state
After advancing, recompute the yield function and all rates at `(sig_new, R_new, alpha_new)`:

```
n_new          = dev(sig_new) - alpha_new
sigma_eq_new   = sqrt(3/2 n_new:n_new)
f_new          = sigma_eq_new - (sigma_y + R_new)
lambda_dot_new = <f_new / eta>^n
eps_p_dot_new  = lambda_dot_new * (3/2) * n_new / sigma_eq_new
R_dot_new      = H * lambda_dot_new
alpha_dot_new  = (2/3)*C1*eps_p_dot_new - gamma*alpha_new*lambda_dot_new
```

These rates are stored in `state` and become the "old rates" for the NEXT step.

### WRONG pattern — never do this
```python
# WRONG: computing Δλ from fy_trial (trial yield function)
overstress  = fy_trial / params.eta
bracket     = 0.5 * (overstress + jnp.abs(overstress))
dlambda     = dt * jnp.power(bracket, params.n_visc)   # ← WRONG
delta_eps_p = 1.5 * dlambda * flow_dir_from_trial       # ← WRONG
```

### CORRECT pattern
```python
# CORRECT: advance using old rate, then re-evaluate
dlambda       = lambda_dot_old * dt                     # ← Phase 1
delta_eps_p   = eps_p_dot_old * dt                      # ← Phase 1
eps_p_new     = eps_p_old + delta_eps_p
sig_new       = C @ (eps - eps_p_new)
# ... update R_new, alpha_new ...
# Re-evaluate f at new state
fy_new        = sigma_eq_new - (sigma_y + R_new)        # ← Phase 2
x             = fy_new / params.eta
bracket       = 0.5 * (x + jnp.abs(x))
lambda_dot_new = jnp.power(bracket, params.n_visc)      # ← Phase 2
eps_p_dot_new = lambda_dot_new * 1.5 * flow_dir_new     # ← Phase 2
```

---

## Full Implementation Template

```python
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from dolfinx_materials.material.jax import JAXMaterial, tangent_AD


@dataclass
class ModelParams:
    """
    Material parameters. Customize for the specific model equations.
    """
    # Elastic
    E: float        # Young's modulus [MPa]
    nu: float       # Poisson's ratio [-]

    # Yield
    sigma_y: float  # Initial yield stress [MPa]

    # Viscoplastic (Perzyna-type)
    eta: float      # Viscosity parameter [MPa·s^(1/n)]
    n_visc: float   # Viscosity exponent [-]

    # Isotropic hardening
    H: float        # Hardening modulus [MPa]

    # Kinematic hardening (Armstrong-Frederick)
    C1: float       # Kinematic modulus [MPa]
    gamma1: float   # Dynamic recovery [-]

    # Add further parameters as needed (C2, gamma2, Dc, eps_R, ...)


class ConstitutiveMaterial(JAXMaterial):
    """
    Explicit integration scheme for viscoplastic constitutive model.
    Uses the two-phase corrected explicit sequence:
      Phase 1: advance state from old rates
      Phase 2: re-evaluate rates at new state for next step
    """

    def __init__(self, elastic_model, params: ModelParams):
        super().__init__()
        self.elastic_model = elastic_model
        self.params = params
        self.dt = 0.0

    @property
    def gradient_names(self):
        return ("Strain",)

    @property
    def flux_names(self):
        return ("Stress",)

    @property
    def internal_state_variables(self):
        """
        CRITICAL: Must declare ALL state variables AND all rate variables.
        Rates are required so the next step can advance from them (Phase 1).
        """
        return {
            # Integrated state
            "eps_p":      6,   # Plastic strain tensor (Voigt)
            "eps_e":      6,   # Elastic strain tensor (Voigt)
            "R":          1,   # Isotropic hardening variable (scalar)
            "alpha1":     6,   # Backstress tensor (Voigt)
            # Add "alpha2": 6 for two-backstress models, etc.

            # Rate variables — MANDATORY for Phase 1 of next step
            "lambda_dot": 1,   # Viscoplastic multiplier rate
            "dlambda":    1,   # Viscoplastic multiplier increment
            "eps_p_dot":  6,   # Plastic strain rate (Voigt)
            "R_dot":      1,   # Isotropic hardening rate
            "alpha1_dot": 6,   # Backstress rate (Voigt)
            # Add "alpha2_dot": 6 for two-backstress models, etc.

            # Diagnostics
            "fy":         1,   # Yield function value
            "is_plastic": 1,   # Plastic loading flag
        }

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _deviatoric(self, sig):
        """Deviatoric part: sig - (1/3) tr(sig) I  (Voigt notation)"""
        p = (sig[0] + sig[1] + sig[2]) / 3.0
        return jnp.array([
            sig[0] - p, sig[1] - p, sig[2] - p,
            sig[3], sig[4], sig[5],
        ], dtype=sig.dtype)

    def _norm_voigt(self, vec):
        """Inner product in Voigt notation: v:v = v1²+v2²+v3²+2(v4²+v5²+v6²)"""
        return (
            vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]
            + 2.0*(vec[3]*vec[3] + vec[4]*vec[4] + vec[5]*vec[5])
        )

    def _equivalent_norm(self, vec):
        """
        Equivalent norm: sqrt(3/2 * vec:vec)
        Regularized to avoid sqrt(0) in gradients.
        """
        val     = 1.5 * self._norm_voigt(vec)
        val_pos = jnp.maximum(val, 0.0)
        eps_reg = 1e-16
        phys    = jnp.sqrt(val_pos)
        reg     = jnp.sqrt(val_pos + eps_reg)
        return jax.lax.stop_gradient(phys - reg) + reg

    def _hydrostatic(self, sig):
        return (sig[0] + sig[1] + sig[2]) / 3.0

    def _equiv_stress(self, sig):
        return self._equivalent_norm(self._deviatoric(sig))

    # ========================================================================
    # MAIN CONSTITUTIVE UPDATE
    # ========================================================================

    @tangent_AD
    def constitutive_update(self, eps, state, dt):
        """
        Explicit integration for one time step.

        Algorithm:
          1. Extract old state AND old rates
          2. Elastic predictor (trial stress from old εp)
          3. Yield check on trial state
          4. Elastic or plastic branch (jax.lax.cond)
          5. Update state dictionary (state + rates)
        """
        params = self.params
        C = self.elastic_model.C   # Always use this — never recompute

        # ====================================================================
        # 1. EXTRACT OLD STATE AND OLD RATES
        # ====================================================================
        eps_old        = state["Strain"]
        deps           = eps - eps_old          # total strain increment

        eps_p_old      = state["eps_p"]
        eps_e_old      = state["eps_e"]
        R_old          = state["R"][0]
        alpha1_old     = state["alpha1"]
        # Add: alpha2_old = state["alpha2"]  etc. for multi-backstress

        # OLD RATES — used in Phase 1 of plastic update
        lambda_dot_old  = state["lambda_dot"][0]
        eps_p_dot_old   = state["eps_p_dot"]
        R_dot_old       = state["R_dot"][0]
        alpha1_dot_old  = state["alpha1_dot"]
        # Add: alpha2_dot_old = state["alpha2_dot"]  etc.

        # ====================================================================
        # 2. ELASTIC PREDICTOR (trial stress using OLD plastic strain)
        # ====================================================================
        sig_trial = C @ (eps - eps_p_old)
        # For damaged models: sig_trial = sig_old + (1.0 - D_old)*(C @ deps)

        # ====================================================================
        # 3. YIELD CHECK on trial state
        # ====================================================================
        sig_trial_dev   = self._deviatoric(sig_trial)
        n_trial         = sig_trial_dev - alpha1_old   # subtract all backstresses
        sigma_eq_trial  = self._equivalent_norm(n_trial)
        fy_trial        = sigma_eq_trial - (params.sigma_y + R_old)

        # ====================================================================
        # 4. PACK OPERAND
        # ====================================================================
        operand = (
            eps, eps_old, deps,
            eps_p_old, eps_e_old, R_old, alpha1_old,
            lambda_dot_old, eps_p_dot_old, R_dot_old, alpha1_dot_old,
            sig_trial, n_trial, sigma_eq_trial, fy_trial,
            dt,
        )

        # ====================================================================
        # 5. ELASTIC BRANCH
        # ====================================================================
        def _elastic_update(operand):
            (
                eps, eps_old, deps,
                eps_p_old, eps_e_old, R_old, alpha1_old,
                lambda_dot_old, eps_p_dot_old, R_dot_old, alpha1_dot_old,
                sig_trial, n_trial, sigma_eq_trial, fy_trial,
                dt,
            ) = operand

            sig_new        = sig_trial
            eps_p_new      = eps_p_old
            eps_e_new      = eps - eps_p_new
            R_new          = R_old
            alpha1_new     = alpha1_old

            # All rates are zero — no evolution in elastic zone
            lambda_dot_new  = jnp.array(0.0, dtype=eps.dtype)
            dlambda         = jnp.array(0.0, dtype=eps.dtype)
            eps_p_dot_new   = jnp.zeros(6, dtype=eps.dtype)
            R_dot_new       = jnp.array(0.0, dtype=eps.dtype)
            alpha1_dot_new  = jnp.zeros(6, dtype=eps.dtype)

            is_plastic_out  = jnp.array(0.0, dtype=eps.dtype)
            fy_out          = fy_trial

            return (
                sig_new, eps_p_new, eps_e_new, R_new, alpha1_new,
                lambda_dot_new, dlambda, eps_p_dot_new, R_dot_new, alpha1_dot_new,
                fy_out, is_plastic_out,
            )

        # ====================================================================
        # 6. PLASTIC BRANCH — CORRECTED TWO-PHASE SEQUENCE
        # ====================================================================
        def _plastic_update(operand):
            (
                eps, eps_old, deps,
                eps_p_old, eps_e_old, R_old, alpha1_old,
                lambda_dot_old, eps_p_dot_old, R_dot_old, alpha1_dot_old,
                sig_trial, n_trial, sigma_eq_trial, fy_trial,
                dt,
            ) = operand

            # ── PHASE 1: Advance state using OLD rates ──────────────────────
            #
            # These rates were computed at the END of the previous step.
            # Forward-Euler: Q(t+dt) = Q(t) + Q_dot(t) * dt

            dlambda      = lambda_dot_old * dt
            delta_eps_p  = eps_p_dot_old * dt

            eps_p_new    = eps_p_old + delta_eps_p
            eps_e_new    = eps_e_old + (deps - delta_eps_p)
            R_new        = R_old + R_dot_old * dt
            alpha1_new   = alpha1_old + alpha1_dot_old * dt
            # For two-backstress: alpha2_new = alpha2_old + alpha2_dot_old * dt

            # Recompute stress from the updated plastic strain
            sig_new      = C @ (eps - eps_p_new)
            # For damaged models:
            # D_new = D_old + D_dot_old * dt
            # sig_new = sig_old + (1.0 - D_new) * (C @ (deps - delta_eps_p))

            # ── PHASE 2: Re-evaluate ALL rates at the NEW state ─────────────
            #
            # These become the "old rates" consumed by the NEXT step's Phase 1.

            sig_new_dev    = self._deviatoric(sig_new)
            n_new          = sig_new_dev - alpha1_new   # subtract all backstresses
            sigma_eq_new   = self._equivalent_norm(n_new)
            fy_new         = sigma_eq_new - (params.sigma_y + R_new)

            x              = fy_new / params.eta
            bracket        = 0.5 * (x + jnp.abs(x))   # McCauley bracket
            lambda_dot_new = jnp.power(bracket, params.n_visc)

            inv_sigma_eq   = jnp.where(sigma_eq_new > 0.0, 1.0/sigma_eq_new, 0.0)
            flow_dir       = n_new * inv_sigma_eq
            eps_p_dot_new  = lambda_dot_new * 1.5 * flow_dir

            R_dot_new      = params.H * lambda_dot_new

            alpha1_dot_new = (
                (2.0/3.0) * params.C1 * eps_p_dot_new
                - params.gamma1 * alpha1_new * lambda_dot_new
            )
            # For two-backstress:
            # alpha2_dot_new = (2/3)*C2*eps_p_dot_new - gamma2*alpha2_new*lambda_dot_new

            # Damage rate (if applicable):
            # sigma_eq_s  = self._equiv_stress(sig_new)
            # sigma_H_s   = self._hydrostatic(sig_new)
            # bracket_D   = (2/3)*(1+nu)*sigma_eq_s**2 + 3*(1-2*nu)*sigma_H_s**2
            # D_dot_new   = (Dc/(eps_R - eps_D + 1e-12)) * bracket_D * lambda_dot_new

            is_plastic_out = jnp.array(1.0, dtype=eps.dtype)

            return (
                sig_new, eps_p_new, eps_e_new, R_new, alpha1_new,
                lambda_dot_new, dlambda, eps_p_dot_new, R_dot_new, alpha1_dot_new,
                fy_new, is_plastic_out,
            )

        # ====================================================================
        # 7. CONDITIONAL BRANCHING
        # ====================================================================
        (
            sig_new, eps_p_new, eps_e_new, R_new, alpha1_new,
            lambda_dot_new, dlambda, eps_p_dot_new, R_dot_new, alpha1_dot_new,
            fy_out, is_plastic_out,
        ) = jax.lax.cond(
            fy_trial > 0.0,
            _plastic_update,
            _elastic_update,
            operand,
        )

        # ====================================================================
        # 8. UPDATE STATE — integrated variables AND rates
        # ====================================================================
        state["Strain"]      = eps
        state["Stress"]      = sig_new
        state["eps_p"]       = eps_p_new
        state["eps_e"]       = eps_e_new
        state["R"]           = jnp.array([R_new])
        state["alpha1"]      = alpha1_new

        # Rates (consumed by Phase 1 of the next step)
        state["lambda_dot"]  = jnp.array([lambda_dot_new])
        state["dlambda"]     = jnp.array([dlambda])
        state["eps_p_dot"]   = eps_p_dot_new
        state["R_dot"]       = jnp.array([R_dot_new])
        state["alpha1_dot"]  = alpha1_dot_new

        # Diagnostics
        state["fy"]          = jnp.array([fy_out])
        state["is_plastic"]  = jnp.array([is_plastic_out])

        return sig_new, state
```

---

## Key Points

1. **Two-phase plastic update is mandatory**. Phase 1 advances state with old rates. Phase 2 recomputes rates at the new state. Never compute `dlambda` from `fy_trial`.

2. **Rate variables must be in state**. Every `Q_dot` used in Phase 1 must be stored in `internal_state_variables` and updated every step.

3. **Both branches have identical return signature**. `jax.lax.cond` requires this.

4. **Update ALL state variables every step**. Even unchanged variables must be assigned.

5. **Scalar wrapping**. Scalars stored as length-1 arrays: `jnp.array([value])`.

6. **Elastic stiffness**. Always `self.elastic_model.C`, never recomputed.

7. **Numerical stability**. Use regularized `_equivalent_norm`, safe division with `jnp.where`, clipping for physical bounds.

8. **Pure inner functions**. No side effects; all data passed via `operand`.
