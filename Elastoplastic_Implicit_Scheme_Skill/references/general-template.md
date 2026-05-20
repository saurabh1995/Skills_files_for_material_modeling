This file provides a fully annotated template for implementing implicit integration schemes
for elastoplastic constitutive models with any yield surface.

**Read `von-mises-template.md` first** — this file extends that exact structure.
The code pattern (operand tuple, elastic_update/plastic_update, Newton inside plastic branch)
is identical. Only the yield function and flow direction change.

---

## Full Implementation

```python
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from dolfinx_materials.material.jax import JAXMaterial, JAXNewton, tangent_AD


@dataclass
class GeneralPlasticityParams:
    """Material parameters for general elastoplasticity — parameters only, no logic."""
    E:       float   # Young's modulus (MPa)
    nu:      float   # Poisson's ratio (-)
    sigma_0: float   # Initial yield stress / cohesion (MPa)
    # Add yield/hardening parameters as needed


class GeneralIsotropicHardening(JAXMaterial):
    """
    General yield surface with isotropic hardening.
    Uses the same operand/lax.cond structure as von-mises-template.md.
    The only parts that change for a different yield surface are:
      - _equivalent_stress(sig)  — defines the surface shape
      - yield_stress(alpha)      — defines R(p)
      - n_tr computation         — flow direction ∂f/∂σ
    """

    def __init__(self, elastic_model, params: GeneralPlasticityParams):
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
        return {
            "p":     1,   # equivalent plastic strain (scalar)
            "dlam":  1,   # plastic multiplier (diagnostic)
            "eps_p": 6,   # plastic strain tensor (Voigt)
            "eps_e": 6,   # elastic strain tensor (Voigt)
            "fy":    1,   # trial yield function value (diagnostic)
        }

    # ── hardening law ─────────────────────────────────────────────────────────

    def yield_stress(self, alpha):
        """R(p) — replace with the appropriate hardening law."""
        return self.params.sigma_0   # perfect plasticity placeholder

    # ── helpers ───────────────────────────────────────────────────────────────

    def _shear_modulus(self):
        """μ = E / (2(1+ν)). ALWAYS use this method — never compute μ inline."""
        return self.params.E / (2.0 * (1.0 + self.params.nu))

    def _deviatoric(self, sig):
        """Deviatoric part of Voigt 6-vector."""
        mean = (sig[0] + sig[1] + sig[2]) / 3.0
        return jnp.array(
            [sig[0]-mean, sig[1]-mean, sig[2]-mean, sig[3], sig[4], sig[5]],
            dtype=sig.dtype,
        )

    def _equivalent_stress(self, sig):
        """
        Equivalent stress defining the yield surface shape.
        Replace this method to change the yield surface.

        Von Mises (default):  σ_eq = √(3/2 s:s)
        Drucker-Prager:       σ_eq = √J₂ + α·I₁   (pressure-dependent)
        """
        s = self._deviatoric(sig)
        s_dot_s = (
            s[0]**2 + s[1]**2 + s[2]**2
            + 2.0 * (s[3]**2 + s[4]**2 + s[5]**2)
        )
        return jnp.sqrt(1.5 * s_dot_s)

    # ── constitutive update ───────────────────────────────────────────────────

    @tangent_AD
    def constitutive_update(self, eps, state, dt):

        # ── 1. unpack ALL state variables ─────────────────────────────────────
        eps_old   = state["Strain"]
        sig_old   = state["Stress"]
        eps_p_old = state["eps_p"]      # tensor (6,) — no [0]
        eps_e_old = state["eps_e"]      # tensor (6,) — no [0]
        p_old     = state["p"][0]       # scalar — ALWAYS [0]

        # ── 2. elastic predictor ──────────────────────────────────────────────
        C      = self.elastic_model.C   # ← ONLY valid source
        mu     = self._shear_modulus()  # ← ONLY valid source of μ
        deps   = eps - eps_old
        sig_tr = sig_old + C @ deps

        # ── 3. yield check ────────────────────────────────────────────────────
        sigma_eq_tr     = self._equivalent_stress(sig_tr)
        sigma_y_old     = self.yield_stress(p_old)
        yield_criterion = sigma_eq_tr - sigma_y_old

        # ── 4. flow direction at trial state (semi-implicit) ──────────────────
        # For von Mises: n_tr = s_tr / σ_eq_tr  (deviatoric → tr(n)=0 → C:n = 2μn)
        # For Drucker-Prager: n_tr has volumetric part → use full C @ (deps - dlam*n_tr)
        s_tr = self._deviatoric(sig_tr)
        n_tr = s_tr / jnp.clip(sigma_eq_tr, a_min=1e-8)

        # ── 5. operand ────────────────────────────────────────────────────────
        operand = (eps, eps_old, p_old, sig_old, sig_tr, eps_p_old, yield_criterion)

        # ── 6. elastic branch ─────────────────────────────────────────────────
        def elastic_update(operand):
            eps, eps_old, p_old, sig_old, sig_tr, eps_p_old, yield_criterion = operand
            sig_new   = sig_tr
            eps_p_new = eps_p_old
            eps_e_new = eps_old + deps
            p_new     = p_old
            dlam      = 0.0             # Python float — NOT jnp.array([0.0])
            return sig_new, p_new, eps_e_new, eps_p_new, dlam, yield_criterion

        # ── 7. plastic branch ─────────────────────────────────────────────────
        def plastic_update(operand):
            eps, eps_old, p_old, sig_old, sig_tr, eps_p_old, yield_criterion = operand

            def R_plastic(dlam):
                # Step A3: stress in terms of Δλ
                sig_new = sig_tr - 3.0 * mu * dlam * n_tr
                # Step A4: hardening update
                sigma_eq_new = self._equivalent_stress(sig_new)
                sigma_y_new  = self.yield_stress(p_old + dlam)
                # Step A5: consistency residual
                return sigma_eq_new - sigma_y_new

            newton = JAXNewton()
            newton.set_residual(R_plastic)
            dlam, _ = newton.solve(0.0)

            # Recompute fresh with converged dlam (never reuse values from R_plastic)
            sig_new   = sig_old + C @ (deps - 1.5 * dlam * n_tr)
            eps_p_new = eps_p_old + 1.5 * n_tr * dlam
            eps_e_new = eps - eps_p_new
            p_new     = p_old + dlam

            return sig_new, p_new, eps_e_new, eps_p_new, dlam, yield_criterion

        # ── 8. branch ─────────────────────────────────────────────────────────
        is_plastic = yield_criterion >= 0.0
        sig_new, p_new, eps_e_new, eps_p_new, dlam, fy = jax.lax.cond(
            is_plastic, plastic_update, elastic_update, operand
        )

        # ── 9. write ALL state variables back ─────────────────────────────────
        state["Strain"] = eps
        state["Stress"] = sig_new
        state["eps_p"]  = eps_p_new
        state["eps_e"]  = eps_e_new
        state["p"]      = jnp.array([p_new])
        state["dlam"]   = jnp.array([dlam])
        state["fy"]     = jnp.array([fy])

        return sig_new, state
```

---

## Adapting for a different yield surface

**Only these three things change** between von Mises and any other surface:

### 1. `_equivalent_stress(sig)` — surface shape

```python
# Drucker-Prager: f = √J₂ + α·I₁ - k
def _equivalent_stress(self, sig):
    s     = self._deviatoric(sig)
    J2    = jnp.sqrt(0.5 * (s[0]**2 + s[1]**2 + s[2]**2 + 2*(s[3]**2+s[4]**2+s[5]**2)))
    I1    = sig[0] + sig[1] + sig[2]
    return J2 + self.params.alpha * I1
```

### 2. `n_tr` — flow direction `∂f/∂σ`

```python
# Drucker-Prager — has volumetric part (tr(n) ≠ 0):
# ∂f/∂σ = ∂(√J₂)/∂σ + α·∂I₁/∂σ
#        = s/(2√J₂)·(3/σ_eq) + α·[1,1,1,0,0,0]
# In practice: use JAX autograd to avoid manual computation:
n_tr = jax.grad(lambda s: self._equivalent_stress(s))(sig_tr)

# NOTE: for Drucker-Prager, tr(n_tr) ≠ 0, so C:n_tr ≠ 2μ·n_tr.
# The stress correction inside R_plastic must be: sig_tr - dlam * (C @ n_tr)
# NOT: sig_tr - 3*mu*dlam*n_tr
```

### 3. `R_plastic` — when flow direction has volumetric part

```python
# Drucker-Prager R_plastic:
def R_plastic(dlam):
    sig_new      = sig_tr - dlam * (C @ n_tr)          # full C needed
    sigma_eq_new = self._equivalent_stress(sig_new)
    sigma_y_new  = self.yield_stress(p_old + dlam)
    return sigma_eq_new - sigma_y_new
```

> **Warning:** The `3μΔλ` shortcut ONLY works when `tr(n_tr) = 0` (von Mises, kinematic
> backstress models). For any pressure-dependent surface (Drucker-Prager, Mohr-Coulomb),
> use `dlam * (C @ n_tr)` in the stress correction.