# Complete von Mises Template

von Mises elastoplasticity with isotropic hardening — implicit return mapping (backward Euler).

---

## Mandatory Code Structure

Every generated model **must** follow this structure exactly. Read the annotated template
below before writing a single line of code.

### Structure checklist (in order)

```
@dataclass  MyModel_Params           ← parameters only, no logic
class       MyModel(JAXMaterial)
  __init__(elastic_model, params)
  internal_state_variables           ← {"p":1, "dlam":1, "eps_p":6, "eps_e":6, "fy":1}
  yield_stress(alpha)                ← hardening law R(p) as a method on self
  _shear_modulus()                   ← μ = E/(2(1+ν)), always a method, never inline
  _deviatoric(sig)
  _equivalent_stress(sig)
  @tangent_AD constitutive_update:
    1. unpack ALL state variables
    2. C = self.elastic_model.C
    3. sig_tr = sig_old + C @ deps    ← elastic predictor
    4. n_tr = s_tr / clip(sigma_eq_tr)
    5. operand = (all vars needed by both branches)
    6. def elastic_update(operand)    ← returns full tuple, dlam=0.0 as Python float
    7. def plastic_update(operand):
         def R_plastic(dlam): ...     ← residual only, no lax.cond inside
         newton = JAXNewton()
         newton.set_residual(R_plastic)
         dlam, _ = newton.solve(0.0)
         [recompute all state with converged dlam]
    8. jax.lax.cond(yield_criterion >= 0.0, plastic_update, elastic_update, operand)
    9. write ALL state variables back to state dict
```

---

## Full Implementation

```python
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from dolfinx_materials.material.jax import JAXMaterial, JAXNewton, tangent_AD


@dataclass
class MyModelParams:
    """
    Material parameters — parameters only, NO logic here.

    Model equations
    ---------------
    Additive split:   ε = εe + εp
    Elastic law:      σ = C : εe = C : (ε - εp)
    Yield function:   f = σ_eq - R(p)   where σ_eq = √(3/2 s:s)
    Flow rule:        Δεp = Δλ · (3/2) · n_tr   where n_tr = s_tr / σ_eq_tr
    Hardening:        p_{n+1} = p_n + Δλ
    Consistency:      f_{n+1} = σ_eq,tr - 3μΔλ - R(p_n + Δλ) = 0
    """
    E:       float   # Young's modulus (MPa)
    nu:      float   # Poisson's ratio (-)
    sigma_0: float   # Initial yield stress (MPa)
    # Add hardening parameters here, e.g.:
    # Q: float      # Voce saturation stress (MPa)
    # b: float      # Voce saturation rate (-)
    # H: float      # Linear hardening modulus (MPa)


class MyModel(JAXMaterial):
    """
    Von Mises elastoplasticity with isotropic hardening.
    Implicit return mapping — Newton solver for Δλ.
    """

    def __init__(self, elastic_model, params: MyModelParams):
        super().__init__()
        self.elastic_model = elastic_model
        self.params = params
        self.dt = 0.0

    # ── properties ────────────────────────────────────────────────────────────

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
            "dlam":  1,   # plastic multiplier increment (diagnostic)
            "eps_p": 6,   # plastic strain tensor (Voigt)
            "eps_e": 6,   # elastic strain tensor (Voigt)
            "fy":    1,   # trial yield function value (diagnostic)
        }

    # ── hardening law ─────────────────────────────────────────────────────────

    def yield_stress(self, alpha):
        """
        R(p) — yield strength as function of equivalent plastic strain.
        Replace body with the appropriate hardening law.

        Examples (pick one, delete the others):
          Perfect plasticity:  return self.params.sigma_0
          Linear hardening:    return self.params.sigma_0 + self.params.H * alpha
          Voce saturation:     return self.params.sigma_0 + self.params.Q * (1 - exp(-b*p))
        """
        return self.params.sigma_0 + self.params.Q * (
            1.0 - jnp.exp(-self.params.b * alpha)
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _shear_modulus(self):
        """μ = E / (2(1+ν)). ALWAYS use this method — never compute μ inline."""
        return self.params.E / (2.0 * (1.0 + self.params.nu))

    def _deviatoric(self, sig):
        """Deviatoric part of Voigt 6-vector [s11,s22,s33,s12,s23,s13]."""
        mean = (sig[0] + sig[1] + sig[2]) / 3.0
        return jnp.array(
            [sig[0]-mean, sig[1]-mean, sig[2]-mean, sig[3], sig[4], sig[5]],
            dtype=sig.dtype,
        )

    def _equivalent_stress(self, sig):
        """Von Mises σ_eq = √(3/2 · s:s).  Voigt: 2× on shear terms."""
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
        eps_p_old = state["eps_p"]          # tensor, shape (6,) — no [0]
        eps_e_old = state["eps_e"]          # tensor, shape (6,) — no [0]
        p_old     = state["p"][0]           # scalar — ALWAYS [0] to unpack

        # ── 2. elastic predictor ──────────────────────────────────────────────
        C      = self.elastic_model.C       # ← ONLY valid source of stiffness
        mu     = self._shear_modulus()      # ← ONLY valid source of μ
        deps   = eps - eps_old
        sig_tr = sig_old + C @ deps

        # ── 3. yield check at trial state ─────────────────────────────────────
        s_tr            = self._deviatoric(sig_tr)
        sigma_eq_tr     = self._equivalent_stress(sig_tr)
        sigma_y_old     = self.yield_stress(p_old)
        yield_criterion = sigma_eq_tr - sigma_y_old   # > 0 → plastic

        # ── 4. flow direction (fixed at trial state — semi-implicit) ──────────
        # tr(n_tr) = 0  ⟹  C:n_tr = 2μ n_tr  ⟹  stress correction = 3μ Δλ n_tr
        n_tr = s_tr / jnp.clip(sigma_eq_tr, a_min=1e-8)

        # ── 5. operand — identical tuple passed to both branches ───────────────
        operand = (eps, eps_old, p_old, sig_old, sig_tr, eps_p_old, yield_criterion)

        # ── 6. elastic branch ─────────────────────────────────────────────────
        def elastic_update(operand):
            eps, eps_old, p_old, sig_old, sig_tr, eps_p_old, yield_criterion = operand
            sig_new   = sig_tr
            eps_p_new = eps_p_old
            eps_e_new = eps_old + deps      # full strain increment is elastic
            p_new     = p_old
            dlam      = 0.0                 # Python float — NOT jnp.array([0.0])
            return sig_new, p_new, eps_e_new, eps_p_new, dlam, yield_criterion

        # ── 7. plastic branch ─────────────────────────────────────────────────
        def plastic_update(operand):
            eps, eps_old, p_old, sig_old, sig_tr, eps_p_old, yield_criterion = operand

            # Hand derivation of R(Δλ) = 0
            # Step A3: σ_{n+1} = σ_tr - 3μ Δλ n_tr  (n_tr deviatoric → only μ appears)
            # Step A4: p_{n+1} = p_n + Δλ            (linear ODE, trivial)
            # Step A5: residual = σ_eq_{n+1} - R(p_{n+1}) = 0  (consistency)
            def R_plastic(dlam):
                sig_new      = sig_tr - 3.0 * mu * dlam * n_tr   # Step A3
                sigma_eq_new = self._equivalent_stress(sig_new)
                sigma_y_new  = self.yield_stress(p_old + dlam)    # Step A4
                return sigma_eq_new - sigma_y_new                  # Step A5 residual

            newton = JAXNewton()
            newton.set_residual(R_plastic)
            dlam, _ = newton.solve(0.0)    # initial guess always 0.0

            # Recompute final state with converged dlam
            # Do NOT reuse values from inside R_plastic — always recompute fresh
            sig_new   = sig_old + C @ (deps - 1.5 * dlam * n_tr)
            eps_p_new = eps_p_old + 1.5 * n_tr * dlam
            eps_e_new = eps - eps_p_new
            p_new     = p_old + dlam

            return sig_new, p_new, eps_e_new, eps_p_new, dlam, yield_criterion

        # ── 8. branch: plastic when yield_criterion >= 0 ──────────────────────
        is_plastic = yield_criterion >= 0.0
        sig_new, p_new, eps_e_new, eps_p_new, dlam, fy = jax.lax.cond(
            is_plastic, plastic_update, elastic_update, operand
        )

        # ── 9. write ALL state variables back ─────────────────────────────────
        state["Strain"] = eps
        state["Stress"] = sig_new
        state["eps_p"]  = eps_p_new
        state["eps_e"]  = eps_e_new
        state["p"]      = jnp.array([p_new])    # scalar → wrap in array
        state["dlam"]   = jnp.array([dlam])     # scalar → wrap in array
        state["fy"]     = jnp.array([fy])       # scalar → wrap in array

        return sig_new, state
```

---

## How to adapt for any hardening law

Only the `yield_stress(self, alpha)` method body changes. The entire rest of the class
is identical regardless of hardening law.

### Linear hardening — R(p) = σ₀ + H·p

Add `H: float` to params dataclass, then:
```python
def yield_stress(self, alpha):
    return self.params.sigma_0 + self.params.H * alpha
```

Residual derivation (Steps A3–A5):
```
Step A3: σ_eq,tr - 3μΔλ
Step A4: p_{n+1} = p_n + Δλ  →  R(p_{n+1}) = σ₀ + H(p_n + Δλ)
Step A5: σ_eq,tr - 3μΔλ - σ₀ - H(p_n + Δλ) = 0
         σ_eq,tr - σ₀ - H·p_n = (3μ + H)·Δλ
         Δλ = (σ_eq,tr - σ₀ - H·p_n) / (3μ + H)   ← closed form; Newton finds in 1 step
```

### Voce saturation hardening — R(p) = σ₀ + Q·(1 - exp(-b·p))

Add `Q: float, b: float` to params, then:
```python
def yield_stress(self, alpha):
    return self.params.sigma_0 + self.params.Q * (1.0 - jnp.exp(-self.params.b * alpha))
```

Residual derivation (Steps A3–A5):
```
Step A5: σ_eq,tr - 3μΔλ - σ₀ - Q·(1 - exp(-b·(p_n + Δλ))) = 0
         ← nonlinear in Δλ; Newton needed, typically 3–5 iterations
```

### Power law hardening — R(p) = σ₀ + K·pⁿ

Add `K: float, n_exp: float` to params, then:
```python
def yield_stress(self, alpha):
    return self.params.sigma_0 + self.params.K * jnp.power(alpha, self.params.n_exp)
```

---

## How to add a new internal tensor variable (kinematic hardening)

Adding backstress X (Armstrong-Frederick: Ẋ = a·ṗ·n - c·X·ṗ) requires 7 steps.
The implicit update derivation is: X_{n+1} = (X_n + a·Δλ·n_tr) / (1 + c·Δλ).

```python
# 1. Declare in internal_state_variables
"X": 6,   # backstress tensor (Voigt)

# 2. Unpack in constitutive_update
X_old = state["X"]   # tensor — no [0]

# 3. Add to operand tuple
operand = (..., X_old, ...)

# 4. Pass through unchanged in elastic_update (and add to return tuple)
X_new = X_old

# 5. Add implicit update inside R_plastic
X_new    = (X_old + params.a * dlam * n_tr) / (1.0 + params.c * dlam)
sig_new  = sig_tr - 3.0 * mu * dlam * n_tr
xi_new   = self._deviatoric(sig_new - X_new)    # effective stress
sigma_eq_new = self._equivalent_stress(xi_new)  # ← use xi_new not sig_new

# 6. Recompute after newton.solve with same formula
X_new = (X_old + params.a * dlam * n_tr) / (1.0 + params.c * dlam)

# 7. Write back
state["X"] = X_new   # tensor — no wrapping
```

Both branches must return the same element count and shapes. Count every return value.

---

## Critical rules — never violate

| Rule | Correct | Wrong |
|------|---------|-------|
| Stiffness C | `C = self.elastic_model.C` | Manual Lamé construction inline |
| Shear modulus μ | `mu = self._shear_modulus()` | `mu = E/(2*(1+nu))` inline |
| Scalar state unpack | `p_old = state["p"][0]` | `p_old = state["p"]` |
| Scalar state write | `state["p"] = jnp.array([p_new])` | `state["p"] = p_new` |
| Tensor state unpack | `X_old = state["X"]` | `X_old = state["X"][0]` |
| `dlam` in elastic branch | `dlam = 0.0` (Python float) | `jnp.array([0.0])` |
| Newton placement | Inside `plastic_update` only | Outside `lax.cond` (runs unconditionally) |
| `R_plastic` residual | No `lax.cond` inside | Nested `lax.cond` elastic/plastic inside residual |
| Newton initial guess | `newton.solve(0.0)` | `newton.solve(1e-12)` |
| Post-Newton state | Recompute fresh with converged `dlam` | Reuse closure values from `R_plastic` |
| `lax.cond` condition | `yield_criterion >= 0.0` | `yield_criterion > 0.0` |
| `lax.cond` branch order | `plastic_update, elastic_update` | Swapped |
| Flow direction `n_tr` | Computed once outside `R_plastic` | Recomputed inside Newton loop |
| `_equivalent_stress` divisor | `jnp.clip(sigma_eq_tr, a_min=1e-8)` | Raw division |

---

## Pitfalls and fixes

| Symptom | Root cause | Fix |
|---------|-----------|-----|
| Newton doesn't converge | Residual sign wrong | Return `sigma_eq_new - sigma_y_new` |
| `dlam` always 0, no plastic flow | Newton runs outside `lax.cond` | Move Newton fully inside `plastic_update` |
| `dlam` always 0, wrong yield check | `yield_criterion > 0` misses exact yield point | Use `>= 0.0` |
| `jax.lax.cond` shape error | Return tuple count or shapes differ | Count every return element in both branches |
| `state["p"]` wrong value | Missing `[0]` on scalar unpack | `p_old = state["p"][0]` |
| μ not in scope in residual | μ computed outside method | `mu = self._shear_modulus()` at top of `constitutive_update` |
| `eps_p` grows in elastic step | Not passed through elastic branch | `eps_p_new = eps_p_old` in `elastic_update` |
| Stress wrong after convergence | Reused stale value from inside `R_plastic` | Always recompute after `newton.solve` |
| `dlam` in `state` is scalar but written as array | Write-back wraps correctly but `lax.cond` returns scalar | `jnp.array([dlam])` wrapping happens **after** `lax.cond` in write-back |

---

## Testing snippet

```python
from dolfinx_materials.material.jax import LinearElasticModel

elastic = LinearElasticModel(200000.0, 0.3)
params  = MyModelParams(E=200000.0, nu=0.3, sigma_0=250.0, Q=150.0, b=20.0)
mat     = MyModel(elastic, params)

state = {
    "Strain": jnp.zeros(6),
    "Stress": jnp.zeros(6),
    "eps_p":  jnp.zeros(6),
    "eps_e":  jnp.zeros(6),
    "p":      jnp.array([0.0]),
    "dlam":   jnp.array([0.0]),
    "fy":     jnp.array([0.0]),
}

eps = jnp.array([0.005, 0.0, 0.0, 0.0, 0.0, 0.0])  # uniaxial tension
sig, state = mat.constitutive_update(eps, state, 1.0)

print(f"Stress xx : {sig[0]:.2f} MPa")
print(f"Plastic p : {state['p'][0]:.6f}")
print(f"dlam      : {state['dlam'][0]:.6f}")
```