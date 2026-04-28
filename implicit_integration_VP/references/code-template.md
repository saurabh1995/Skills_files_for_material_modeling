# code-template.md — Implicit Viscoplastic Model Code Templates
# FEniCSx-JAX | dolfinx_materials framework

READ THIS FILE FIRST before writing any model code.
Three complete working templates cover all practical cases.
The adaptation guide at the end explains how to handle any new model.

---

## MANDATORY IMPORTS (every model)

```python
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from jax_newton_solver import JAXNewton
from dolfinx_materials.material.jax import JAXMaterial, tangent_AD
```

---

## MANDATORY CLASS STRUCTURE (every model)

Every model is always exactly TWO classes:

```
@dataclass  MyModel_Params(JAXMaterial)    ← parameters only, no logic
class       MyModel(JAXMaterial)           ← all logic, inherits JAXMaterial
```

Every model class has these components in this exact order:

```
__init__(self, elastic_model, params)
gradient_names   → ("Strain",)
flux_names       → ("Stress",)
internal_state_variables  → {name: size}   scalar=1, Voigt tensor=6

helper methods (paste verbatim from STANDARD HELPERS section below):
  _deviatoric(sig)
  _J2(sig_dev)
  _flow_direction(sig, X)     ← when backstress present
  df_dsigma(sig)               ← when no backstress (Perzyna)

@tangent_AD
constitutive_update(self, eps, state, dt):
  1. unpack state
  2. C = self.elastic_model.C          ← ONLY valid source of C
  3. deps = eps - eps_old
  4. sig_trial = sig_old + C @ deps    ← elastic predictor
  5. fy = yield_check(sig_trial, ...)  ← yield check
  6. operand = (all variables needed by both branches)
  7. def elastic_update(operand): ...
  8. def plastic_update(operand):
       def R_plastic(dlam): ...       ← A3/A4/A5/A6 inside here
       newton = JAXNewton()
       newton.set_residual(R_plastic)
       dlam, _ = newton.solve(0.0)
       [recompute all state with converged dlam]
  9. jax.lax.cond(fy > 0.0, plastic_update, elastic_update, operand)
  10. write ALL state variables back to state dict
  11. return sig_new, state
```

---

## STANDARD HELPERS — copy verbatim into every model

```python
def _deviatoric(self, sig):
    """
    Deviatoric part of Voigt 6-vector [s11, s22, s33, s12, s23, s13].
    Subtracts p = tr(sig)/3 from normal components only.
    """
    p = (sig[0] + sig[1] + sig[2]) / 3.0
    return jnp.array(
        [sig[0]-p, sig[1]-p, sig[2]-p, sig[3], sig[4], sig[5]],
        dtype=sig.dtype,
    )

def _J2(self, sig_dev):
    """
    Regularized J2 = sqrt(3/2 * s:s).
    Voigt inner product: s:s = s11²+s22²+s33² + 2*(s12²+s23²+s13²)
    stop_gradient trick: forward = physical value, gradient = smooth (no sqrt kink at 0).
    ALWAYS use this — never use raw jnp.sqrt for J2.
    """
    s = sig_dev
    ss = (s[0]*s[0] + s[1]*s[1] + s[2]*s[2]
          + 2.0*(s[3]*s[3] + s[4]*s[4] + s[5]*s[5]))
    val     = jnp.maximum(1.5 * ss, 0.0)
    J2_phys = jnp.sqrt(val)
    J2_reg  = jnp.sqrt(val + 1e-16)
    return jax.lax.stop_gradient(J2_phys - J2_reg) + J2_reg

def _flow_direction(self, sig, X):
    """
    Flow direction N = dev(sig - X) / J2(sig - X).
    Use for Lemaitre-Chaboche and any model with backstress X.
    Always evaluated at PREVIOUS STEP (sig_old, X_old) — semi-implicit.
    Key property: tr(N) = 0  =>  C:N = 2*mu*N  (stress correction = 3*mu*dlam*N).
    """
    xi     = sig - X
    xi_dev = self._deviatoric(xi)
    J2     = self._J2(xi_dev)
    return xi_dev / J2

def df_dsigma(self, sig):
    """
    Flow direction N = (3/2) * dev(sig) / J2(sig) = gradient of von Mises surface.
    Use for Perzyna-type models WITHOUT backstress.
    """
    s11, s22, s33, s12, s23, s13 = sig
    p = (s11 + s22 + s33) / 3.0
    s_dev = jnp.array([s11-p, s22-p, s33-p, s12, s23, s13])
    sigma_eq = jnp.sqrt(1.5 * (
        (s11-p)**2 + (s22-p)**2 + (s33-p)**2
        + 2.0*(s12**2 + s23**2 + s13**2)
    ))
    return (3.0 / (2.0 * sigma_eq)) * s_dev
```

---

## TEMPLATE 1 — Perzyna, No Hardening

**Equations:**
```
ε̇ᵖ = η·(f/σ_y)^N·∂f/∂σ
f   = J2(σ) - σ_y
```
**Residual (Step A6):** `R(Δλ) = Δλ - η·Δt·(f(S_{n+1})/σ_y)^N = 0`

```python
@dataclass
class Perzyna_Params(JAXMaterial):
    E:     float   # Young's modulus (MPa)
    nu:    float   # Poisson's ratio (-)
    sig_y: float   # yield stress (MPa)
    eta:   float   # fluidity / viscosity (MPa⁻¹·s⁻¹)
    N:     float   # power-law exponent (-)


class Perzyna(JAXMaterial):

    def __init__(self, elastic_model, params: Perzyna_Params):
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
            "eps_p": 6,   # plastic strain (Voigt)
            "eps_e": 6,   # elastic strain (Voigt)
            "fy":    1,   # yield function value (diagnostic)
            "dlam":  1,   # plastic multiplier (diagnostic)
        }

    # paste _deviatoric, _J2, df_dsigma from STANDARD HELPERS

    @tangent_AD
    def constitutive_update(self, eps, state, dt):

        eps_old   = state["Strain"]
        sig_old   = state["Stress"]
        eps_p_old = state["eps_p"]
        eps_e_old = state["eps_e"]

        params = self.params
        C      = self.elastic_model.C

        deps      = eps - eps_old
        sig_trial = sig_old + C @ deps                           # Step A2

        fy = self._J2(self._deviatoric(sig_trial)) - params.sig_y   # Step A2

        operand = (eps_old, sig_old, eps_p_old, eps_e_old, deps, sig_trial, fy, dt)

        def elastic_update(operand):
            (eps_old, sig_old, eps_p_old, eps_e_old, deps, sig_trial, fy, dt) = operand
            return sig_trial, eps_p_old, eps_e_old + deps, fy, 0.0

        def plastic_update(operand):
            (eps_old, sig_old, eps_p_old, eps_e_old, deps, sig_trial, fy, dt) = operand

            n = self.df_dsigma(sig_old)                          # semi-implicit direction

            def R_plastic(dlam):
                sig_new = sig_old + C @ (deps - dlam * n)        # Step A3
                J2      = self._J2(self._deviatoric(sig_new))
                f       = J2 - params.sig_y
                phi     = jnp.where(f > 0.0, (f / params.sig_y)**params.N, 0.0)
                return dlam - params.eta * dt * phi               # Step A6

            newton = JAXNewton()
            newton.set_residual(R_plastic)
            dlam, _ = newton.solve(0.0)

            sig_new   = sig_old + C @ (deps - dlam * n)
            eps_p_new = eps_p_old + dlam * n
            eps_e_new = eps_e_old + (deps - dlam * n)
            return sig_new, eps_p_new, eps_e_new, fy, dlam

        is_plastic = fy > 0.0
        sig_new, eps_p_new, eps_e_new, fy, dlam = jax.lax.cond(
            is_plastic, plastic_update, elastic_update, operand
        )

        state["Strain"] = eps
        state["Stress"] = sig_new
        state["eps_p"]  = eps_p_new
        state["eps_e"]  = eps_e_new
        state["fy"]     = jnp.array([fy])
        state["dlam"]   = jnp.array([dlam])
        return sig_new, state
```

---

## TEMPLATE 2 — Perzyna with Isotropic Hardening

**New equations over Template 1:**
```
f = J2(σ) - (σ_y + H·p)     linear isotropic hardening
ṗ = Ēdot_p                   equivalent plastic strain rate
```
**Implicit update (Step A4 — linear, trivial):** `p_{n+1} = p_n + Δλ`

```python
@dataclass
class Perzyna_IsoHard_Params(JAXMaterial):
    E:     float
    nu:    float
    sig_y: float   # initial yield stress (MPa)
    eta:   float
    N:     float
    H:     float   # isotropic hardening modulus (MPa), H>0 = hardening, H<0 = softening


class Perzyna_IsoHard(JAXMaterial):

    def __init__(self, elastic_model, params: Perzyna_IsoHard_Params):
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
            "eps_p": 6,
            "eps_e": 6,
            "fy":    1,
            "dlam":  1,
        }

    # paste _deviatoric, _J2, df_dsigma from STANDARD HELPERS

    @tangent_AD
    def constitutive_update(self, eps, state, dt):

        eps_old   = state["Strain"]
        sig_old   = state["Stress"]
        eps_p_old = state["eps_p"]
        eps_e_old = state["eps_e"]
        p_old     = state["p"][0]

        params = self.params
        C      = self.elastic_model.C

        deps      = eps - eps_old
        sig_trial = sig_old + C @ deps

        fy = self._J2(self._deviatoric(sig_trial)) - (params.sig_y + params.H * p_old)

        operand = (eps_old, sig_old, eps_p_old, eps_e_old, p_old, deps, sig_trial, fy, dt)

        def elastic_update(operand):
            (eps_old, sig_old, eps_p_old, eps_e_old,
             p_old, deps, sig_trial, fy, dt) = operand
            return sig_trial, eps_p_old, eps_e_old + deps, p_old, fy, 0.0

        def plastic_update(operand):
            (eps_old, sig_old, eps_p_old, eps_e_old,
             p_old, deps, sig_trial, fy, dt) = operand

            n = self.df_dsigma(sig_old)

            def R_plastic(dlam):
                sig_new = sig_old + C @ (deps - dlam * n)        # Step A3
                p_new   = p_old + dlam                            # Step A4: linear, trivial
                J2      = self._J2(self._deviatoric(sig_new))
                f       = J2 - (params.sig_y + params.H * p_new)
                phi     = jnp.where(f > 0.0, (f / params.sig_y)**params.N, 0.0)
                return dlam - params.eta * dt * phi               # Step A6

            newton = JAXNewton()
            newton.set_residual(R_plastic)
            dlam, _ = newton.solve(0.0)

            sig_new   = sig_old + C @ (deps - dlam * n)
            eps_p_new = eps_p_old + dlam * n
            eps_e_new = eps_e_old + (deps - dlam * n)
            p_new     = p_old + dlam
            return sig_new, eps_p_new, eps_e_new, p_new, fy, dlam

        is_plastic = fy > 0.0
        sig_new, eps_p_new, eps_e_new, p_new, fy, dlam = jax.lax.cond(
            is_plastic, plastic_update, elastic_update, operand
        )

        state["Strain"] = eps
        state["Stress"] = sig_new
        state["eps_p"]  = eps_p_new
        state["eps_e"]  = eps_e_new
        state["p"]      = jnp.array([p_new])
        state["fy"]     = jnp.array([fy])
        state["dlam"]   = jnp.array([dlam])
        return sig_new, state
```

---

## TEMPLATE 3 — Lemaitre-Chaboche (Iso + Kinematic Hardening)

**Complete governing equations:**
```
Ėp     = (3/2)·Ēdot_p·N              flow rule, N = dev(S-X)/J₂(S-X)
Ēdot_p = ⟨(J₂(S-X) - R - k)/K⟩ⁿ   overstress law (McCauley bracket)
Ẋ      = a·Ēdot_p·N - s·X·Ēdot_p    backstress (Armstrong-Frederick)
Ṙ      = b₁·(b₂-R)·Ēdot_p           isotropic hardening (Voce saturation)
S      = C:(E - Ep)
```

**Derived implicit updates (Steps A3–A6, from LC_implicit_scheme_hand_calculation.pdf):**
```
N     = dev(Sₙ - Xₙ)/J₂(Sₙ - Xₙ)                      semi-implicit (fixed)
S_t   = Ŝ - 3μΔλN  ≡  Sₙ + C:(Δε - 1.5ΔλN)            Step A3
R_t   = (Rₙ + b₁·b₂·Δλ) / (1 + b₁·Δλ)                  Step A4
X_t   = (Xₙ + a·Δλ·N) / (1 + s·Δλ)                      Step A5
R(Δλ) = J₂(S_t - X_t) - k - R_t - K·(Δλ/Δt)^(1/n) = 0  Step A6
```

```python
@dataclass
class LC_Params(JAXMaterial):
    # Elastic
    E:      float   # Young's modulus (MPa)
    nu:     float   # Poisson's ratio (-)
    # Viscoplastic
    k:      float   # initial yield stress (MPa)
    K_visc: float   # viscosity K (MPa)
    n_visc: float   # viscosity exponent n (-)
    # Isotropic hardening R:  Ṙ = b·(R1 - R)·Ēdot_p   [Voce]
    b:      float   # saturation rate b₁ (-)
    R1:     float   # saturation value b₂ (MPa)
    # Kinematic hardening X:  Ẋ = a·Ēdot_p·N - c·X·Ēdot_p   [Armstrong-Frederick]
    a:      float   # kinematic modulus (MPa)
    c:      float   # dynamic recovery s (-)


class LC(JAXMaterial):

    def __init__(self, elastic_model, params: LC_Params):
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
            "R":     1,   # isotropic hardening variable (MPa, scalar)
            "X":     6,   # backstress tensor (Voigt)
            "eps_p": 6,   # plastic strain tensor (Voigt)
            "eps_e": 6,   # elastic strain tensor (Voigt)
            "fy":    1,   # yield check value (diagnostic)
            "dlam":  1,   # plastic multiplier (diagnostic)
        }

    # paste ALL FOUR helpers: _deviatoric, _J2, _flow_direction, df_dsigma

    @tangent_AD
    def constitutive_update(self, eps, state, dt):

        # ── 1. unpack ──────────────────────────────────────────────────────
        eps_old   = state["Strain"]
        sig_old   = state["Stress"]
        eps_p_old = state["eps_p"]
        eps_e_old = state["eps_e"]
        X_old     = state["X"]
        R_old     = state["R"][0]
        p_old     = state["p"][0]

        params = self.params
        C      = self.elastic_model.C              # ← always from elastic_model

        # ── 2. elastic predictor ───────────────────────────────────────────
        deps      = eps - eps_old
        sig_trial = sig_old + C @ deps             # Ŝ = S_n + C:ΔE

        # ── 3. yield check at trial state (X, R from previous step) ────────
        J2_trial = self._J2(self._deviatoric(sig_trial - X_old))
        fy       = J2_trial - (params.k + R_old)

        # ── 4. operand — identical structure for both branches ──────────────
        operand = (eps_old, sig_old, eps_p_old, eps_e_old,
                   X_old, R_old, p_old, deps, sig_trial, fy, dt)

        # ── 5. elastic branch ───────────────────────────────────────────────
        def elastic_update(operand):
            (eps_old, sig_old, eps_p_old, eps_e_old,
             X_old, R_old, p_old, deps, sig_trial, fy, dt) = operand
            sig_new   = sig_trial
            eps_p_new = eps_p_old
            eps_e_new = eps_e_old + deps
            X_new     = X_old
            R_new     = R_old
            p_new     = p_old
            dlam      = 0.0
            return sig_new, eps_p_new, eps_e_new, X_new, R_new, p_new, fy, dlam

        # ── 6. plastic branch ───────────────────────────────────────────────
        def plastic_update(operand):
            (eps_old, sig_old, eps_p_old, eps_e_old,
             X_old, R_old, p_old, deps, sig_trial, fy, dt) = operand

            # Flow direction at PREVIOUS step — semi-implicit
            # tr(N)=0 => C:N = 2μN => stress correction = 3μ·Δλ·N
            N = self._flow_direction(sig_old, X_old)

            # ── scalar residual R(Δλ) = 0 ──────────────────────────────────
            def R_plastic(dlam):
                # Step A3: stress in terms of Δλ
                sig_new = sig_old + C @ (deps - 1.5 * dlam * N)

                # Step A4: isotropic hardening — Ṙ = b(R1-R)Ēdot_p
                #   backward Euler analytical: R_t = (R_n + b·R1·Δλ)/(1+b·Δλ)
                R_new = (R_old + params.b * params.R1 * dlam) / (1.0 + params.b * dlam)

                # Step A5: backstress — Ẋ = a·Ēdot_p·N - c·X·Ēdot_p
                #   backward Euler analytical: X_t = (X_n + a·Δλ·N)/(1+c·Δλ)
                X_new = (X_old + params.a * dlam * N) / (1.0 + params.c * dlam)

                # Step A6: viscoplastic overstress K·(Δλ/Δt)^(1/n)
                phi = jnp.where(dlam > 0.0,
                                (params.k * dlam / dt)**(1.0 / params.n_visc),
                                0.0)

                # Step A6: scalar residual = 0
                J2_eff = self._J2(self._deviatoric(sig_new - X_new))
                return J2_eff - (params.k + R_new) - phi

            newton = JAXNewton()
            newton.set_residual(R_plastic)
            dlam, _ = newton.solve(0.0)              # initial guess always 0

            # ── final state — recompute with converged dlam ─────────────────
            # (do NOT reuse anything from inside R_plastic above)
            sig_new   = sig_old + C @ (deps - 1.5 * dlam * N)
            eps_p_new = eps_p_old + 1.5 * dlam * N
            eps_e_new = eps_e_old + (deps - 1.5 * dlam * N)
            R_new     = (R_old + params.b * params.R1 * dlam) / (1.0 + params.b * dlam)
            X_new     = (X_old + params.a * dlam * N) / (1.0 + params.c * dlam)
            p_new     = p_old + dlam

            return sig_new, eps_p_new, eps_e_new, X_new, R_new, p_new, fy, dlam

        # ── 7. branch ───────────────────────────────────────────────────────
        is_plastic = fy > 0.0
        (sig_new, eps_p_new, eps_e_new,
         X_new, R_new, p_new, fy, dlam) = jax.lax.cond(
            is_plastic, plastic_update, elastic_update, operand
        )

        # ── 8. write back ALL state variables ───────────────────────────────
        state["Strain"] = eps
        state["Stress"] = sig_new
        state["eps_p"]  = eps_p_new
        state["eps_e"]  = eps_e_new
        state["X"]      = X_new
        state["R"]      = jnp.array([R_new])
        state["p"]      = jnp.array([p_new])
        state["fy"]     = jnp.array([fy])
        state["dlam"]   = jnp.array([dlam])

        return sig_new, state
```

**LC material parameters — validated values from Tandale & Stoffel (2024) Table 1:**

| Material | E (MPa)  | ν    | n    | a (MPa)  | c (s)  | K (MPa) | k (MPa) |
|----------|----------|------|------|----------|--------|---------|---------|
| Aluminum | 67 400   | 0.32 | 2.79 | 3100     | 110    | 3.42    | 110     |
| Copper   | 113 066  | 0.32 | 8.15 | 98 939   | 1533   | 11.45   | 180     |

---

## HOW TO ADAPT FOR ANY NEW MODEL

### Step 1 — identify what is different

Run through this checklist against the new equations:

| Question | If YES → action |
|---|---|
| Is there a backstress X? | Use `_flow_direction(sig, X)` not `df_dsigma`. Use `(3/2)*dlam*N` factor. |
| New scalar hardening ODE `q̇ = f(q)·p̊`? | Derive implicit update (SKILL.md Step A4), add to state, R_plastic, and final update |
| New tensor hardening ODE `Q̇ = f(Q)·p̊`? | Derive implicit update (SKILL.md Step A5), add size-6 to state |
| Damage variable D? | Add `D: 1` to state, add `D_new = D_old + (Y/S)^s * dlam` to R_plastic and final update, multiply C by `(1-D)` in sig_trial |
| Different overstress function φ? | Change `phi` line inside R_plastic only |
| Different yield function f? | Change `J2_eff - (k + R_new) - phi` line in R_plastic |
| Two backstresses X1, X2? | Add X2 to state, add X2 update in R_plastic, use `X_total = X1_new + X2_new` in J2_eff |

### Step 2 — template for adding any new variable

```python
# ── params dataclass — add new parameters ──────────────────────────────────
@dataclass
class MyModel_Params(JAXMaterial):
    # ... existing params ...
    A_q: float   # new parameter for new ODE
    B_q: float   # new parameter for new ODE

# ── internal_state_variables — add new variable ────────────────────────────
"q": 1,   # new scalar internal variable

# ── constitutive_update — unpack ───────────────────────────────────────────
q_old = state["q"][0]

# ── operand — add q_old ────────────────────────────────────────────────────
operand = (..., q_old, ...)

# ── elastic_update — pass through unchanged ─────────────────────────────────
q_new = q_old

# ── R_plastic — add implicit update ─────────────────────────────────────────
# from SKILL.md Step A4: q̇ = (A_q - B_q·q)·Ēdot_p
q_new = (q_old + params.A_q * dlam) / (1.0 + params.B_q * dlam)

# ── after newton.solve — recompute (same formula as inside R_plastic) ───────
q_new = (q_old + params.A_q * dlam) / (1.0 + params.B_q * dlam)

# ── state write-back ─────────────────────────────────────────────────────────
state["q"] = jnp.array([q_new])
```

### Step 3 — lax.cond branch shape rule

Count every return value. Both branches must match exactly:
```python
# With new variable q added:
# elastic: return sig_new, eps_p_new, eps_e_new, X_new, R_new, p_new, q_new, fy, dlam
# plastic: return sig_new, eps_p_new, eps_e_new, X_new, R_new, p_new, q_new, fy, dlam
# ← same count, same shapes ─────────────────────────────────────────────────────────
```

---

## CRITICAL RULES — never violate

| Rule | Correct | Wrong |
|------|---------|-------|
| Stiffness C | `C = self.elastic_model.C` | Manual Lamé construction |
| Scalar state unpack | `R_old = state["R"][0]` | `R_old = state["R"]` |
| Scalar state write | `state["R"] = jnp.array([R_new])` | `state["R"] = R_new` |
| Newton guess | `newton.solve(0.0)` | `newton.solve(dlam_old)` |
| Flow direction | `N = _flow_direction(sig_old, X_old)` outside R_plastic | Computed inside R_plastic |
| Post-Newton recompute | Fresh computation with converged dlam | Reusing closure values from R_plastic |
| Nonlinear ODE update | Rational `(old + A*dlam)/(1+B*dlam)` | Explicit `old + A*dlam` |
| J2 computation | `_J2` with stop_gradient | `jnp.sqrt(1.5 * ss)` directly |