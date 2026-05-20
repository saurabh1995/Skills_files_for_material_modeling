# box-algorithms.md — Algorithm Boxes for Implicit Viscoplastic Integration
# Mathematical reference — all models
# Code implementation: see code-template.md

---

## HOW TO USE THIS FILE

Each box gives the mathematical algorithm for a model type.
The boxes show the derivation steps (A1–A6 from SKILL.md) applied to each specific model.
Use these when:
- Deriving a new scheme by hand
- Verifying what goes inside R_plastic(dlam)
- Cross-checking signs and factors

---

## BOX 1 — Perzyna, Pure Plasticity (no hardening)

**Governing equations:**
```
Ėp     = η·φ(f)·n           flow rule
f      = J₂(σ) - σ_y        von Mises yield function
φ      = (f/σ_y)^N          overstress function (power law)
n      = ∂f/∂σ              flow direction
```

**Derivation:**
```
A2: Ŝ = S_n + C:ΔE;  f_y = J₂(Ŝ) - σ_y

A3: ΔEp = Δλ·n  →  S_{n+1} = Ŝ - C:(Δλ·n) = S_n + C:(ΔE - Δλ·n)

A4/A5: no hardening ODEs

A6: Ēdot_p = η·φ(f)  →  Δλ = η·Δt·φ(f(S_{n+1}))
    Residual: R(Δλ) = Δλ - η·Δt·(f(S_{n+1})/σ_y)^N = 0
```

**Algorithm:**
```
GIVEN: ΔE, state at t-1

1. Trial stress:    Ŝ = S_{t-1} + C:ΔE
2. Yield check:     f_y = J₂(Ŝ) - σ_y
   if f_y ≤ 0:      ELASTIC — accept Ŝ, done
   else:             PLASTIC — continue

3. Flow direction (semi-implicit):  n = (3/2)·dev(S_{t-1})/J₂(S_{t-1})

4. Newton solve for Δλ:
   R(Δλ) = Δλ - η·Δt·((J₂(S_{t-1} + C:(ΔE - Δλ·n)) - σ_y)/σ_y)^N = 0
   Initial guess: Δλ = 0

5. Final update:
   S_t   = S_{t-1} + C:(ΔE - Δλ·n)
   Ep_t  = Ep_{t-1} + Δλ·n
   Ee_t  = Ee_{t-1} + ΔE - Δλ·n
```

**R_plastic code:**
```python
def R_plastic(dlam):
    sig_new = sig_old + C @ (deps - dlam * n)
    J2      = self._J2(self._deviatoric(sig_new))
    f       = J2 - params.sig_y
    phi     = jnp.where(f > 0.0, (f / params.sig_y)**params.N, 0.0)
    return dlam - params.eta * dt * phi
```

---

## BOX 2 — Perzyna, Isotropic Hardening

**Additional equations over Box 1:**
```
f  = J₂(σ) - (σ_y + H·p)    linear hardening  [or Voce, see below]
ṗ  = Ēdot_p                  equivalent plastic strain
```

**Derivation:**
```
A4: ṗ = Ēdot_p  →  p_{n+1} = p_n + Δλ   (trivial — linear ODE)

A6: R(Δλ) = Δλ - η·Δt·((J₂(S_{n+1}) - σ_y - H·(p_n + Δλ))/σ_y)^N = 0
```

**Algorithm:**
```
1. Trial:  Ŝ = S_{t-1} + C:ΔE
2. Check:  f_y = J₂(Ŝ) - (σ_y + H·p_{t-1})
           if f_y ≤ 0: ELASTIC, done
3. n = (3/2)·dev(S_{t-1})/J₂(S_{t-1})
4. Newton: R(Δλ) = 0  (see A6 above)
5. Update: S_t = S_{t-1} + C:(ΔE - Δλ·n)
           p_t = p_{t-1} + Δλ
           Ep_t = Ep_{t-1} + Δλ·n
```

**Nonlinear Voce hardening variant:**
```
Replace linear H·p with Voce R: Ṙ = b·(R_sat - R)·Ēdot_p
Implicit update (Step A4): R_t = (R_{t-1} + b·R_sat·Δλ) / (1 + b·Δλ)
Yield function becomes: f = J₂(S_{n+1}) - (σ_y + R_{n+1})
```

---

## BOX 3 — Duvaut-Lions Model

**Governing equations:**
```
Ėvp = (1/τ)·C⁻¹·(σ - σ̄)    viscoplastic strain rate
σ̄   = projection of σ onto the yield surface (inviscid backbone stress)
τ   = relaxation time (viscosity parameter)
```

**Algorithm (one-step implicit, θ = 1):**
```
GIVEN: ΔE, state at t-1

1. Trial stress:    Ŝ = S_{t-1} + C:ΔE
2. Yield check:     f_y = J₂(Ŝ) - σ_y
   if f_y ≤ 0: ELASTIC, done

3. Backbone stress (inviscid return mapping):
   σ̄_t = σ̄_{t-1} + D♯¹:ΔE
   where D♯¹ = C - (C·n·n̄ᵀ·C)/(h + n̄ᵀ·C·n̄),   n̄ = ∂f/∂σ̄

4. Algorithmic tangent:
   D♯ = τ/(τ + Δt)·[C + (Δt/τ)·D♯¹]

5. Pseudo-force:
   Δq = (τ·Δt/(τ+Δt))·[(1/τ)·(S_{t-1} - σ̄_{t-1})]

6. Stress update:
   S_t = S_{t-1} + D♯:ΔE - Δq

7. Viscoplastic strain rate:
   ε̇vp_t = (1/τ)·C⁻¹·(S_t - σ̄_t)

8. Internal variable:
   p_t = p̄_t - τ·|ε̇vp_t|
   where p̄_t = backbone equivalent plastic strain
```

**Notes:**
- Two-step process: first inviscid backbone, then viscoplastic relaxation
- No scalar Newton needed — closed-form update
- τ = relaxation time; τ→0: rate-independent plasticity; τ→∞: purely elastic

---

## BOX 4 — Consistency Model (Rate-Dependent Yield Surface)

**Governing equations:**
```
f(σ, κ, κ̇) = 0               rate-dependent yield surface
f = J₂(σ) - (σ_y + H·κ + m·κ̇)
κ̇ = Ēdot_p                   internal variable rate
```

**Algorithm:**
```
GIVEN: ΔE, state at t-1

1. Trial stress:  Ŝ = S_{t-1} + C:ΔE
2. Yield check:   f_y = f(Ŝ, κ_{t-1}, κ̇_{t-1})
   if f_y ≤ 0: ELASTIC, done

3. n = df/dσ = (3/2)·dev(S_{t-1})/J₂(S_{t-1})

4. Newton solve for Δλ:
   R(Δλ) = J₂(S_{n+1}) - σ_y - H·(κ_n + Δλ) - m·(Δλ/Δt) = 0
   where S_{n+1} = S_n + C:(ΔE - Δλ·n)
   Initial guess: Δλ = 0

5. Final update:
   S_t  = S_{t-1} + C:(ΔE - Δλ·n)
   κ_t  = κ_{t-1} + Δλ
   κ̇_t = Δλ/Δt
   Ep_t = Ep_{t-1} + Δλ·n
```

**R_plastic code:**
```python
def R_plastic(dlam):
    sig_new   = sig_old + C @ (deps - dlam * n)
    kappa_new = kappa_old + dlam
    kappa_dot = dlam / dt
    J2        = self._J2(self._deviatoric(sig_new))
    return J2 - params.sig_y - params.H * kappa_new - params.m * kappa_dot
```

**Notes:**
- Enforces f = 0 exactly (unlike Perzyna which allows f > 0)
- m > 0: rate-dependent hardening (S-type instabilities)
- H < 0: strain softening (H-type instabilities)
- Can handle both H-type and S-type instabilities in same model

---

## BOX 5 — Lemaitre-Chaboche (Full Derivation)

This box shows the complete hand-calculation derivation following the procedure in
`LC_implicit_scheme_hand_calculation.pdf` (Siempelkamp notation).

**Governing equations:**
```
Additive split:   Ė = Ėe + Ėp
Flow rule:        Ėp = (3/2)·Ēdot_p·N        N = dev(S-X)/J₂(S-X)
Overstress:       Ēdot_p = ⟨(J₂(S-X) - R - k)/K⟩ⁿ
Backstress:       Ẋ = a·Ēdot_p·N - s·X·Ēdot_p   (Armstrong-Frederick)
Iso hardening:    Ṙ = b₁·(b₂ - R)·Ēdot_p         (Voce saturation)
Hooke's law:      S = C:(E - Ep)
```

**Step A2 — Elastic predictor:**
```
Ŝ = S_{t-1} + C:ΔE          trial stress (freeze Ep)
X̂ = X_{t-1}                  backstress unchanged

f_y = J₂(Ŝ - X̂) - R_{t-1} - k

if f_y ≤ 0: ELASTIC
if f_y > 0: PLASTIC corrector needed
```

**Step A3 — Stress in terms of Δλ:**
```
ΔEp = (3/2)·Δλ·N

S_t = Ŝ - C:ΔEp = Ŝ - C:((3/2)·Δλ·N)

Since tr(N) = 0 (N is deviatoric):
  C = (κ - 2μ/3)(I⊗I) + 2μ·𝕀
  C:N = (κ - 2μ/3)·tr(N)·I + 2μ·N = 0 + 2μN = 2μN

∴ S_t = Ŝ - 3μ·Δλ·N     [equivalently: S_n + C:(ΔE - 1.5·Δλ·N)]
```

**Step A4 — Isotropic hardening R (from hand-calc page 3):**
```
Ṙ = b₁·(b₂ - R)·Ēdot_p
Backward Euler: R_t - R_{t-1} = b₁·(b₂ - R_t)·Δλ
R_t + b₁·R_t·Δλ = R_{t-1} + b₁·b₂·Δλ
R_t·(1 + b₁·Δλ) = R_{t-1} + b₁·b₂·Δλ

∴ R_t = (R_{t-1} + b₁·b₂·Δλ) / (1 + b₁·Δλ)
```

**Step A5 — Backstress X (from hand-calc page 4):**
```
Ẋ = a·Ēdot_p·N - s·X·Ēdot_p
Backward Euler (N fixed at t-1 — semi-implicit):
X_t - X_{t-1} = a·Δλ·N - s·X_t·Δλ
X_t·(1 + s·Δλ) = X_{t-1} + a·Δλ·N

∴ X_t = (X_{t-1} + a·Δλ·N) / (1 + s·Δλ)
```

**Step A6 — Scalar residual (from hand-calc page 5):**
```
Overstress law (plastic domain, drop ⟨⟩):
  Ēdot_p = ((J₂(S-X) - R - k)/K)ⁿ

Multiply by Δt:
  Δλ = Δt·((J₂(S_t - X_t) - R_t - k)/K)ⁿ

Take (1/n)-th power:
  K·(Δλ/Δt)^(1/n) = J₂(S_t - X_t) - R_t - k

Rearrange to zero:
  R(Δλ) = J₂(S_t - X_t) - k - R_t - K·(Δλ/Δt)^(1/n) = 0

where S_t, R_t, X_t are all functions of Δλ from Steps A3–A5.
This is a SCALAR equation in ONE unknown Δλ.
```

**Complete algorithm:**
```
GIVEN: ΔE = E_t - E_{t-1}, full state at t-1

1. Trial:    Ŝ = S_{t-1} + C:ΔE
2. Check:    f_y = J₂(Ŝ - X_{t-1}) - R_{t-1} - k
             if f_y ≤ 0: ELASTIC — accept trial, done

3. N = dev(S_{t-1} - X_{t-1}) / J₂(S_{t-1} - X_{t-1})   [fixed for Newton]

4. Newton solve for Δλ starting from 0:
   Inside R_plastic(Δλ):
     S_t = S_{t-1} + C:(ΔE - 1.5·Δλ·N)
     R_t = (R_{t-1} + b₁·b₂·Δλ) / (1 + b₁·Δλ)
     X_t = (X_{t-1} + a·Δλ·N)   / (1 + s·Δλ)
     φ   = K·(Δλ/Δt)^(1/n)
     return J₂(S_t - X_t) - k - R_t - φ

5. Final state (recompute with converged Δλ):
   S_t   = S_{t-1} + C:(ΔE - 1.5·Δλ·N)
   Ep_t  = Ep_{t-1} + 1.5·Δλ·N
   Ee_t  = Ee_{t-1} + ΔE - 1.5·Δλ·N
   R_t   = (R_{t-1} + b₁·b₂·Δλ) / (1 + b₁·Δλ)
   X_t   = (X_{t-1} + a·Δλ·N)   / (1 + s·Δλ)
   p_t   = p_{t-1} + Δλ
```

**Notes:**
- Quadratic convergence of Newton (implicit scheme)
- R and X updates are bounded: R→b₂, X→(a/s)N as Δλ→∞
- Semi-implicit N direction is standard — avoids tensor Newton solve
- Equivalent to Algorithm 1 in Tandale & Stoffel (2024)

---

## BOX 6 — Extension: Lemaitre Ductile Damage

**Additional equation over Box 5:**
```
Ḋ = (Y/S)^s · Ēdot_p    where  Y = σ²_eq / (2E(1-D)²)   strain energy release rate
```

**Modified stress equation:**
```
S = C:(E - Ep) becomes:
S = (1-D)·C:(E - Ep)    [effective stress concept]

Trial stress with damage:
Ŝ = S_{t-1} + (1-D_{t-1})·C:ΔE
```

**Damage update (semi-implicit in D):**
```
Y_{t-1} = σ²_eq_{t-1} / (2·E·(1-D_{t-1})²)    [at previous step]
D_t = D_{t-1} + (Y_{t-1}/params.S_d)^params.s · Δλ
```

**Add to R_plastic:**
```python
def R_plastic(dlam):
    # existing LC lines (sig, R, X updates) ...

    # damage — semi-implicit (uses D_old to keep scalar in dlam)
    Y_old = sig_eq_old**2 / (2.0 * params.E * (1.0 - D_old)**2)
    D_new = D_old + (Y_old / params.S_d)**params.s_exp * dlam

    # modified yield check accounts for damage
    J2_eff = self._J2(self._deviatoric((sig_new - X_new) / (1.0 - D_new)))
    return J2_eff - (params.k + R_new) - phi
```

---

## COMMON NOTATION

| Symbol | Meaning | Voigt components |
|--------|---------|-----------------|
| S | Cauchy stress | [S₁₁, S₂₂, S₃₃, S₁₂, S₂₃, S₁₃] |
| E | total strain | [E₁₁, E₂₂, E₃₃, E₁₂, E₂₃, E₁₃] |
| Ep | plastic strain | same structure |
| X | backstress | same structure |
| N | flow direction = dev(S-X)/J₂(S-X) | same structure |
| C | elastic stiffness | 6×6 matrix |
| Δλ | plastic multiplier increment | scalar |
| n | Ēdot_p exponent (LC) | scalar |
| K | viscosity parameter (LC) | scalar (MPa) |
| k | initial yield stress (LC) | scalar (MPa) |
| b₁, b₂ | Voce hardening parameters | scalar |
| a, s | Armstrong-Frederick parameters | scalar |
| η | fluidity (Perzyna) | scalar |
| N | power-law exponent (Perzyna) | scalar |
| τ | relaxation time (Duvaut-Lions) | scalar |
| Δt | time step | scalar |

**Voigt inner product convention:**
```
A:B = A₁₁B₁₁ + A₂₂B₂₂ + A₃₃B₃₃ + 2(A₁₂B₁₂ + A₂₃B₂₃ + A₁₃B₁₃)
```
Factor-of-2 on shear terms is embedded in `_J2` implementation in `code-template.md`.