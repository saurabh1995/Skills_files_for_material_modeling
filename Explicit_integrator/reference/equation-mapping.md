# Equation to Code Mapping Guide

This reference shows how to translate common constitutive equations into JAX-compatible Python code for the explicit integration scheme.

## Fundamental Equations

### 1. Strain Decomposition (Eq. 1)

**Equation:**
```
ε̇ = ε̇ᵉ + ε̇ᴵ
```

**Code (forward Euler):**
```python
# At time step t:
eps = eps_e + eps_p  # Total = elastic + plastic

# During update:
delta_eps_e = deps - delta_eps_I  # Elastic = total - inelastic
eps_e_new = eps_e_old + delta_eps_e
eps_p_new = eps_p_old + delta_eps_I
```

### 2. Stress-Strain Law (Eq. 2)

**Equation (with damage):**
```
σ̇ = (1-D) C : ε̇ᵉ = (1-D) C : (ε̇ - ε̇ᴵ)
```

**Code:**
```python
# Elastic predictor (assume elastic)
sig_trial = sig_old + (1.0 - D_old) * (C @ deps)

# Plastic corrector (if plastic)
delta_eps_e = deps - delta_eps_I
sig_new = sig_old + (1.0 - D_new) * (C @ delta_eps_e)
```

**Alternative (without damage):**
```python
sig_new = sig_old + C @ (deps - delta_eps_I)
```

### 3. Yield Function (Eq. 7)

**Equation (Lemaitre-Chaboche):**
```
f = J₂(σ - X)/(1-D) - (R + k)
```

**Code:**
```python
# Compute effective stress (remove backstress)
sig_eff = sig_trial - X_old
sig_eff_dev = self._deviatoric(sig_eff)
J2_eff = self._J2(sig_eff_dev)

# Account for damage
denom_D = 1.0 - D_old
sigma_eff = J2_eff / denom_D

# Yield function
fy = sigma_eff - (R_old + params.k)

# Check: fy > 0 means plastic, fy <= 0 means elastic
is_plastic = fy > 0.0
```

**von Mises (no backstress, no damage):**
```python
sig_dev = self._deviatoric(sig_trial)
J2 = self._J2(sig_dev)
fy = J2 - (R_old + params.k)
```

### 4. Flow Rule (Eq. 8)

**Equation (associative plasticity):**
```
ε̇ᴵ = (3/2) ṗ × ∂f/∂σ = (3/2) ṗ × (σ' - X')/J₂(σ - X)
```

**Code:**
```python
# Compute flow direction (normal to yield surface)
sig_eff = sig_trial - X_old  # or sig_new - X_new
sig_eff_dev = self._deviatoric(sig_eff)
J2_eff = self._J2(sig_eff_dev)

# Safe division
inv_J2 = jnp.where(J2_eff > 0.0, 1.0 / J2_eff, 0.0)
flow_dir = sig_eff_dev * inv_J2  # Normalized direction

# Inelastic strain rate
eps_I_dot = 1.5 * p_dot * flow_dir

# Inelastic strain increment
delta_eps_I = eps_I_dot * dt
```

## Viscoplastic Consistency

### 5. Perzyna Viscoplasticity (Eq. 9)

**Equation:**
```
ṗ = 〈[J₂(σ-X)/(1-D) - R - k]/K〉ⁿ
```

where `〈x〉 = max(x, 0)` (McCauley bracket)

**Code:**
```python
# McCauley bracket implementation
x = fy / params.K_visc
bracket = 0.5 * (x + jnp.abs(x))  # = max(x, 0)

# Plastic strain rate
p_dot = jnp.power(bracket, params.n_visc)

# Plastic strain increment
dp = p_dot * dt
p_new = p_old + dp
```

**Rate-independent plasticity:**
For rate-independent plasticity, use very large `n_visc` (e.g., 100) and very small `K_visc` (e.g., 0.01) to approximate the limit.

## Hardening Evolution

### 6. Isotropic Hardening (Eq. 10)

**Equation (exponential saturation):**
```
Ṙ = b(R₁ - R)ṗ
```

**Code:**
```python
# Rate
R_dot = params.b * (params.R1 - R_old) * p_dot

# Update (forward Euler)
R_new = R_old + R_dot * dt
```

**Linear hardening:**
```
Ṙ = H ṗ
```

```python
R_dot = params.H * p_dot
R_new = R_old + R_dot * dt
```

### 7. Kinematic Hardening

**Armstrong-Frederick (Eq. 11):**
```
Ẋ = (2/3) a ε̇ᴵ - c X ṗ
```

**Code:**
```python
# Rate
X_dot = (2.0/3.0) * params.a * eps_I_dot - params.c * X_old * p_dot

# Update
X_new = X_old + X_dot * dt
```

**Prager (linear kinematic):**
```
Ẋ = (2/3) a ε̇ᴵ
```

```python
X_dot = (2.0/3.0) * params.a * eps_I_dot
X_new = X_old + X_dot * dt
```

**Chaboche (multiple backstresses):**
```
Ẋᵢ = (2/3) aᵢ ε̇ᴵ - cᵢ Xᵢ ṗ,  i = 1,2,...,M
X = Σ Xᵢ
```

```python
# Need M backstress tensors in state
for i in range(M):
    X_i_dot = (2.0/3.0) * params.a[i] * eps_I_dot - params.c[i] * X_i_old * p_dot
    X_i_new = X_i_old + X_i_dot * dt

# Total backstress
X_new = sum(X_i_new for all i)
```

## Damage Evolution

### 8. Lemaitre Damage (Eq. 7)

**Equation:**
```
Ḋ = [Dc/(εR - εD)] × [(2/3)(1+ν)σ²ₑq + 3(1-2ν)σ²ₕ] × ṗ
```

**Code:**
```python
# Compute equivalent and hydrostatic stress
sigma_eq = self._equiv_stress(sig_new)  # Huber-Mises
sigma_H = self._hydrostatic(sig_new)     # Mean normal stress

# Damage strain energy release rate
nu = params.nu
bracket_damage = (
    (2.0/3.0) * (1.0 + nu) * sigma_eq**2 +
    3.0 * (1.0 - 2.0*nu) * sigma_H**2
)

# Damage rate
factor_D = params.Dc / (params.eps_R - params.eps_D + 1e-12)
D_dot = factor_D * bracket_damage * p_dot

# Update
D_new = D_old + D_dot * dt

# Enforce physical bounds: D ∈ [0, Dc]
D_new = jnp.clip(D_new, 0.0, params.Dc)
```

**Simplified damage (proportional to plastic work):**
```
Ḋ = A σₑq ṗ
```

```python
sigma_eq = self._equiv_stress(sig_new)
D_dot = params.A * sigma_eq * p_dot
D_new = jnp.clip(D_old + D_dot * dt, 0.0, params.Dc)
```

## Helper Functions

### J2 Invariant (Huber-Mises Stress)

**Definition:**
```
J₂ = √(3/2 s:s) = √[(s₁₁² + s₂₂² + s₃₃² + 2s₁₂² + 2s₁₃² + 2s₂₃²) × 3/2]
```

where s is deviatoric stress.

**Code (with regularization):**
```python
def _J2(self, sig_dev):
    s = sig_dev
    
    # Voigt notation: factor of 2 for shear components
    s_colon_s = (
        s[0]**2 + s[1]**2 + s[2]**2 +
        2.0 * (s[3]**2 + s[4]**2 + s[5]**2)
    )
    
    val = 1.5 * s_colon_s
    val_pos = jnp.maximum(val, 0.0)
    
    # Physical value
    J2_phys = jnp.sqrt(val_pos)
    
    # Regularized for gradient (avoid sqrt(0))
    eps_reg = 1e-16
    J2_reg = jnp.sqrt(val_pos + eps_reg)
    
    # Return physical value with regularized gradient
    return jax.lax.stop_gradient(J2_phys - J2_reg) + J2_reg
```

### Deviatoric Stress

**Definition:**
```
s = σ - (1/3)tr(σ)I
```

**Code:**
```python
def _deviatoric(self, sig):
    # Mean normal stress
    p = (sig[0] + sig[1] + sig[2]) / 3.0
    
    # Deviatoric components (Voigt)
    return jnp.array([
        sig[0] - p,
        sig[1] - p,
        sig[2] - p,
        sig[3],      # Shear components unchanged
        sig[4],
        sig[5],
    ], dtype=sig.dtype)
```

### Hydrostatic Stress

**Definition:**
```
σₕ = (1/3)tr(σ) = (σ₁₁ + σ₂₂ + σ₃₃)/3
```

**Code:**
```python
def _hydrostatic(self, sig):
    return (sig[0] + sig[1] + sig[2]) / 3.0
```

### Equivalent (von Mises) Stress

**Definition:**
```
σₑq = √(3 J₂)
```

**Code:**
```python
def _equiv_stress(self, sig):
    sig_dev = self._deviatoric(sig)
    return self._J2(sig_dev)  # Already includes √(3/2 s:s)
```

## Equation Patterns

### Pattern 1: Rate-Dependent Evolution

For any variable Q evolving as Q̇ = f(Q, p, ...)ṗ:

```python
# In _plastic_update:
Q_dot = f(Q_old, p_old, ...) * p_dot
Q_new = Q_old + Q_dot * dt
```

### Pattern 2: Saturation-Type Evolution

For Q̇ = a(Q∞ - Q)ṗ:

```python
Q_dot = params.a * (params.Q_inf - Q_old) * p_dot
Q_new = Q_old + Q_dot * dt
```

### Pattern 3: Recovery-Type Evolution

For Q̇ = a ε̇ᴵ - b Q ṗ:

```python
Q_dot = params.a * eps_I_dot - params.b * Q_old * p_dot
Q_new = Q_old + Q_dot * dt
```

### Pattern 4: Stress-Driven Evolution

For Q̇ = g(σ) ṗ:

```python
# Compute stress-dependent term
g_sigma = compute_stress_function(sig_new, params)

# Evolution
Q_dot = g_sigma * p_dot
Q_new = Q_old + Q_dot * dt
```

## Common Model Combinations

### Model 1: J2 Plasticity (von Mises)

- Yield: f = J₂(σ) - (R + k)
- Flow: ε̇ᴵ = (3/2) ṗ σ'/J₂
- Hardening: Ṙ = H ṗ
- No damage, no backstress

### Model 2: Perzyna Viscoplasticity

- Yield: f = J₂(σ) - (R + k)
- Rate: ṗ = 〈f/K〉ⁿ
- Flow: ε̇ᴵ = (3/2) ṗ σ'/J₂
- Hardening: Ṙ = b(R₁ - R)ṗ

### Model 3: Chaboche Model

- Yield: f = J₂(σ - X) - (R + k)
- Rate: ṗ = 〈f/K〉ⁿ
- Flow: ε̇ᴵ = (3/2) ṗ (σ' - X')/J₂(σ - X)
- Isotropic: Ṙ = b(R₁ - R)ṗ
- Kinematic: Ẋ = (2/3)a ε̇ᴵ - c X ṗ

### Model 4: Lemaitre-Chaboche (Full)

- All from Model 3, plus:
- Damage in yield: f = J₂(σ - X)/(1-D) - (R + k)
- Damage evolution: Ḋ = [Dc/(εR-εD)] × Y × ṗ
- Stress update: σ̇ = (1-D) C : ε̇ᵉ

## Time Integration

All equations use **forward Euler** (explicit):

```
Q(t+Δt) = Q(t) + Q̇(t) × Δt
```

For better accuracy with larger time steps, consider:
- **Backward Euler** (implicit, requires iteration)
- **Midpoint rule** (semi-implicit)
- **Runge-Kutta methods** (multi-stage explicit)

The current template uses forward Euler for simplicity and compatibility with JAX's automatic differentiation.