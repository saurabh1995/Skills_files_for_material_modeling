# Example Material Models

Complete implementations of classic constitutive models using the explicit integration scheme framework.

## Example 1: von Mises Plasticity with Linear Isotropic Hardening

Simplest rate-independent plasticity model.

**Equations:**
- Yield: f = J₂(σ) - (R + k)
- Flow: ε̇ᴵ = λ̇ σ'/J₂
- Hardening: R = H p

**Parameters:**
```python
@dataclass
class VonMisesParams:
    E: float = 200000.0    # Young's modulus [MPa]
    nu: float = 0.3        # Poisson's ratio
    k: float = 200.0       # Initial yield stress [MPa]
    H: float = 1000.0      # Hardening modulus [MPa]
    K_visc: float = 0.01   # Small for rate-independent
    n_visc: float = 100.0  # Large for rate-independent
```

**Key Implementation Points:**
```python
# In _plastic_update:
# 1. Compute plastic strain rate
x = fy / params.K_visc
bracket = 0.5 * (x + jnp.abs(x))
p_dot = jnp.power(bracket, params.n_visc)

# 2. Flow direction (no backstress)
sig_dev = self._deviatoric(sig_trial)
J2 = self._J2(sig_dev)
inv_J2 = jnp.where(J2 > 0.0, 1.0 / J2, 0.0)
flow_dir = sig_dev * inv_J2

# 3. Inelastic strain
eps_I_dot = 1.5 * p_dot * flow_dir
delta_eps_I = eps_I_dot * dt

# 4. Linear hardening
R_new = params.H * p_new  # Direct relationship

# 5. Stress update (no damage)
sig_new = sig_old + C @ (deps - delta_eps_I)
```

**State Variables:**
```python
return {
    "p": 1,
    "R": 1,
    "eps_p": 6,
    "eps_e": 6,
    "fy": 1,
    "p_dot": 1,
    "dp": 1,
    "eps_I_dot": 6,
    "is_plastic": 1
}
```

## Example 2: Perzyna Viscoplasticity with Exponential Hardening

Rate-dependent model with saturation hardening.

**Equations:**
- Yield: f = J₂(σ) - (R + k)
- Rate: ṗ = 〈f/K〉ⁿ
- Flow: ε̇ᴵ = (3/2) ṗ σ'/J₂
- Hardening: Ṙ = b(R₁ - R)ṗ

**Parameters:**
```python
@dataclass
class PerzynaParams:
    E: float = 200000.0
    nu: float = 0.3
    k: float = 200.0       # Initial yield
    K_visc: float = 10.0   # Viscosity [MPa·s^(1/n)]
    n_visc: float = 5.0    # Rate exponent
    b: float = 10.0        # Saturation rate
    R1: float = 100.0      # Saturation value [MPa]
```

**Key Implementation Points:**
```python
# In _plastic_update:
# Hardening evolution (saturation type)
R_dot = params.b * (params.R1 - R_old) * p_dot
R_new = R_old + R_dot * dt

# Note: As p → ∞, R → R1 (saturation)
# Initial hardening rate is high (R far from R1)
# Later hardening rate decreases (R approaches R1)
```

## Example 3: Armstrong-Frederick Kinematic Hardening

Includes backstress for cyclic loading.

**Equations:**
- Yield: f = J₂(σ - X) - (R + k)
- Rate: ṗ = 〈f/K〉ⁿ
- Flow: ε̇ᴵ = (3/2) ṗ (σ' - X')/J₂(σ - X)
- Isotropic: Ṙ = b(R₁ - R)ṗ
- Kinematic: Ẋ = (2/3)a ε̇ᴵ - c X ṗ

**Parameters:**
```python
@dataclass
class ArmstrongFrederickParams:
    E: float = 200000.0
    nu: float = 0.3
    k: float = 200.0
    K_visc: float = 10.0
    n_visc: float = 5.0
    b: float = 10.0
    R1: float = 100.0
    a: float = 10000.0     # Kinematic modulus [MPa]
    c: float = 100.0       # Dynamic recovery
```

**Key Implementation Points:**
```python
# In yield function:
# Account for backstress
sig_eff = sig_trial - X_old  # Effective stress
sig_eff_dev = self._deviatoric(sig_eff)
J2_eff = self._J2(sig_eff_dev)
fy = J2_eff - (R_old + params.k)

# In _plastic_update:
# Flow direction includes backstress
sig_eff = sig_trial - X_old
sig_eff_dev = self._deviatoric(sig_eff)
J2_eff = self._J2(sig_eff_dev)
inv_J2 = jnp.where(J2_eff > 0.0, 1.0 / J2_eff, 0.0)
flow_dir = sig_eff_dev * inv_J2

# Kinematic hardening evolution
X_dot = (2.0/3.0) * params.a * eps_I_dot - params.c * X_old * p_dot
X_new = X_old + X_dot * dt

# The c*X term provides "dynamic recovery" - X saturates
# As X grows, the recovery term limits further growth
```

**State Variables:**
```python
return {
    "p": 1,
    "R": 1,
    "X": 6,        # Backstress tensor
    "eps_p": 6,
    "eps_e": 6,
    "fy": 1,
    "p_dot": 1,
    "dp": 1,
    "R_dot": 1,
    "X_dot": 6,    # Backstress rate
    "eps_I_dot": 6,
    "is_plastic": 1
}
```

## Example 4: Lemaitre Damage Model

Includes isotropic damage evolution.

**Equations:**
- Yield: f = J₂(σ)/(1-D) - (R + k)
- Damage: Ḋ = [Dc/(εR-εD)] × [(2/3)(1+ν)σ²ₑq + 3(1-2ν)σ²ₕ] × ṗ
- Stress: σ̇ = (1-D) C : ε̇ᵉ

**Parameters:**
```python
@dataclass
class LemaitreParams:
    E: float = 200000.0
    nu: float = 0.3
    k: float = 200.0
    K_visc: float = 10.0
    n_visc: float = 5.0
    b: float = 10.0
    R1: float = 100.0
    Dc: float = 0.3        # Critical damage
    eps_R: float = 0.2     # Critical plastic strain
    eps_D: float = 0.05    # Damage threshold
```

**Key Implementation Points:**
```python
# In yield function:
# Damage affects effective stress
sig_dev = self._deviatoric(sig_trial)
J2 = self._J2(sig_dev)
denom_D = 1.0 - D_old
sigma_eff = J2 / denom_D
fy = sigma_eff - (R_old + params.k)

# In _plastic_update:
# Compute damage energy release rate
sigma_eq = self._equiv_stress(sig_new)
sigma_H = self._hydrostatic(sig_new)

nu = params.nu
bracket_damage = (
    (2.0/3.0) * (1.0 + nu) * sigma_eq**2 +
    3.0 * (1.0 - 2.0*nu) * sigma_H**2
)

# Damage rate
factor_D = params.Dc / (params.eps_R - params.eps_D + 1e-12)
D_dot = factor_D * bracket_damage * p_dot

# Update damage
D_new = D_old + D_dot * dt
D_new = jnp.clip(D_new, 0.0, params.Dc)

# Stress update includes damage
delta_eps_e = deps - delta_eps_I
sig_new = sig_old + (1.0 - D_new) * (C @ delta_eps_e)
```

**State Variables:**
```python
return {
    "p": 1,
    "D": 1,        # Damage
    "R": 1,
    "eps_p": 6,
    "eps_e": 6,
    "fy": 1,
    "p_dot": 1,
    "dp": 1,
    "D_dot": 1,    # Damage rate
    "R_dot": 1,
    "eps_I_dot": 6,
    "is_plastic": 1
}
```

## Example 5: Full Lemaitre-Chaboche Model

Complete model with damage and kinematic hardening (matches user's uploaded code).

**Equations:**
- Yield: f = J₂(σ - X)/(1-D) - (R + k)
- Rate: ṗ = 〈f/K〉ⁿ
- Flow: ε̇ᴵ = (3/2) ṗ (σ' - X')/J₂(σ - X)
- Isotropic: Ṙ = b(R₁ - R)ṗ
- Kinematic: Ẋ = (2/3)a ε̇ᴵ - c X ṗ
- Damage: Ḋ = [Dc/(εR-εD)] × Y × ṗ
- Stress: σ̇ = (1-D) C : ε̇ᵉ

**Parameters (Copper from Tandale paper):**
```python
@dataclass
class LemaitreChabocheParams:
    E: float = 113066.0
    nu: float = 0.32
    k: float = 180.0
    K_visc: float = 11.45
    n_visc: float = 8.15
    b: float = 1533.41
    R1: float = 98939.30
    a: float = 11.45       # From Table 1: s parameter
    c: float = 98939.30    # From Table 1: K parameter
    Dc: float = 0.22
    eps_R: float = 0.1629
    eps_D: float = 0.0
```

**Complete constitutive_update (abbreviated):**
```python
@tangent_AD
def constitutive_update(self, eps, state, dt):
    # Extract state
    eps_old = state["Strain"]
    deps = eps - eps_old
    sig_old = state["Stress"]
    eps_p_old = state["eps_p"]
    eps_e_old = state["eps_e"]
    X_old = state["X"]
    p_old = state["p"][0]
    D_old = state["D"][0]
    R_old = state["R"][0]
    
    # Elastic predictor
    C = self.elastic_model.C
    sig_trial = sig_old + (1.0 - D_old) * (C @ deps)
    
    # Yield function
    sig_eff = sig_trial - X_old
    sig_eff_dev = self._deviatoric(sig_eff)
    J2_eff = self._J2(sig_eff_dev)
    denom_D = 1.0 - D_old
    sigma_eff = J2_eff / denom_D
    fy = sigma_eff - (R_old + params.k)
    
    # Prepare operand
    operand = (...)
    
    def _elastic_update(operand):
        # No plasticity, no evolution
        return (sig_trial, eps_e_old + deps, X_old, ...)
    
    def _plastic_update(operand):
        # Viscoplastic strain rate
        x = fy / params.K_visc
        bracket = 0.5 * (x + jnp.abs(x))
        p_dot = jnp.power(bracket, params.n_visc)
        dp = p_dot * dt
        p_new = p_old + dp
        
        # Flow direction
        inv_J2 = jnp.where(J2_eff > 0.0, 1.0 / J2_eff, 0.0)
        flow_dir = sig_eff_dev * inv_J2
        eps_I_dot = 1.5 * p_dot * flow_dir
        delta_eps_I = eps_I_dot * dt
        
        # Update strains
        eps_p_new = eps_p_old + delta_eps_I
        delta_eps_e = deps - delta_eps_I
        eps_e_new = eps_e_old + delta_eps_e
        
        # Damage evolution
        sigma_eq = self._equiv_stress(sig_trial)
        sigma_H = self._hydrostatic(sig_trial)
        bracket_damage = (
            (2.0/3.0) * (1.0 + params.nu) * sigma_eq**2 +
            3.0 * (1.0 - 2.0*params.nu) * sigma_H**2
        )
        factor_D = params.Dc / (params.eps_R - params.eps_D + 1e-12)
        D_dot = factor_D * bracket_damage * p_dot
        D_new = jnp.clip(D_old + D_dot * dt, 0.0, params.Dc)
        
        # Stress update
        sig_new = sig_old + (1.0 - D_new) * (C @ delta_eps_e)
        
        # Hardening evolution
        R_dot = params.b * (params.R1 - R_old) * p_dot
        R_new = R_old + R_dot * dt
        
        X_dot = (2.0/3.0) * params.a * eps_I_dot - params.c * X_old * p_dot
        X_new = X_old + X_dot * dt
        
        return (sig_new, eps_e_new, X_new, ...)
    
    # Conditional execution
    is_plastic = fy > 0.0
    results = jax.lax.cond(is_plastic, _plastic_update, _elastic_update, operand)
    
    # Update state dictionary
    state["Strain"] = eps
    state["Stress"] = results[0]
    # ... update all other variables
    
    return results[0], state
```

**State Variables:**
```python
return {
    "p": 1,
    "D": 1,
    "R": 1,
    "eps_p": 6,
    "X": 6,
    "eps_e": 6,
    "fy": 1,
    "p_dot": 1,
    "dp": 1,
    "D_dot": 1,
    "R_dot": 1,
    "X_dot": 6,
    "eps_I_dot": 6,
    "is_plastic": 1
}
```

## Example 6: Simplified Damage Model

For educational purposes - minimal damage model.

**Equations:**
- Yield: f = J₂(σ) - k
- Damage: Ḋ = A σₑq ṗ
- No hardening

**Parameters:**
```python
@dataclass
class SimpleDamageParams:
    E: float = 200000.0
    nu: float = 0.3
    k: float = 200.0
    K_visc: float = 1.0
    n_visc: float = 10.0
    A: float = 0.001       # Damage coefficient
    Dc: float = 0.5        # Critical damage
```

**Key Points:**
```python
# Very simple damage evolution
sigma_eq = self._equiv_stress(sig_new)
D_dot = params.A * sigma_eq * p_dot
D_new = jnp.clip(D_old + D_dot * dt, 0.0, params.Dc)

# No hardening evolution
R_new = 0.0
X_new = jnp.zeros((6,), dtype=eps.dtype)
```

## Comparison Table

| Model | Kinematic | Isotropic | Damage | Rate Dep. | Use Case |
|-------|-----------|-----------|--------|-----------|----------|
| von Mises | No | Linear | No | No | Simple plasticity |
| Perzyna | No | Exp. Sat. | No | Yes | Creep, rate effects |
| Armstrong-Frederick | Yes | Exp. Sat. | No | Yes | Cyclic loading |
| Lemaitre | No | Exp. Sat. | Yes | Yes | Ductile failure |
| Lemaitre-Chaboche | Yes | Exp. Sat. | Yes | Yes | Complex loading |
| Simple Damage | No | No | Yes | Yes | Educational |

## Testing Material Parameters

Typical parameter ranges:

**Elastic (metals):**
- E: 70,000-400,000 MPa (Al to steel)
- ν: 0.2-0.35

**Yield:**
- k: 50-500 MPa (soft to hard metals)

**Viscoplastic:**
- K: 1-100 MPa·s^(1/n)
- n: 3-15 (higher = more rate sensitive)

**Hardening:**
- H: 100-10,000 MPa (linear)
- b: 1-100, R1: 50-500 MPa (saturation)
- a: 1,000-50,000 MPa, c: 10-500 (kinematic)

**Damage:**
- Dc: 0.1-0.5 (typical for ductile metals)
- εR: 0.1-1.0, εD: 0.0-0.1

## Validation Tests

For each model, test with:

1. **Uniaxial tension** - monotonic loading
2. **Uniaxial compression** - asymmetry check
3. **Cyclic loading** - ratcheting, Bauschinger effect
4. **Strain rate sensitivity** - different loading rates
5. **Damage accumulation** - load until failure

Expected behaviors:

- **von Mises**: Symmetric, rate-independent, linear hardening
- **Perzyna**: Rate-dependent yield, saturation hardening
- **Armstrong-Frederick**: Bauschinger effect, kinematic shift
- **Lemaitre**: Damage accumulation, softening, failure
- **Lemaitre-Chaboche**: All of the above combined