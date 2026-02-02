---
name: implicit-integrator
description: Generate fully implicit integration schemes for differential equations, particularly viscoplastic and plasticity models. Use when users provide differential equations (e.g., stress-strain relations, rate-dependent plasticity, consistency conditions) and request implicit time integration algorithms, backward Euler methods, return mapping algorithms, or stress update procedures. Especially relevant for computational mechanics, finite element implementations, and material constitutive models.
---

# Implicit Integration Scheme Generator

Generate robust, fully implicit time integration schemes for differential equations with automatic Newton-Raphson solver implementation.

## Core Workflow

### 1. Analyze Input Equations

Extract key information from user's equations:
- **Constitutive relations**: Stress-strain, flow rules, hardening laws
- **Evolution equations**: Plastic strain rates, internal variables, yield criteria
- **Model type**: Perzyna, Duvaut-Lions, consistency, or custom viscoplasticity
- **Variables**: State variables, material parameters, time step

### 2. Identify Integration Strategy

Based on equation structure:

**For overstress models (Perzyna, Duvaut-Lions)**:
- Use residual formulation: `R(Δλ) = Δλ - η·Δt·φ(σ, κ)`
- Solve scalar Newton-Raphson for plastic multiplier Δλ

**For consistency models**:
- Enforce yield condition: `f(σ, κ, κ̇) = 0`
- Use consistency condition for Δλ: `∂f/∂σ·Δσ + ∂f/∂κ·Δκ + ∂f/∂κ̇·Δκ̇ = 0`

**For general rate-dependent models**:
- Identify coupled differential equations
- Formulate residuals for all unknowns
- Apply implicit backward Euler discretization

### 3. Formulate Residual Function

Transform continuous equations to discrete residuals:

```
Continuous: ε̇ᵖ = γ·φ(f)·∂f/∂σ
Discrete:   R(Δλ) = Δλ - γ·Δt·φ(f(σⁿ⁺¹))
```

Key steps:
- Replace rates with finite differences: `κ̇ → Δκ/Δt`
- Express unknowns at time n+1 in terms of increments
- Include elastic predictor: `σᵗʳⁱᵃˡ = σₙ + C:(ε - εₙ)`
- Account for plastic correction: `σⁿ⁺¹ = σₙ + C:(Δε - Δλ·n)`

### 4. Generate Newton-Raphson Solver

Structure the solver following this pattern:

```python
def residual(dlam):
    # Update state at n+1 using current dlam estimate
    sig_new = sig_old + C @ ((eps - eps_old) - dlam * n)
    
    # Evaluate yield/flow function
    f = yield_function(sig_new, kappa_new)
    phi = overstress_function(f)  # if applicable
    
    # Return residual
    return dlam - eta * dt * phi
```

Use JAX Newton solver pattern:
```python
newton = JAXNewton()
newton.set_residual(residual)
dlam, _ = newton.solve(initial_guess)
```

### 5. Implement State Update

Complete the constitutive update:

```python
# Elastic-plastic split
operand = (eps, eps_old, sig_old, eps_p_old, dt, ...)

def elastic_update(operand):
    # No plastic flow
    sig_new = sig_trial
    eps_p_new = eps_p_old
    return sig_new, eps_p_new, ...

def plastic_update(operand):
    # Solve for dlam via Newton
    dlam = newton_solve(...)
    
    # Update plastic strain
    n = df_dsigma(sig_old)  # or sig_new for implicit
    eps_p_new = eps_p_old + dlam * n
    
    # Update stress
    sig_new = sig_old + C @ (deps_elastic)
    return sig_new, eps_p_new, ...

# Conditional branching
is_plastic = (f_trial > 0.0)
sig_new, eps_p_new, ... = jax.lax.cond(
    is_plastic, plastic_update, elastic_update, operand
)
```

## Pattern Library

### Pattern 1: Perzyna Overstress Model (Box 2)

**Governing equations**:
```
ε̇ᵖ = γ·(f/σ₀)ᴺ·∂f/∂σ  (for f > 0)
f = J₂(σ) - σᵧ
```

**Implicit residual**:
```python
def R_plastic(dlam):
    sig_new = sig_old + C @ ((eps - eps_old) - dlam * df_dsigma(sig_old))
    J2 = equiv_stress(sig_new)
    f = J2 - sig_y
    phi = (f / sig_y)**N if f > 0 else 0.0
    return dlam - eta * dt * phi
```

Key features:
- Flow direction evaluated at σₙ (semi-implicit) or σⁿ⁺¹ (fully implicit)
- Power law viscosity function
- Scalar Newton solve for Δλ

### Pattern 2: Consistency Model (Box 4)

**Governing equations**:
```
f(σ, κ, κ̇) = 0
f = J₂(σ) - (σᵧ + H·κ + m·κ̇)
```

**Implicit residual**:
```python
def residual(dlam):
    lam_new = lam_old + dlam
    lam_dot_new = dlam / dt if dt > 0 else 0.0
    sig_new = sig_old + C @ ((eps - eps_old) - dlam * n)
    return yield_function(sig_new, lam_new, lam_dot_new)
```

Key features:
- Rate-dependent yield surface
- Enforces f = 0 exactly
- Suitable for S-type instabilities (strain-rate softening)

### Pattern 3: Duvaut-Lions Model

**Governing equations**:
```
ε̇ᵖ = (1/τ)·D⁻¹·(σ - σ̄)
σ̄ = projection of σ onto yield surface
```

**Implementation**:
```python
# Step 1: Compute backbone stress (inviscid projection)
sig_bar = project_to_yield_surface(sig_trial)

# Step 2: Viscoplastic relaxation
def residual(dlam):
    eps_vp_dot = (1/tau) * C_inv @ (sig - sig_bar)
    return dlam - dt * norm(eps_vp_dot)
```

### Pattern 4: General Rate-Dependent Plasticity

For coupled systems with multiple internal variables:

```python
def residual_system(unknowns):
    dlam, dkappa1, dkappa2 = unknowns
    
    # Update all state variables
    sig_new = sig_old + C @ (deps - dlam * n)
    kappa1_new = kappa1_old + dkappa1
    kappa2_new = kappa2_old + dkappa2
    
    # Residuals for each equation
    R1 = dlam - dt * flow_rule(sig_new, ...)
    R2 = dkappa1 - dt * evolution1(sig_new, kappa1_new, ...)
    R3 = dkappa2 - dt * evolution2(sig_new, kappa2_new, ...)
    
    return jnp.array([R1, R2, R3])
```

## Key Derivatives

Many implicit schemes require yield surface gradients:

### First derivative (flow direction):
```python
def df_dsigma(sig):
    s11, s22, s33, s12, s23, s13 = sig
    p = (s11 + s22 + s33) / 3.0
    s_dev = [s11-p, s22-p, s33-p, s12, s23, s13]
    
    J2 = sqrt(1.5 * (s_dev @ s_dev))  # von Mises
    return (3.0 / (2.0 * J2)) * s_dev
```

### Second derivative (for consistent tangent):
```python
def d2f_dsigma2(sig):
    # Deviatoric projection tensor
    P_dev = [[2/3, -1/3, -1/3, 0, 0, 0],
             [-1/3, 2/3, -1/3, 0, 0, 0],
             [-1/3, -1/3, 2/3, 0, 0, 0],
             [0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 1]]
    
    s_dev = deviatoric(sig)
    sigma_eq = equiv_stress(sig)
    
    return (3/(2*sigma_eq)) * P_dev - (3/(2*sigma_eq**3)) * outer(s_dev, s_dev)
```

## Implementation Checklist

- [ ] Identify equation type (overstress vs consistency)
- [ ] Formulate residual function R(Δλ) or R(unknowns)
- [ ] Implement elastic predictor
- [ ] Set up Newton-Raphson solver with residual
- [ ] Handle elastic-plastic branching with `jax.lax.cond`
- [ ] Update all state variables (stress, plastic strain, internal variables)
- [ ] Include regularization for σₑq = 0 (use small epsilon)
- [ ] Test with elastic case (should return trial stress)
- [ ] Test with large plastic increment

## Common Pitfalls

**Issue**: Division by zero in `df_dsigma` when stress is zero
**Solution**: Use regularized J₂ with small epsilon (1e-16)

**Issue**: Newton solver doesn't converge
**Solution**: 
- Check residual formulation (sign errors)
- Verify initial guess (often 0.0 or previous Δλ)
- Ensure consistent units (stress, time, viscosity)

**Issue**: Incorrect plastic flow direction
**Solution**: Evaluate gradient at correct stress state (σₙ vs σⁿ⁺¹)

**Issue**: State variables not updating
**Solution**: Remember to update state dictionary at end of constitutive_update

## References

See `references/box-algorithms.md` for the three main algorithm boxes from Wang et al. (1997):
- Box 2: Perzyna fully implicit algorithm
- Box 3: Duvaut-Lions algorithm  
- Box 4: Consistency model algorithm
