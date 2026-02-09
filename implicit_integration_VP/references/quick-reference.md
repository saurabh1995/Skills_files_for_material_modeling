# Quick Reference: Implicit Integration Patterns

## Decision Tree: Which Model to Use?

```
Does the model allow σ outside yield surface (f > 0)?
├─ YES → Overstress model
│   │
│   └─ How is viscoplastic flow defined?
│       ├─ Direct flow rule: ε̇ᵖ = γ·φ(f)·n → Perzyna
│       └─ Stress difference: ε̇ᵖ = (1/τ)·(σ - σ̄) → Duvaut-Lions
│
└─ NO → Consistency model
    │
    └─ Is yield surface rate-dependent?
        ├─ YES: f(σ, κ, κ̇) → Full consistency model
        └─ NO: f(σ, κ) → Standard rate-independent plasticity
```

## Pattern Matching Guide

### Input Recognition

**User provides**: `ε̇ᵖ = γ·(f/σ₀)ᴺ·∂f/∂σ`
→ **Use**: Perzyna template with power law

**User provides**: `f = J₂ - (σᵧ + H·κ + m·κ̇)`
→ **Use**: Consistency template

**User provides**: `ε̇ᵖ = (σ - σ̄)/η` or mentions "relaxation time"
→ **Use**: Duvaut-Lions template

**User provides**: Multiple coupled rate equations
→ **Use**: General multi-variable template

## Code Structure Patterns

### Minimal Working Example (Perzyna)

```python
def residual(dlam):
    sig = sig_old + C @ (deps - dlam * n)
    f = J2(sig) - sig_y
    phi = max(0, f/sig_y)**N
    return dlam - eta * dt * phi

dlam = newton_solve(residual, x0=0.0)
eps_p_new = eps_p_old + dlam * n
sig_new = sig_old + C @ (deps - dlam * n)
```

### Minimal Working Example (Consistency)

```python
def residual(dlam):
    lam_new = lam_old + dlam
    lam_dot = dlam / dt
    sig = sig_old + C @ (deps - dlam * n)
    return J2(sig) - (sig_y + H*lam_new + m*lam_dot)

dlam = newton_solve(residual, x0=0.0)
```

## Common Equations Reference

### Von Mises Yield Function
```python
def f(sig):
    s_dev = sig - tr(sig)/3 * I
    J2 = sqrt(0.5 * s_dev : s_dev)
    return sqrt(3*J2) - sig_y
    # OR equivalently:
    return J2 - sig_y  # if using J2 directly
```

### Flow Direction (∂f/∂σ)
```python
def df_dsig(sig):
    p = tr(sig)/3
    s = sig - p*I
    J2 = sqrt(1.5 * s:s)
    return (3/(2*J2)) * s  # normalized deviatoric stress
```

### Elastic Predictor
```python
sig_trial = sig_n + C @ (eps - eps_n)
f_trial = yield_function(sig_trial)
```

### Backward Euler Discretization
```
Continuous: ẋ = g(x, t)
Discrete:   (x_{n+1} - x_n)/Δt = g(x_{n+1}, t_{n+1})
Residual:   R = x_{n+1} - x_n - Δt·g(x_{n+1}, t_{n+1})
```

## JAX-Specific Patterns

### Conditional Branching
```python
def elastic(operand):
    return sig_trial, eps_p_old, ...

def plastic(operand):
    # solve for dlam
    return sig_new, eps_p_new, ...

is_plastic = (f_trial > 0.0)
sig, eps_p, ... = jax.lax.cond(is_plastic, plastic, elastic, operand)
```

### Newton Solver
```python
from jax_newton_solver import JAXNewton

def residual(x):
    # ... compute R(x) ...
    return R

newton = JAXNewton()
newton.set_residual(residual)
x_solution, info = newton.solve(x_initial)
```

### Regularization for Division by Zero
```python
# Bad: division by zero when J2 = 0
n = 3/(2*J2) * s_dev

# Good: regularized
J2_safe = jnp.maximum(J2, 1e-12)
n = 3/(2*J2_safe) * s_dev

# Better: stop_gradient trick for smooth AD
J2_phys = jnp.sqrt(val)
J2_reg = jnp.sqrt(val + 1e-16)
J2 = jax.lax.stop_gradient(J2_phys - J2_reg) + J2_reg
```

## Debugging Checklist

Common errors and solutions:

| Error | Likely Cause | Solution |
|-------|-------------|----------|
| Newton doesn't converge | Wrong sign in residual | Check: R = Δλ - ... (not ... - Δλ) |
| Stress explodes | Flow direction wrong | Verify n = ∂f/∂σ evaluated correctly |
| No plastic flow | Yield check wrong | Ensure f_trial > 0 triggers plastic |
| Division by zero | σ_eq = 0 | Add regularization: max(σ_eq, 1e-12) |
| State not updating | Missing state update | Add state["var"] = new_value |
| Wrong units | Time/viscosity mismatch | Check: [η] = stress·time |

## Material Parameter Typical Ranges

For reference when checking if parameters are reasonable:

- **Young's modulus (E)**: 10³ - 10⁵ MPa (metals), 10¹ - 10³ MPa (polymers)
- **Yield stress (σᵧ)**: 10² - 10³ MPa (metals), 10 - 100 MPa (polymers)
- **Fluidity (γ)**: 10⁻⁶ - 10⁻² MPa⁻¹·s⁻¹
- **Viscosity (η or τ)**: 10⁻⁴ - 10 MPa·s
- **Power law exponent (N)**: 1 - 5 (typical), 1 = linear viscosity
- **Hardening modulus (H)**: -E/10 to E/10 (negative = softening)

## Time Step Guidelines

For stable implicit integration:

- **Quasi-static**: Δt can be large (0.01 - 1.0 s)
- **Dynamic**: Δt < Δt_critical = h/c_wave (CFL condition)
- **Very viscous**: Δt << η/E for accuracy
- **Rate-dependent**: Δt affects results; check convergence

## File Organization

Typical structure for material model implementation:

```
material_model.py
├── Material dataclass (parameters)
├── Model class
│   ├── __init__
│   ├── property definitions
│   ├── helper methods (_deviatoric, _J2, etc.)
│   ├── yield_function
│   ├── df_dsigma (and d2f_dsigma2 if needed)
│   └── constitutive_update (main algorithm)
│       ├── elastic predictor
│       ├── elastic_update function
│       ├── plastic_update function
│       │   └── Newton solver with residual
│       ├── jax.lax.cond branching
│       └── state update
```
