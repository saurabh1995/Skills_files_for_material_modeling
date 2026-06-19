---
name: elastoplasticity-integration
description: Generate return mapping algorithms for rate-independent elastoplastic constitutive models with isotropic hardening. Use when users provide yield function, hardening law, and flow rule and need a JAX-compatible implicit integration scheme (return mapping) with Newton solver. Trigger for elastoplasticity with implicit scheme, elastoplasticity, von Mises plasticity, Tresca, Drucker-Prager, Mohr-Coulomb, or custom yield surfaces with associative/non-associative flow rules.
---

# Elastoplasticity Integration Scheme Generator

Generate implicit scheme for rate-independent elastoplastic constitutive models, following the implicit backward-Euler approach with Newton-Raphson solver.

## ⚠️ CRITICAL: Read This First

**BEFORE generating any code, ALWAYS read `CLAUDE.md`** for critical implementation patterns specific to elastoplasticity return mapping.

## Core Workflow

When the user provides elastoplastic constitutive equations:

1. **Identify model components**:
   - Yield function f(σ, internal variables)
   - Hardening law (isotropic, kinematic, or mixed)
   - Flow rule (associative or non-associative)
   - Elastic behavior
   
2. **Generate elastic predictor**

3. **Generate yield check**

4. **Generate plasic corrector step**:
  

5. **Format as JAX-compatible Python code** with proper control flow


## Code Structure Pattern

The implicit integration scheme for elastoplasticity follows this structure (see `references/general-template.md` for complete annotated template): 



```python
@tangent_AD
def constitutive_update(self, eps, state, dt):
    # 1. Extract state
    eps_old = state["Strain"]
    deps = eps - eps_old
    p_old = state["p"][0]
    sig_old = state["Stress"]
    
    # 2. Elastic predictor
    C = self.elastic_model.C
    sig_el = sig_old + C @ deps
    
    # 3. Yield check
    sig_eq_el = self._equivalent_stress(sig_el)
    sig_Y_old = self.yield_stress(p_old)
    yield_criterion = sig_eq_el - sig_Y_old
    
    # 4. Normal to yield surface (von Mises specific)
    sig_dev_el = self._deviatoric(sig_el)
    n_el = sig_dev_el / jnp.clip(sig_eq_el, a_min=1e-8)
    
    # 5. Plastic strain increment function
    def deps_p(dp, yield_criterion):
        def deps_p_elastic(dp):
            return jnp.zeros(6)
        
        def deps_p_plastic(dp):
            return 3/2 * n_el * dp  # von Mises specific
        
        return jax.lax.cond(
            yield_criterion < 0.0,
            deps_p_elastic,
            deps_p_plastic,
            dp
        )
    
    # 6. Residual equation for Δp
    def r(dp):
        r_elastic = lambda dp: dp  # Trivial: Δp = 0
        r_plastic = lambda dp: sig_eq_el - 3*mu*dp - self.yield_stress(p_old + dp)
        return jax.lax.cond(
            yield_criterion < 0.0,
            r_elastic,
            r_plastic,
            dp
        )
    
    # 7. Solve for Δp using Newton solver
    newton = JAXNewton(r)
    dp, res = newton.solve(0.0)
    
    # 8. Update stress and state
    sig = sig_el - 2*mu * deps_p(dp, yield_criterion)
    
    state["Strain"] = eps
    state["p"] = jnp.array([p_old + dp])
    state["Stress"] = sig
    
    return sig, state
```


## Key Implementation Principles

### JAX Control Flow

- **Use jax.lax.cond**: Not Python if/else
- **Trivial elastic equations**: r_elastic = lambda x: x ensures x = 0
- **Differentiable everywhere**: Both branches must be differentiable

### Newton Solver Integration

- **JAXNewton from dolfinx_materials**: Fully differentiable Newton solver
- **Automatic jacobian**: Uses AD, no manual derivatives
- **Works with tangent_AD**: Compatible with automatic tangent operator

### Numerical Stability

```python
# 1. Clip equivalent stress to avoid division by zero
sig_eq_el = jnp.clip(self.equivalent_stress(sig_el), a_min=1e-8)

# 2. Safe normal computation
n_el = sig_dev_el / jnp.clip(sig_eq_el, a_min=1e-8)

# 3. Small initial guess for Newton
dp_init = 0.0  # For scalar
x_init = jnp.zeros((7,))  # For system
```

### Deviatoric Tensor Helper

```python
def _deviatoric(self, sig):
    """Deviatoric part in Voigt notation"""
    p = (sig[0] + sig[1] + sig[2]) / 3.0
    return jnp.array([
        sig[0] - p, sig[1] - p, sig[2] - p,
        sig[3], sig[4], sig[5]
    ], dtype=sig.dtype)
```

## Input Specification Format

### Required Information

1. **Yield function**:
   ```
   f(σ, p) = σ̄(σ) - R(p)
   ```
   where σ̄ is equivalent stress and R is yield strength

2. **Equivalent stress** (choose one):
   - von Mises: σ̄ = √(3/2 s:s)
   - Tresca: σ̄ = (σ₁ - σ₃)
   - Drucker-Prager: σ̄ = √J₂ + α I₁
   - Custom

3. **Hardening law**:
   ```
   R(p) = ...
   ```
   Examples:
   - Linear: R(p) = σ₀ + H·p
   - Exponential saturation: R(p) = σ₀ + (σ∞ - σ₀)(1 - e^(-bp))
   - Power law: R(p) = σ₀ + K·p^n

4. **Flow rule**:
   - Associative: ε̇^p = λ̇ ∂f/∂σ
   - Non-associative: ε̇^p = λ̇ ∂g/∂σ (specify g)

5. **Material parameters**:
   - Elastic: E, ν
   - Plasticity: σ₀, H (or other hardening parameters)

### Elastic Stiffness Tensor

**CRITICAL: The elastic stiffness tensor must ALWAYS be handled exactly as shown in `code-template.md`:**

**Step 1: Class Initialization**
```python
def __init__(self, elastic_model, params: ModelParams):
    super().__init__()
    self.elastic_model = elastic_model  # Store the elastic model
    self.params = params
    self.dt = 0.0
    # ... other initialization
```

**Step 2: Use in constitutive_update**
```python
@tangent_AD
def constitutive_update(self, eps, state, dt):
    # Extract stiffness from elastic model
    C = self.elastic_model.C
    
    # Use C in elastic predictor
    sig_trial = sig_old + (1.0 - D_old) * (C @ deps)
    # ...
```

**DO NOT:**
- Calculate C manually inside `constitutive_update` using Lamé parameters
- Use any other method to obtain the stiffness tensor
- Modify or recompute C from material parameters

**The elastic_model is passed during initialization and contains the pre-computed stiffness tensor C. Always use `self.elastic_model.C` directly.**

## General Yield Surface Features

### When to Use General Template

Use general template for:
- Tresca: f = (σ₁ - σ₃) - R(p)
- Drucker-Prager: f = √J₂ + α·I₁ - R(p)
- Mohr-Coulomb: f = (σ₁ - σ₃) + (σ₁ + σ₃)sin(φ) - c·cos(φ)
- Hosford: f = (|σ₁-σ₂|^a + |σ₂-σ₃|^a + |σ₃-σ₁|^a)^(1/a) - R(p)
- Any custom yield surface


## Internal State Variables

For isotropic hardening only:

```python
@property
def internal_state_variables(self):
    return {
        "p": 1,  # Cumulated plastic strain
    }
```

Note: Unlike viscoplasticity, we **do NOT store ε^p** for isotropic hardening since only p is needed.

For kinematic or mixed hardening, add backstress:

```python
return {
    "p": 1,      # Cumulated plastic strain
    "alpha": 6,  # Backstress (Voigt)
}
```

## Quality Checklist

Before delivering code, verify:

- [ ] Correct template chosen (von Mises specific vs general)
- [ ] JAXNewton solver imported and used correctly
- [ ] jax.lax.cond for elastic/plastic branching
- [ ] Trivial equations in elastic branch (r = x → x = 0)
- [ ] Equivalent stress clipped to avoid division by zero
- [ ] Normal computed safely
- [ ] @tangent_AD decorator present
- [ ] All state variables updated
- [ ] Initial guess for Newton solver provided
- [ ] Correct system dimension (1D for von Mises, 7D for general)

## Common Yield Surfaces

See `references/yield-surfaces.md` for:
- von Mises (J₂ plasticity)
- Tresca (maximum shear stress)
- Drucker-Prager (pressure-dependent)
- Mohr-Coulomb (friction angle)
- Hosford (anisotropic)
- Gurson (porous plasticity)

## Common Hardening Laws

See `references/hardening-laws.md` for:
- Linear hardening
- Exponential saturation
- Power law
- Voce law
- Swift law
- Combined hardening

## Resources

**Reference Priority Order:**

1. **PRIMARY: **`references/general-template.md`** - Complete annotated template with all implementation steps. **Always follow this structure exactly.**
2. **SUPPORTING:** 
    - **`references/general-template.md`** - Complete general yield surface implementation
    - **`references/yield-surfaces.md`** - Common yield surface implementations
    - **`references/hardening-laws.md`** - Common hardening law implementations
3. **`CLAUDE.md`** - Critical patterns and mistakes to avoid
references/von-mises-template.md
## Output Format

Generate complete implementation with:

1. **Parameter dataclass** with elastic and plastic parameters
2. **Material class** inheriting from JAXMaterial
3. **internal_state_variables property**
4. **Helper methods** (_deviatoric, _equivalent_stress as needed)
5. **constitutive_update method** with:
   - Elastic predictor
   - Yield check
   - Conditional branching with jax.lax.cond
   - Newton solver for return mapping
   - State update
6. **Inline comments** explaining each step

Match code style from dolfinx_materials examples.
