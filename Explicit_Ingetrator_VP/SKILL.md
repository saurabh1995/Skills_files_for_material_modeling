---
name: explicit-integration-scheme
description: Generate explicit time integration schemes for constitutive models. Use when users provide constitutive equations (stress-strain relations, damage evolution, hardening laws, flow rules) and need a FEniCSx-JAX-compatible explicit integration algorithm in the style of Lemaitre-Chaboche or Perzyna viscoplasticity with elastic predictor-plastic corrector structure. Trigger when user asks to create explicit scheme, integration algorithm, or material model implementation from equations.
---

# Explicit Integration Scheme Generator

Generate explicit time integration schemes for constitutive material models, following the radial return mapping algorithm pattern.

## Core Workflow

When the user provides constitutive equations, follow this sequence:
0. **Review common mistakes** - Consult `claude.md` to avoid known error patterns


1. **Identify equation types** in the provided model:
   - Strain decomposition (elastic + inelastic)
   - Stress-strain relationship (damaged/undamaged)
   - Yield function
   - Flow rule (plastic/viscoplastic strain rate)
   - Hardening evolution (isotropic and/or kinematic)
   - Damage evolution
   
2. **Generate elastic predictor step** - Trial stress assuming elastic behavior

3. **Generate yield check** - Evaluate yield function with trial stress

4. **Generate plastic corrector step** - If yielding occurs, compute:
   - Inelastic strain increment
   - Updated stress accounting for plasticity
   - Evolution of internal variables (hardening, damage)

5. **Format as FEniCSx-JAX-compatible Python code** following the reference template structure strictly

## Code Structure Pattern

The explicit integration scheme follows this structure (see `references/code-template.md` for complete annotated template):

```python
@tangent_AD
def constitutive_update(self, eps, state, dt):
    # 1. EXTRACT OLD STATE VARIABLES
    eps_old = state["Strain"]
    deps = eps - eps_old  # total strain increment
    sig_old = state["Stress"]
    # ... extract other internal variables
    
    # 2. ELASTIC PREDICTOR (Algorithm 1, line 2)
    C = self.elastic_model.C
    sig_trial = sig_old + (1.0 - D_old) * (C @ deps)
    
    # 3. YIELD CHECK (Algorithm 1, line 7)
    fy = [yield function evaluation with sig_trial]
    
    # 4. PREPARE OPERAND FOR BRANCHING
    operand = (eps, eps_old, sig_old, ..., deps, sig_trial, fy, dt, ...)
    
    # 5. DEFINE ELASTIC UPDATE BRANCH
    def _elastic_update(operand):
        # Unpack operand
        # Return elastic response (no plasticity) 
        # Set plastic-related rates to zero
        return sig_new, eps_e_new, ..., is_plastic_out
    
    # 6. DEFINE PLASTIC UPDATE BRANCH
    def _plastic_update(operand):
        # Unpack operand
        # Compute plastic strain increment
        # Update stress accounting for plasticity
        # Evolve internal variables (R, X, D)
        # Recalculate rates for next step
        return sig_new, eps_e_new, ..., is_plastic_out
    
    # 7. CONDITIONAL BRANCHING 
    is_plastic = fy > 0.0
    sig_new, ... = jax.lax.cond(
        is_plastic,
        _plastic_update,
        _elastic_update,
        operand
    )
    
    # 8. UPDATE STATE DICTIONARY
    state["Strain"] = eps
    state["Stress"] = sig_new
    state["p"] = jnp.array([p_new])
    # ... update ALL state variables
    
    return sig_new, state
```

## Key Implementation Principles

### JAX Compatibility Requirements

- **No Python conditionals**: Use `jax.lax.cond` for if/else, `jax.lax.switch` for multiple branches
- **Use jnp not np**: All operations must use `jax.numpy` (imported as `jnp`)
- **No in-place mutations**: Create new arrays instead of modifying existing ones
- **Decorator required**: Mark with `@tangent_AD` for automatic differentiation support
- **Pure functions**: Inner functions (_elastic_update, _plastic_update) must be pure

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

### State Management

- **Pre-declare all variables**: Every internal variable must be listed in `internal_state_variables` property with its shape
- **Update all state every step**: Even if unchanged, set state variables to their current value
- **Consistent tensor notation**: Use Voigt notation [s11, s22, s33, s12, s13, s23] for 3D stress/strain
- **Scalar wrapping**: Scalar internal variables stored as length-1 arrays: `state["p"] = jnp.array([p_new])`

### Numerical Stability Techniques

Use these patterns from the reference code:

```python
# 1. Regularized square root for J2 invariant
eps_reg = 1e-16
J2_phys = jnp.sqrt(jnp.maximum(val, 0.0))
J2_reg = jnp.sqrt(val + eps_reg)
J2 = jax.lax.stop_gradient(J2_phys - J2_reg) + J2_reg

# 2. Safe division with where clause
inv_J2 = jnp.where(J2_eff > 0.0, 1.0 / J2_eff, 0.0)

# 3. McCauley brackets for viscoplasticity
bracket = 0.5 * (x + jnp.abs(x))  # Returns max(x, 0)

# 4. Clipping damage to physical range
D_new = jnp.clip(D_new, 0.0, params.Dc)
```

### Variable Naming Conventions

Follow Algorithm 1 naming from Tandale et al. paper:

| Variable | Meaning | State Key |
|----------|---------|-----------|
| `sig_trial` or `S_hat` | Trial stress (elastic predictor) | - |
| `deps` or `Delta_E` | Total strain increment | - |
| `deps_I` or `Delta_E_I` | Inelastic strain increment | - |
| `eps_I_dot` | Inelastic strain rate | "eps_I_dot" |
| `p` | Equivalent plastic strain | "p" |
| `p_dot` | Equivalent plastic strain rate | "p_dot" |
| `D` | Damage parameter (0 to Dc) | "D" |
| `D_dot` | Damage rate | "D_dot" |
| `R` | Isotropic hardening variable | "R" |
| `R_dot` | Isotropic hardening rate | "R_dot" |
| `X` | Backstress (kinematic hardening) | "X" |
| `X_dot` | Backstress rate | "X_dot" |
| `fy` | Yield function value | "fy" |

## Input Specification Format

For comprehensive implementation guidance including step-by-step code generation and detailed examples, see `references/code-template.md` and `references/equation-mapping.md`.

Request users to provide equations in this structured format:

### Required Constitutive Equations

1. **Strain decomposition**
2. **Stress-strain law**
3. **Yield function**
4. **Flow rule**
5. **Consistency parameter** (plastic/viscoplastic strain rate)

### Optional Evolution Laws

6. **Isotropic hardening**
7. **Kinematic hardening**
8. **Damage evolution**

### Material Parameters

Request all material constants with units.

## Plane Stress Considerations

The user's uploaded code does NOT include plane stress iteration (Algorithm 1, lines 14-32 are skipped). 

**If plane stress σ₃₃ = 0 is NOT required:** Use the standard implementation pattern.

**If plane stress IS required in the future:** See `references/plane-stress-extension.md` for implementation details.

## Quality Checklist

Before delivering the code, verify:

- [ ] All equations provided by user are implemented
- [ ] JAX compatibility (jnp, jax.lax.cond, no Python if/else)
- [ ] All internal variables declared in `internal_state_variables`
- [ ] All state variables updated in state dictionary
- [ ] Numerical stability techniques applied (regularized sqrt, safe division)
- [ ] Variable naming follows conventions
- [ ] Code style matches user's template (indentation, comments)
- [ ] Helper methods included if needed
- [ ] Comments reference equation numbers from user's input
- [ ] @tangent_AD decorator present

## Resources

**Reference Priority Order:**

1. **PRIMARY: `references/code-template.md`** - Complete annotated template with all implementation steps. **Always follow this structure exactly.**
2. **SECONDARY: `references/equation-mapping.md` - Maps common constitutive equations to code
3. **SUPPORTING:** 
   - `references/example-models.md`** - Full implementations of classic models for reference examples
   - `references/plane-stress-extension.md` - Plane stress constraint satisfaction (if needed)
   - **`claude.md`** - Common mistakes and anti-patterns to avoid

**Code structure must match `code-template.md` exactly.** Use `example-models.md` only for seeing complete working examples, not for structure.