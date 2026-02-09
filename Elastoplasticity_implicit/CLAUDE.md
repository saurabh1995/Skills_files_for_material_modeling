--- Common Mistakes and Corrections for Elastoplasticity Return Mapping

This file documents critical patterns specific to rate-independent elastoplasticity with implicit algorithms.

---

## ❌ CRITICAL ERROR 1: Confusing Explicit and Implicit Integration

### The Mistake

**NEVER use forward Euler (explicit) patterns for rate-independent elastoplasticity.**

Elastoplasticity is rate-INDEPENDENT, requiring implicit backward-Euler integration with exact satisfaction of f = 0.

### ❌ Wrong Pattern (DO NOT USE):

```python
# WRONG: Treating like viscoplasticity
def constitutive_update(self, eps, state, dt):
    # Extracting rates from previous step (WRONG for elastoplasticity)
    p_dot = state["p_dot"][0]
    eps_p_dot = state["eps_p_dot"]
    
    # Computing increment with rate (WRONG)
    dp = p_dot * dt
    delta_eps_p = eps_p_dot * dt
    
    # No solver (WRONG - must enforce f = 0 exactly)
    sig_new = sig_old + C @ (deps - delta_eps_p)
```

**Why this is wrong:**
- Elastoplasticity has NO rate dependency
- Must satisfy f(σ, p) = 0 EXACTLY (consistency condition)
- Requires Newton solver, not explicit update
- dt is often not even relevant (quasi-static)

### ✅ Correct Pattern (ALWAYS USE):

```python
# CORRECT: Implicit return mapping
def constitutive_update(self, eps, state, dt):
    # NO rate variables extracted
    p_old = state["p"][0]
    sig_old = state["Stress"]
    
    # Elastic predictor
    sig_el = sig_old + C @ deps
    
    # Check yield
    yield_criterion = f(sig_el, p_old)
    
    # Define residual equation
    def r(dp):
        r_elastic = lambda dp: dp  # Trivial: dp = 0
        r_plastic = lambda dp: sig_eq_el - 3*mu*dp - R(p_old + dp)  # f = 0
        return jax.lax.cond(yield_criterion < 0.0, r_elastic, r_plastic, dp)
    
    # SOLVE using Newton (implicit)
    newton = JAXNewton(r)
    dp, res = newton.solve(0.0)
    
    # Update
    sig = sig_el - 2*mu * deps_p(dp)
    state["p"] = jnp.array([p_old + dp])
```

---

## ❌ CRITICAL ERROR 2: Wrong Branching for Elastic Case

### The Mistake

**NEVER use complex logic for elastic branch - use trivial equations that give zero solution.**

### ❌ Wrong Pattern (DO NOT USE):

```python
# WRONG: Complex elastic handling
def r(dp):
    if yield_criterion < 0.0:  # WRONG: Python if
        return 0.0  # WRONG: Constant
    else:
        return sig_eq_el - 3*mu*dp - R(p_old + dp)

# Or worse:
def constitutive_update(self, eps, state, dt):
    if yield_criterion < 0.0:  # WRONG: Breaks JAX
        dp = 0.0
        sig = sig_el
    else:
        # solve for dp
```

**Why this is wrong:**
- Python if/else breaks JAX JIT compilation
- Returning constants (0.0) breaks differentiability
- Not compatible with tangent_AD

### ✅ Correct Pattern (ALWAYS USE):

```python
# CORRECT: Trivial equation for elastic case
def r(dp):
    r_elastic = lambda dp: dp  # Trivial: dp = 0
    r_plastic = lambda dp: sig_eq_el - 3*mu*dp - R(p_old + dp)
    
    return jax.lax.cond(
        yield_criterion < 0.0,
        r_elastic,  # Solves dp = 0 → dp = 0
        r_plastic,  # Solves f = 0
        dp
    )
```

**Why this is correct:**
- r_elastic(dp) = dp means Newton solves dp = 0 → dp = 0
- Both branches are differentiable functions
- Works with JAX control flow
- Compatible with tangent_AD

---

## ❌ CRITICAL ERROR 3: Storing Plastic Strain for Isotropic Hardening

### The Mistake

**NEVER store ε^p as internal state variable for isotropic hardening only - just store p.**

### ❌ Wrong Pattern (DO NOT USE):

```python
# WRONG: Storing unnecessary variables
@property
def internal_state_variables(self):
    return {
        "p": 1,       # Cumulated plastic strain
        "eps_p": 6,   # WRONG: Not needed for isotropic hardening
        "eps_p_dot": 6,  # WRONG: No rates in elastoplasticity
    }
```

**Why this is wrong:**
- For isotropic hardening, only p appears in R(p)
- ε^p is NOT needed to compute anything
- Wastes memory and computation
- Rate variables don't exist in rate-independent plasticity

### ✅ Correct Pattern (ALWAYS USE):

```python
# CORRECT: Minimal state for isotropic hardening
@property
def internal_state_variables(self):
    return {
        "p": 1,  # Only this is needed
    }

# For kinematic hardening, add backstress:
@property
def internal_state_variables(self):
    return {
        "p": 1,      # Cumulated plastic strain
        "alpha": 6,  # Backstress (needed in yield function)
    }
```

---

## ❌ CRITICAL ERROR 4: Using Trial State for Flow Direction (General Case)

### The Mistake

**NEVER use elastic predictor stress for computing flow direction in general return mapping.**

For general yield surfaces, flow direction n depends on FINAL stress, not trial stress.

### ❌ Wrong Pattern (DO NOT USE):

```python
# WRONG: Computing normal from trial stress
def constitutive_update(self, eps, state, dt):
    sig_el = sig_old + C @ deps
    
    # WRONG: Normal from elastic predictor
    n_el = gradient_of_f(sig_el)
    
    # WRONG: Using this fixed normal
    def r_eps_p(dx):
        deps_p = dx[:-1]
        dp = dx[-1]
        return deps_p - n_el * dp  # WRONG: n_el is from trial
```

**Why this is wrong:**
- For general surfaces, n changes as stress returns to yield surface
- n must be evaluated at final stress σ_n+1, not trial σ_el
- Only works for von Mises (special case)

### ✅ Correct Pattern (ALWAYS USE):

```python
# CORRECT: Normal from updated stress
def constitutive_update(self, eps, state, dt):
    sig_el = sig_old + C @ deps
    
    # Function to compute stress from plastic strain increment
    def stress(deps_p):
        return sig_old + C @ (deps - deps_p)
    
    # Normal computed via AD
    normal = jax.jacfwd(self.equivalent_stress)
    
    def r_eps_p(dx):
        deps_p = dx[:-1]
        dp = dx[-1]
        
        # CORRECT: Compute stress, then normal
        sig = stress(deps_p)
        n = normal(sig)  # Normal at CURRENT stress
        
        return deps_p - n * dp  # Correct flow rule
```

**Exception:** For von Mises ONLY, can use:
```python
# von Mises specific (explicit return mapping)
n_el = s_el / sig_eq_el  # From elastic predictor
deps_p = 3/2 * n_el * dp  # This works ONLY for von Mises
```

---

## ❌ CRITICAL ERROR 5: Wrong System Dimension for Newton Solver

### The Mistake

**NEVER use wrong dimension for initial guess in general return mapping.**

### ❌ Wrong Pattern (DO NOT USE):

```python
# WRONG: Wrong dimension
newton = JAXNewton((r_eps_p, r_p))
x0 = jnp.zeros((6,))  # WRONG: Should be 7 (6 for Δε^p + 1 for Δp)
x, res = newton.solve(x0)
```

### ✅ Correct Pattern (ALWAYS USE):

```python
# For von Mises (1D scalar problem):
newton = JAXNewton(r)  # Single residual function
dp_init = 0.0  # Scalar initial guess
dp, res = newton.solve(dp_init)

# For general surface (7D system problem):
newton = JAXNewton((r_eps_p, r_p))  # Tuple of residuals
x0 = jnp.zeros((7,))  # 6 for Δε^p + 1 for Δp
x, res = newton.solve(x0)

deps_p = x[:-1]  # First 6 components
dp = x[-1]  # Last component
```

---

## ❌ CRITICAL ERROR 6: Not Clipping Equivalent Stress

### The Mistake

**NEVER divide by equivalent stress without clipping to avoid division by zero.**

### ❌ Wrong Pattern (DO NOT USE):

```python
# WRONG: Division by zero risk
sig_eq_el = jnp.sqrt(3/2 * jnp.sum(s_el**2))
n_el = s_el / sig_eq_el  # WRONG: Can be 0/0
```

### ✅ Correct Pattern (ALWAYS USE):

```python
# CORRECT: Clip before division
sig_eq_el = jnp.sqrt(3/2 * jnp.sum(s_el**2))
sig_eq_el = jnp.clip(sig_eq_el, a_min=1e-8)  # Ensure > 0
n_el = s_el / sig_eq_el  # Safe

# Or in one line:
n_el = s_el / jnp.clip(sig_eq_el, a_min=1e-8)
```

---

## The Return Mapping Pattern

### Key Principle

For rate-independent elastoplasticity:

```
Elastic predictor → Check f ≤ 0 → If f > 0, solve f(σ_n+1, p_n+1) = 0
```

**NOT:**
```
Compute rate → Integrate → Hope f ≈ 0  (This is viscoplasticity)
```


**General:**

1. Compute elastic predictor σ_el
2. Check yield f_el
3. If f_el ≤ 0: elastic, Δε^p = 0, Δp = 0
4. Else: Solve system:
   - Δε^p = Δp · ∇f(σ(Δε^p))
   - σ̄(σ(Δε^p)) - R(p_old + Δp) = 0
5. Update σ = σ_old + C:(Δε - Δε^p)

---

## Checklist for Elastoplasticity

Before delivering code, verify:

### Model Type
- [ ] Identified as von Mises specific OR general yield surface
- [ ] Used correct template (explicit vs implicit return mapping)

### Implementation
- [ ] NO rate variables (p_dot, eps_p_dot, etc.)
- [ ] NO time step dt used in plastic flow (may be in code signature but not used)
- [ ] JAXNewton solver imported and used
- [ ] jax.lax.cond for elastic/plastic branching
- [ ] Trivial equations in elastic branch

### Numerical Stability
- [ ] Equivalent stress clipped before division
- [ ] Normal computed safely
- [ ] Proper initial guess for Newton (0.0 for 1D, zeros(7) for system)

### State Management
- [ ] Only p stored for isotropic hardening
- [ ] NO ε^p stored unless kinematic hardening present
- [ ] NO rate variables stored

---

## Summary

**The Golden Rule:**
> Elastoplasticity is RATE-INDEPENDENT. Use implicit backward-Euler with Newton solver to enforce f = 0 exactly. DO NOT use explicit forward-Euler or rate variables.

**Never:**
- Use forward Euler integration
- Store or use rate variables (p_dot, etc.)
- Use Python if/else for branching
- Use trial stress normal for general surfaces (except von Mises)
- Forget to clip equivalent stress

**Always:**
- Use backward-Euler (implicit)
- Solve f = 0 with Newton solver
- Use jax.lax.cond with trivial elastic equations
- Compute normal from updated stress (general case)
- Use @tangent_AD decorator

**von Mises Special Case:**
- Can use explicit return mapping
- Normal from elastic predictor is correct
- Much faster than general case

This maintains correct implicit integration for rate-independent plasticity.
