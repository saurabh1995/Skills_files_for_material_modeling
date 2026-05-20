# Common Mistakes and Corrections for Explicit Integration Schemes

This file documents critical patterns to AVOID and patterns to FOLLOW when generating explicit integration schemes for viscoplastic constitutive models with damage and hardening.

---

## ❌ CRITICAL ERROR 1: Rate Computation Sequencing in Forward Euler

### The Mistake

**NEVER compute rates at the beginning of the plastic update from trial state and use them immediately in the same step.**

This violates the forward Euler integration pattern where rates should be:
1. Stored from the previous step
2. Used to update the current state
3. Recomputed from the UPDATED state
4. Stored for the NEXT step

### ❌ Wrong Pattern (DO NOT USE):

```python
def _plastic_update(operand):
    (eps, eps_old, sig_old, eps_I_old, eps_e_old, X_old,
     p_old, D_old, R_old, deps, sig_trial, sig_eff, sig_eff_dev, J2_eff, fy, dt,
     p_dot_old, eps_I_dot_old) = operand
    
    # WRONG: Computing p_dot from TRIAL state fy
    x = fy / params.K  # fy is from sig_trial - X_old
    bracket = 0.5 * (x + jnp.abs(x))
    p_dot = jnp.power(bracket, params.m)  # NEW p_dot from trial
    
    # WRONG: Using this freshly computed p_dot immediately
    dp = p_dot * dt
    p_new = p_old + dp
    
    # WRONG: Computing flow from trial state
    inv_J2 = jnp.where(J2_eff > 0.0, 1.0 / J2_eff, 0.0)
    flow_dir = sig_eff_dev * inv_J2  # From trial
    eps_I_dot = 1.5 * p_dot * flow_dir
    
    # Using these to update state
    delta_eps_I = eps_I_dot * dt
    eps_I_new = eps_I_old + delta_eps_I
    delta_eps_e = deps - delta_eps_I
    eps_e_new = eps_e_old + delta_eps_e
    
    # Damage evolution with wrong p_dot
    D_dot = (params.alpha / (1.0 - D_old)) * (sigma_eq_old / params.sigma_0) * p_dot
    D_new = D_old + D_dot * dt
    
    # Update stress
    sig_new = sig_old + (1.0 - D_new) * (C @ delta_eps_e)
    
    # Update hardening with wrong p_dot
    R_dot = params.c1 * (params.R_inf - R_old) * p_dot
    R_new = R_old + R_dot * dt
    X_dot = (2.0/3.0) * params.C_kin * eps_I_dot - params.gamma * X_old * p_dot
    X_new = X_old + X_dot * dt
    
    # FINALLY recalculate fy (but p_dot is already wrong!)
    sig_eff_new = sig_new - X_new
    J2_eff_new = self._J2(self._deviatoric(sig_eff_new))
    fy_new = J2_eff_new / (1.0 - D_new) - (R_new + params.sigma_0)
    
    # Return p_dot that is INCONSISTENT with fy_new
    return (..., fy_new, p_dot, ...)  # WRONG! Different states
```

**Why this is wrong:**
- p_dot is computed from trial state (sig_trial, X_old, R_old, D_old)
- This p_dot is used to update the current step's state
- Then fy_new is recalculated from updated state (sig_new, X_new, R_new, D_new)
- **Result:** p_dot (from trial) and fy_new (from updated) are inconsistent
- In the next time step, we use the wrong p_dot with the new state → compounding error
- The hardening evolution rates (R_dot, X_dot) are also wrong

### ✅ Correct Pattern (ALWAYS USE):

```python
def _plastic_update(operand):
    # Extract rates from PREVIOUS step (passed in operand)
    (eps, eps_old, sig_old, eps_I_old, eps_e_old, X_old,
     p_old, D_old, R_old, deps, sig_trial, sig_eff, sig_eff_dev, J2_eff, fy, dt,
     p_dot, eps_I_dot, D_dot, R_dot, X_dot) = operand
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ALL rates from PREVIOUS step
    
    # ✅ STEP 1: Use rates from PREVIOUS step to compute increments
    dp = p_dot * dt  # p_dot from previous step
    p_new = p_old + dp
    
    D_new = D_old + D_dot * dt  # D_dot from previous step
    D_new = jnp.clip(D_new, 0.0, params.Dc)
    
    delta_eps_I = eps_I_dot * dt  # eps_I_dot from previous step
    eps_I_new = eps_I_old + delta_eps_I
    
    # ✅ STEP 2: Update strains
    delta_eps_e = deps - delta_eps_I
    eps_e_new = eps_e_old + delta_eps_e
    eps = eps_e_new + eps_I_new
    
    # ✅ STEP 3: Update stress with NEW damage
    sig_new = sig_old + (1.0 - D_new) * (C @ delta_eps_e)
    
    # ✅ STEP 4: Update hardening variables (uses rates from previous step)
    R_new = R_old + R_dot * dt
    X_new = X_old + X_dot * dt
    
    # ✅ STEP 5: NOW compute NEW rates from UPDATED state
    # Compute NEW effective stress
    sig_eff_new = sig_new - X_new  # Use NEW stress and backstress
    sig_eff_dev_new = self._deviatoric(sig_eff_new)
    J2_eff_new = self._J2(sig_eff_dev_new)
    
    # Compute NEW yield function from UPDATED state
    denom_D_new = 1.0 - D_new
    sigma_eff_new = J2_eff_new / denom_D_new
    fy_new = sigma_eff_new - (R_new + params.sigma_0)
    
    # Compute NEW p_dot from NEW yield function
    x = fy_new / params.K  # Using NEW fy
    bracket = 0.5 * (x + jnp.abs(x))
    p_dot = jnp.power(bracket, params.m)  # NEW p_dot for NEXT step
    
    # ✅ STEP 6: Compute NEW flow direction from UPDATED stress
    inv_J2 = jnp.where(J2_eff_new > 0.0, 1.0 / J2_eff_new, 0.0)
    flow_dir = sig_eff_dev_new * inv_J2  # Using NEW deviatoric stress
    eps_I_dot = 1.5 * p_dot * flow_dir  # NEW rate for NEXT step
    
    # ✅ STEP 7: Compute NEW evolution rates from UPDATED variables
    R_dot = params.c1 * (params.R_inf - R_new) * p_dot  # NEW R in equation
    X_dot = (2.0/3.0) * params.C_kin * eps_I_dot - params.gamma * X_new * p_dot
    
    # Damage rate (can use old stress for stability)
    sigma_eq = self._equiv_stress(sig_old)  # or sig_new
    denom_D_safe = jnp.maximum(1.0 - D_old, 1e-12)
    D_dot = (params.alpha / denom_D_safe) * (sigma_eq / params.sigma_0) * p_dot
    
    is_plastic_out = jnp.array(1.0, dtype=eps.dtype)
    
    # ✅ Return UPDATED state AND NEW rates for next step
    return (sig_new, eps_e_new, eps_I_new, X_new, eps, p_new, D_new, R_new,
            fy_new, p_dot, dp, D_dot, R_dot, X_dot, eps_I_dot, is_plastic_out)
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ NEW rates for NEXT step
```

**Why this is correct:**
- Uses rates from PREVIOUS step (passed in operand) for current increments
- Updates ALL state variables FIRST (stress, strains, hardening, damage)
- Computes NEW rates from fully UPDATED state
- Returns new rates to be stored in state dictionary and used in NEXT time step
- Maintains consistency: fy_new and p_dot are both computed from same updated state

---

## ❌ CRITICAL ERROR 2: Incomplete Operand Structure

### The Mistake

**NEVER omit rate variables from the operand tuple passed to jax.lax.cond.**

Both branches (_elastic_update and _plastic_update) must receive ALL rates, even if elastic branch sets them to zero.

### ❌ Wrong Pattern (DO NOT USE):

```python
# Missing D_dot, R_dot, X_dot in operand
operand = (
    eps, eps_old, sig_old, eps_I_old, eps_e_old, X_old,
    p_old, D_old, R_old, deps, sig_trial, sig_eff, sig_eff_dev, J2_eff, fy, dt,
    p_dot_old, eps_I_dot_old  # Only these two rates
)

# This causes problems in _plastic_update which needs all rates
```

### ✅ Correct Pattern (ALWAYS USE):

```python
# Before creating operand, initialize all rate variables
params = self.params
C = self.elastic_model.C

# Extract rates from previous step
p_dot = state["p_dot"][0]
eps_I_dot = state["eps_I_dot"]

# Initialize other rates (will be updated in plastic branch)
D_dot = 0.0
R_dot = 0.0
X_dot = jnp.zeros((6,), dtype=eps.dtype)  # Tensor rate
R_new = 0.0  # May be used for initialization
X_new = jnp.zeros((6,), dtype=eps.dtype)

# Complete operand with ALL rates
operand = (
    eps, eps_old, sig_old, eps_I_old, eps_e_old, X_old,
    p_old, D_old, R_old, deps, sig_trial, sig_eff, sig_eff_dev, J2_eff, fy, dt,
    p_dot, eps_I_dot, D_dot, R_dot, X_dot  # ALL rates included
)
```

---

## ❌ CRITICAL ERROR 3: Flow Direction from Wrong State

### The Mistake

**NEVER compute the flow direction for NEW rates using trial state quantities.**

The flow direction used to compute eps_I_dot for the NEXT step must be based on the UPDATED stress state.

### ❌ Wrong Pattern (DO NOT USE):

```python
# In _plastic_update, AFTER updating sig_new, X_new:

# WRONG: Using trial state J2_eff and sig_eff_dev
inv_J2 = jnp.where(J2_eff > 0.0, 1.0 / J2_eff, 0.0)  # From trial
flow_dir = sig_eff_dev * inv_J2  # From trial
eps_I_dot = 1.5 * p_dot * flow_dir  # WRONG: inconsistent

# These are from TRIAL (sig_trial - X_old), not UPDATED (sig_new - X_new)
```

### ✅ Correct Pattern (ALWAYS USE):

```python
# In _plastic_update, AFTER updating sig_new, X_new:

# ✅ Compute effective stress from UPDATED state
sig_eff_new = sig_new - X_new  # NEW stress minus NEW backstress
sig_eff_dev_new = self._deviatoric(sig_eff_new)
J2_eff_new = self._J2(sig_eff_dev_new)

# ✅ Flow direction from UPDATED state
inv_J2 = jnp.where(J2_eff_new > 0.0, 1.0 / J2_eff_new, 0.0)
flow_dir = sig_eff_dev_new * inv_J2  # From updated state
eps_I_dot = 1.5 * p_dot * flow_dir  # Consistent with updated state
```

---

## The Forward Euler Pattern

### Key Principle

In explicit forward Euler integration:

```
State(t+Δt) = State(t) + Rate(t) × Δt
Rate(t+Δt) = f(State(t+Δt))  ← Computed for NEXT step
```

**NOT:**
```
Rate(t) = f(Trial_State)  ← WRONG: from trial
State(t+Δt) = State(t) + Rate(t) × Δt
```

### Implementation Sequence

**Correct order in _plastic_update:**

1. **Use previous rates** → Compute increments (dp, delta_eps_I, etc.)
2. **Apply increments** → Update state variables (p_new, D_new, eps_I_new, etc.)
3. **Update stress** → sig_new = sig_old + (1-D_new) × C : delta_eps_e
4. **Update hardening** → R_new = R_old + R_dot × dt, X_new = X_old + X_dot × dt
5. **Compute new fy** → From sig_new, X_new, R_new, D_new
6. **Compute new rates** → p_dot, eps_I_dot, R_dot, X_dot, D_dot from updated state
7. **Return everything** → Updated state AND new rates

---

## State Variable Management

### Internal State Variables Declaration

```python
@property
def internal_state_variables(self):
    return {
        # Primary variables
        "p": 1,           # Equivalent plastic strain (scalar)
        "D": 1,           # Damage (scalar)
        "R": 1,           # Isotropic hardening (scalar)
        "eps_I": 6,       # Inelastic strain tensor (Voigt)
        "eps_e": 6,       # Elastic strain tensor (Voigt)
        "X": 6,           # Backstress tensor (Voigt)
        
        # Diagnostic/rate variables
        "fy": 1,          # Yield function value
        "p_dot": 1,       # Plastic strain rate (for NEXT step)
        "dp": 1,          # Plastic strain increment (current step)
        "D_dot": 1,       # Damage rate (for NEXT step)
        "R_dot": 1,       # Isotropic hardening rate (for NEXT step)
        "X_dot": 6,       # Backstress rate (for NEXT step)
        "eps_I_dot": 6,   # Inelastic strain rate (for NEXT step)
        "is_plastic": 1   # Plasticity flag
    }
```

### State Dictionary Updates

**CRITICAL: Update ALL state variables after jax.lax.cond:**

```python
# After conditional branching
(sig_new, eps_e_new, eps_I_new, X_new, eps, p_new, D_new, R_new,
 fy, p_dot, dp, D_dot, R_dot, X_dot, eps_I_dot, is_plastic_out) = jax.lax.cond(
    is_plastic,
    _plastic_update,
    _elastic_update,
    operand,
)

# Update ALL state variables (even if unchanged)
state["Strain"] = eps
state["Stress"] = sig_new
state["eps_I"] = eps_I_new
state["eps_e"] = eps_e_new
state["X"] = X_new

# Scalars wrapped in arrays
state["p"] = jnp.array([p_new])
state["D"] = jnp.array([D_new])
state["R"] = jnp.array([R_new])
state["fy"] = jnp.array([fy])
state["p_dot"] = jnp.array([p_dot])  # For NEXT step
state["dp"] = jnp.array([dp])
state["D_dot"] = jnp.array([D_dot])  # For NEXT step
state["R_dot"] = jnp.array([R_dot])  # For NEXT step
state["X_dot"] = X_dot  # For NEXT step
state["eps_I_dot"] = eps_I_dot  # For NEXT step
state["is_plastic"] = jnp.array([is_plastic_out])
```

---

## Checklist for Explicit Integration Schemes

Before delivering code, verify:

### Operand Structure
- [ ] All rates from previous step are in operand (p_dot, eps_I_dot, D_dot, R_dot, X_dot)
- [ ] Both _elastic_update and _plastic_update accept same operand signature
- [ ] Rates are extracted from state dictionary before creating operand

### Plastic Update Sequence
- [ ] Step 1: Use previous step rates to compute increments
- [ ] Step 2: Update all primary variables (p, D, eps_I, eps_e, R, X)
- [ ] Step 3: Update stress with new damage
- [ ] Step 4: Compute new fy from updated (sig_new, X_new, R_new, D_new)
- [ ] Step 5: Compute new p_dot from new fy
- [ ] Step 6: Compute new flow direction from updated (sig_new, X_new)
- [ ] Step 7: Compute all new rates (eps_I_dot, R_dot, X_dot, D_dot)
- [ ] Step 8: Return updated state AND new rates

### Consistency Checks
- [ ] fy and p_dot computed from same state (not trial vs updated)
- [ ] Flow direction uses updated stress state for new rates
- [ ] All hardening evolution uses updated variables (R_new, X_new, not old)
- [ ] Damage clamped to [0, Dc] range

### State Updates
- [ ] ALL state variables updated in state dictionary
- [ ] Scalars wrapped in arrays
- [ ] Rates stored for next step

---

## Summary

**The Golden Rule:**
> In forward Euler explicit integration, rates are computed at the END of each step from the UPDATED state, then stored and used at the BEGINNING of the NEXT step.

**Never:**
- Compute rates from trial state and use them in the same step
- Mix states when computing rates (e.g., p_dot from trial fy but using with updated stress)
- Omit rates from operand structure
- Use trial state quantities for computing new rates

**Always:**
- Use previous step's rates → compute increments
- Update all variables → compute new rates from fully updated state
- Return new rates → store for next step
- Maintain temporal consistency throughout

This maintains the correct forward Euler pattern and prevents accumulation of errors in the integration scheme.
