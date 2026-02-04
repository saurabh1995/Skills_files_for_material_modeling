# Plane Stress Extension

This reference describes how to extend the explicit integration scheme to satisfy the plane stress condition (σ₃₃ = 0) for plate and shell structures.

**Note:** The user's uploaded code does NOT currently implement plane stress iteration. This is for future reference only.

## Background

In 3D solid mechanics, all six stress components are non-zero. However, for thin plates and shells, the plane stress assumption applies:

```
σ₃₃ = 0  (out-of-plane normal stress)
```

This constraint must be satisfied iteratively in the plastic regime because the plastic strain increment Δε₃₃ is unknown.

## Algorithm Extension

From Algorithm 1 (Tandale paper, lines 14-32), the plane stress iteration is:

```python
# Inside plastic corrector step:
while |σ₃₃| > tolerance:
    # 1. Compute Δε₃₃ using secant method
    if iteration == 0:
        # Initial guess (elastic solution)
        Δε₃₃⁽⁰⁾ = -(1/(λ+2G)) * [σ₃₃ᵗ⁻¹ + λ(Δε₁₁ + Δε₂₂)]
        Δε₃₃⁽¹⁾ = -(Δε₁₁ + Δε₂₂)  # Alternative guess
    
    # 2. Compute trial values at both guesses
    σ₃₃⁽ⁱ⁾ = σ̂₃₃ - 2μ(1-D)(ΔεᴵI₃₃)⁽ⁱ⁾
    σ₃₃⁽ⁱ⁺¹⁾ = σ̂₃₃ - 2μ(1-D)(ΔεᴵI₃₃)⁽ⁱ⁺¹⁾
    
    # 3. Secant update
    Δε₃₃⁽ⁱ⁺²⁾ = Δε₃₃⁽ⁱ⁾ - σ₃₃⁽ⁱ⁾ * (Δε₃₃⁽ⁱ⁺¹⁾ - Δε₃₃⁽ⁱ⁾) / (σ₃₃⁽ⁱ⁺¹⁾ - σ₃₃⁽ⁱ⁾ + 1e-7)
    
    # 4. Shift for next iteration
    Δε₃₃⁽ⁱ⁾ = Δε₃₃⁽ⁱ⁺¹⁾
    Δε₃₃⁽ⁱ⁺¹⁾ = Δε₃₃⁽ⁱ⁺²⁾
```

## Implementation Challenges with JAX

### Problem 1: While Loops

JAX requires pure functions. Python `while` loops don't work directly. Solutions:

**Option A: jax.lax.while_loop**
```python
def cond_fun(carry):
    _, sig_33, _, iteration = carry
    return (jnp.abs(sig_33) > tolerance) & (iteration < max_iter)

def body_fun(carry):
    deps_33, sig_33, sig_33_prev, iteration = carry
    
    # Secant update
    deps_33_new = compute_secant_update(...)
    sig_33_new = compute_stress_33(deps_33_new)
    
    return (deps_33_new, sig_33_new, sig_33, iteration + 1)

init_carry = (deps_33_init, sig_33_init, 0.0, 0)
final_carry = jax.lax.while_loop(cond_fun, body_fun, init_carry)
deps_33_final, _, _, _ = final_carry
```

**Option B: Fixed Number of Iterations**
```python
def scan_body(carry, _):
    deps_33, sig_33_prev = carry
    
    # Secant update
    deps_33_new = compute_secant_update(deps_33, sig_33_prev)
    sig_33_new = compute_stress_33(deps_33_new)
    
    return (deps_33_new, sig_33_new), sig_33_new

init = (deps_33_init, sig_33_init)
final, history = jax.lax.scan(scan_body, init, None, length=max_iterations)
deps_33_final, sig_33_final = final
```

### Problem 2: Nested Updates

The plane stress iteration is nested inside the plastic update:
- Outer level: Elastic vs. plastic (`jax.lax.cond`)
- Inner level: Plane stress iteration (`jax.lax.while_loop`)

This creates complex nested control flow.

## Neural Network Alternative (Algorithm 2)

Tandale et al. propose using a Legendre Memory Unit (LMU) neural network to replace the iterative solver. The NN predicts Δε₃₃ directly.

### Advantages

1. **No iterations**: NN prediction is a single forward pass
2. **Self-learning**: Physics-based loss ensures σ₃₃ = 0
3. **Faster convergence**: Typically 12-14% speedup
4. **Handles damage**: Better at difficult cases with evolving damage

### Training

Pre-train the NN with:

**Input:**
```python
x = [Δε₁₁, Δε₂₂, ΔεᴵI₃₃, D, μ]
```

**Output:**
```python
y = [Δε₃₃]
```

**Loss function (hybrid):**
```python
# Data-driven term
L_d = MSE(Δε₃₃_predicted, Δε₃₃_actual)

# Physics-based term (plane stress constraint)
L_p = |σ₃₃(Δε₃₃_predicted)|²

# Combined
L = λ₁ L_d + λ₂ L_p
```

During deployment:
- λ₁ = 0 (no data available)
- λ₂ = 1 (enforce physics only)

### Implementation

```python
# In _plastic_update:

# Initialize or load NN model
nn_model = load_lmu_model()

# Prepare NN input
x_nn = jnp.array([deps[0], deps[1], delta_eps_I[2], D_old, params.mu])

# Predict Δε₃₃
deps_33_pred = nn_model(x_nn)

# Update total strain increment
deps_full = jnp.array([
    deps[0], deps[1], deps_33_pred,
    deps[3], deps[4], deps[5]
])

# Continue with standard plastic update using deps_full
```

## Comparison

| Approach | Iterations | Speed | Accuracy | Complexity |
|----------|------------|-------|----------|------------|
| Secant method | 5-15 | Baseline | High | Medium |
| Newton-Raphson | 3-8 | Faster | High | High |
| Neural network | 0-1 | Fastest | High | Very high |

## Recommendation

**For most users:** Don't implement plane stress unless specifically needed. The 3D formulation is simpler and more general.

**If plane stress is required:**
1. Start with fixed-iteration approach (jax.lax.scan)
2. Use 10-15 iterations for safety
3. Add convergence check in post-processing

**Advanced users only:**
- Implement LMU neural network solver
- Requires training data generation
- Requires neural network expertise
- Provides significant speedup for production runs

## Code Template (Secant Method with Fixed Iterations)

```python
def _plastic_update_plane_stress(operand):
    # ... existing plastic update code ...
    
    # After computing eps_I_dot, add plane stress iteration
    
    def plane_stress_iteration(carry, _):
        deps_33, sig_33_prev = carry
        
        # Update total strain with current guess
        deps_3d = jnp.array([
            deps[0], deps[1], deps_33,
            deps[3], deps[4], deps[5]
        ])
        
        # Recompute plastic response
        delta_eps_I_local = eps_I_dot * dt
        delta_eps_e_local = deps_3d - delta_eps_I_local
        
        # Stress update
        sig_local = sig_old + (1.0 - D_new) * (C @ delta_eps_e_local)
        sig_33 = sig_local[2]
        
        # Secant update for deps_33
        if sig_33_prev != 0:
            deps_33_new = deps_33 - sig_33 * (deps_33 - deps_33_prev) / (sig_33 - sig_33_prev + 1e-12)
        else:
            deps_33_new = deps_33
        
        return (deps_33_new, sig_33), sig_33
    
    # Initial guess (elastic)
    lam = params.E * params.nu / ((1 + params.nu) * (1 - 2*params.nu))
    G = params.E / (2 * (1 + params.nu))
    deps_33_init = -(lam / (lam + 2*G)) * (sig_old[2] + lam * (deps[0] + deps[1]))
    
    # Run fixed number of iterations
    (deps_33_final, sig_33_final), history = jax.lax.scan(
        plane_stress_iteration,
        (deps_33_init, 0.0),
        None,
        length=10  # Fixed iterations
    )
    
    # Use deps_33_final for final strain update
    deps_final = jnp.array([
        deps[0], deps[1], deps_33_final,
        deps[3], deps[4], deps[5]
    ])
    
    # ... continue with rest of plastic update ...
```

## Further Reading

- Tandale et al. (2024): "Recurrent neural networks as physics-based self-learning solver..."
- Section 4: "LMU as a nonlinear solver to satisfy plane stress condition"
- Algorithm 2: "Neural Network Enhanced Explicit viscoplastic integration scheme"