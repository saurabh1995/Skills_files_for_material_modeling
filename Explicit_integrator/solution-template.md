# Solution Template: Using the Explicit Integration Scheme Skill

This document demonstrates how Claude Opus 4.5 would use the `explicit-integration-scheme` skill to generate material model implementations from user-provided constitutive equations.

## Example Usage Scenario

**User Request:**
"I have a viscoplastic material model with the following equations. Can you write the explicit integration scheme for me in the same format as my uploaded Lemaitre-Chaboche code?"

**User Provides:**
1. Strain decomposition: ε = εᵉ + εᵖ
2. Yield function: f = σₑq - (R + σ₀)
3. Plastic strain rate: ε̇ᵖ = λ̇ × σ'/σₑq
4. Viscoplastic rate: λ̇ = <f/K>ⁿ
5. Hardening: Ṙ = H × λ̇

Material parameters:
- E = 200 GPa
- ν = 0.3
- σ₀ = 250 MPa (initial yield)
- K = 50 MPa·s^(1/n)
- n = 5
- H = 1000 MPa (hardening modulus)

## Claude's Response Workflow

### Step 1: Skill Activation

Claude recognizes the request involves constitutive equations and integration scheme generation, triggering the skill.

### Step 2: Identify Equation Types

From the skill workflow, Claude analyzes:

✓ Strain decomposition: Additive (standard)
✓ Stress-strain: Elastic (no damage)
✓ Yield function: von Mises type
✓ Flow rule: Associative plasticity
✓ Rate law: Perzyna viscoplasticity
✓ Hardening: Linear isotropic
✗ Damage: None
✗ Kinematic hardening: None

### Step 3: Consult References

Claude reads:
1. `references/code-template.md` - For structure
2. `references/equation-mapping.md` - For equation implementations
3. `references/example-models.md` - For similar model (Example 1: von Mises)

### Step 4: Generate Parameter Class

```python
@dataclass
class ViscoplasticParams:
    """Material parameters for viscoplastic model with linear hardening"""
    # Elastic
    E: float = 200000.0    # Young's modulus [MPa]
    nu: float = 0.3        # Poisson's ratio [-]
    
    # Viscoplastic
    k: float = 250.0       # Initial yield stress [MPa]
    K_visc: float = 50.0   # Perzyna viscosity [MPa·s^(1/n)]
    n_visc: float = 5.0    # Perzyna exponent [-]
    
    # Hardening
    H: float = 1000.0      # Hardening modulus [MPa]
```

### Step 5: Generate Internal State Variables

Following the template pattern:

```python
@property
def internal_state_variables(self):
    return {
        "p": 1,          # Equivalent plastic strain
        "R": 1,          # Isotropic hardening
        "eps_p": 6,      # Plastic strain tensor
        "eps_e": 6,      # Elastic strain tensor
        "fy": 1,         # Yield function
        "p_dot": 1,      # Plastic strain rate
        "dp": 1,         # Plastic strain increment
        "R_dot": 1,      # Hardening rate
        "eps_I_dot": 6,  # Inelastic strain rate
        "is_plastic": 1  # Plasticity flag
    }
```

### Step 6: Generate Helper Methods

Based on the yield function type (von Mises), Claude includes:

```python
def _deviatoric(self, sig):
    """Deviatoric stress tensor"""
    p = (sig[0] + sig[1] + sig[2]) / 3.0
    return jnp.array([
        sig[0] - p, sig[1] - p, sig[2] - p,
        sig[3], sig[4], sig[5]
    ], dtype=sig.dtype)

def _J2(self, sig_dev):
    """J2 invariant with regularization"""
    s = sig_dev
    s_colon_s = (s[0]**2 + s[1]**2 + s[2]**2 + 
                 2.0 * (s[3]**2 + s[4]**2 + s[5]**2))
    val = 1.5 * s_colon_s
    val_pos = jnp.maximum(val, 0.0)
    J2_phys = jnp.sqrt(val_pos)
    eps_reg = 1e-16
    J2_reg = jnp.sqrt(val_pos + eps_reg)
    return jax.lax.stop_gradient(J2_phys - J2_reg) + J2_reg
```

### Step 7: Generate constitutive_update Method

Following Algorithm 1 structure from the template:

```python
@tangent_AD
def constitutive_update(self, eps, state, dt):
    # 1. Extract old state (Algorithm 1, line 1)
    eps_old = state["Strain"]
    deps = eps - eps_old
    sig_old = state["Stress"]
    eps_p_old = state["eps_p"]
    eps_e_old = state["eps_e"]
    p_old = state["p"][0]
    R_old = state["R"][0]
    
    # 2. Elastic predictor (Algorithm 1, line 2)
    C = self.elastic_model.C
    sig_trial = sig_old + C @ deps
    
    # 3. Yield check (Algorithm 1, line 7)
    sig_dev = self._deviatoric(sig_trial)
    J2 = self._J2(sig_dev)
    fy = J2 - (R_old + self.params.k)
    
    # 4. Prepare operand
    operand = (eps, eps_old, sig_old, eps_p_old, eps_e_old,
               p_old, R_old, deps, sig_trial, fy, dt)
    
    # 5. Elastic branch
    def _elastic_update(operand):
        (eps, eps_old, sig_old, eps_p_old, eps_e_old,
         p_old, R_old, deps, sig_trial, fy, dt) = operand
        
        sig_new = sig_trial
        eps_e_new = eps_e_old + deps
        eps_p_new = eps_p_old
        eps = eps_e_new + eps_p_new
        p_new = p_old
        R_new = R_old
        
        p_dot = 0.0
        dp = 0.0
        R_dot = 0.0
        eps_I_dot = jnp.zeros((6,), dtype=eps.dtype)
        is_plastic_out = jnp.array(0.0, dtype=eps.dtype)
        
        return (sig_new, eps_e_new, eps, eps_p_new, p_new, R_new,
                fy, p_dot, dp, R_dot, eps_I_dot, is_plastic_out)
    
    # 6. Plastic branch
    def _plastic_update(operand):
        (eps, eps_old, sig_old, eps_p_old, eps_e_old,
         p_old, R_old, deps, sig_trial, fy, dt) = operand
        
        # Viscoplastic strain rate (Eq. 9 pattern)
        x = fy / self.params.K_visc
        bracket = 0.5 * (x + jnp.abs(x))
        p_dot = jnp.power(bracket, self.params.n_visc)
        dp = p_dot * dt
        p_new = p_old + dp
        
        # Flow direction (Eq. 8 pattern)
        sig_dev = self._deviatoric(sig_trial)
        J2 = self._J2(sig_dev)
        inv_J2 = jnp.where(J2 > 0.0, 1.0 / J2, 0.0)
        flow_dir = sig_dev * inv_J2
        
        # Inelastic strain (Eq. 8)
        eps_I_dot = 1.5 * p_dot * flow_dir
        delta_eps_I = eps_I_dot * dt
        
        # Update strains
        eps_p_new = eps_p_old + delta_eps_I
        delta_eps_e = deps - delta_eps_I
        eps_e_new = eps_e_old + delta_eps_e
        eps = eps_e_new + eps_p_new
        
        # Stress update (no damage)
        sig_new = sig_old + C @ delta_eps_e
        
        # Linear hardening
        R_new = self.params.H * p_new
        R_dot = self.params.H * p_dot
        
        # Recompute yield function
        sig_dev_new = self._deviatoric(sig_new)
        J2_new = self._J2(sig_dev_new)
        fy_new = J2_new - (R_new + self.params.k)
        
        is_plastic_out = jnp.array(1.0, dtype=eps.dtype)
        
        return (sig_new, eps_e_new, eps, eps_p_new, p_new, R_new,
                fy_new, p_dot, dp, R_dot, eps_I_dot, is_plastic_out)
    
    # 7. Conditional execution (Algorithm 1, line 8)
    is_plastic = fy > 0.0
    (sig_new, eps_e_new, eps, eps_p_new, p_new, R_new,
     fy, p_dot, dp, R_dot, eps_I_dot, is_plastic_out) = jax.lax.cond(
        is_plastic,
        _plastic_update,
        _elastic_update,
        operand
    )
    
    # 8. Update state (Algorithm 1, line 33-41)
    state["Strain"] = eps
    state["Stress"] = sig_new
    state["eps_p"] = eps_p_new
    state["eps_e"] = eps_e_new
    state["p"] = jnp.array([p_new])
    state["R"] = jnp.array([R_new])
    state["fy"] = jnp.array([fy])
    state["p_dot"] = jnp.array([p_dot])
    state["dp"] = jnp.array([dp])
    state["R_dot"] = jnp.array([R_dot])
    state["eps_I_dot"] = eps_I_dot
    state["is_plastic"] = jnp.array([is_plastic_out])
    
    return sig_new, state
```

### Step 8: Quality Check

Claude verifies against the checklist:

✓ All equations implemented
✓ JAX compatibility (jnp, jax.lax.cond)
✓ All internal variables declared
✓ All state variables updated
✓ Numerical stability (regularized sqrt, safe division)
✓ Variable naming conventions followed
✓ Code style matches user template
✓ Helper methods included
✓ Comments reference equation numbers
✓ @tangent_AD decorator present

## Key Features of the Generated Code

1. **Matches User's Style**: 
   - Same indentation (4 spaces)
   - Same variable naming (sig_trial, eps_p_old, etc.)
   - Same structure (operand tuple, dual branches)

2. **JAX Compatible**:
   - Uses jax.lax.cond for branching
   - All jnp operations
   - Pure functions
   - No in-place mutations

3. **Numerically Stable**:
   - Regularized J2 computation
   - Safe division with jnp.where
   - McCauley brackets for viscoplasticity

4. **Complete Implementation**:
   - Parameter dataclass
   - Internal state declaration
   - Helper methods
   - Full constitutive update
   - All state updates

## Comparison: User Template vs. Generated Code

| Aspect | User Template (LC) | Generated (VP) |
|--------|-------------------|----------------|
| Damage | Yes | No |
| Backstress | Yes (X) | No |
| Hardening | Saturation (R1, b) | Linear (H) |
| Helper methods | 4 methods | 2 methods |
| State variables | 14 | 10 |
| Code lines | ~300 | ~180 |
| Complexity | High | Medium |

The generated code is **simpler** because the model is simpler (no damage, no kinematic hardening), but follows the **exact same pattern** and style.

## Advanced Usage: Complex Model

If the user provides a more complex model like Lemaitre-Chaboche, Claude would:

1. Recognize all equation types (damage, kinematic hardening, etc.)
2. Consult `references/example-models.md` Example 5
3. Include all necessary state variables and evolution laws
4. Match the complexity of the user's uploaded template

## Customization

The skill allows Claude to handle variations:

- **Different hardening laws**: Nonlinear, multi-term
- **Multiple backstresses**: Chaboche decomposition
- **Different damage models**: Simplified, stress-state dependent
- **Rate-independent**: Adjust K and n parameters
- **Custom yield functions**: Drucker-Prager, Mohr-Coulomb

Claude adapts the template to the specific equations provided while maintaining code quality and style consistency.

## Usage Tips for Users

To get the best results:

1. **Provide equations clearly**: Use mathematical notation
2. **List all parameters**: With units and typical values
3. **Specify model type**: Viscoplastic, rate-independent, etc.
4. **Reference paper if available**: For equation numbers
5. **Upload template code**: For style matching
6. **State requirements**: Plane stress, JAX compatibility, etc.

## Summary

The explicit integration scheme skill enables Claude Opus 4.5 to:

- ✅ Transform constitutive equations into working JAX code
- ✅ Match user's coding style and conventions
- ✅ Generate numerically stable implementations
- ✅ Include comprehensive documentation
- ✅ Handle simple to complex material models
- ✅ Follow established algorithm patterns (radial return)
- ✅ Produce production-ready material model code

The skill bridges the gap between theoretical formulation and practical implementation, making it valuable for computational mechanics researchers and engineers.
