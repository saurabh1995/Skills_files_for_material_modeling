# Complete von Mises Template

von Mises elastoplasticity with isotropic hardening using explicit return mapping.

## Full Implementation

```python
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from dolfinx_materials.material.jax import JAXMaterial, tangent_AD, JAXNewton


@dataclass
class vonMisesParams:
    """
    Material parameters for von Mises elastoplasticity with isotropic hardening.
    
    Constitutive equations:
    1. σ = C : (ε - ε^p) (elastic law)
    2. f = σ_eq - R(p) ≤ 0 (yield condition)
    3. ε̇^p = λ̇ ∂f/∂σ = λ̇ (3/2) s/σ_eq (associative flow)
    4. ṗ = λ̇ (consistency parameter)
    5. R(p) = yield strength function
    
    where:
    - σ_eq = √(3/2 s:s) is von Mises equivalent stress
    - s = dev(σ) is deviatoric stress
    - p is cumulated plastic strain
    """
    # Elastic properties
    E: float       # Young's modulus [MPa]
    nu: float      # Poisson's ratio [-]
    
    # Yield stress (choose one type)
    # Type 1: Constant yield stress
    sigma_0: float = None  # Initial yield stress [MPa]
    
    # Type 2: Linear hardening
    # sigma_0 + H*p
    H: float = None  # Hardening modulus [MPa]
    
    # Type 3: Exponential saturation
    # sigma_0 + (sigma_inf - sigma_0)*(1 - exp(-b*p))
    sigma_inf: float = None  # Saturation yield stress [MPa]
    b: float = None          # Saturation rate [-]


class vonMisesIsotropicHardening(JAXMaterial):
    """
    von Mises elastoplasticity with general isotropic hardening.
    
    Uses explicit return mapping - much faster than general yield surface
    because the flow direction is known from the elastic predictor.
    """
    
    def __init__(self, elastic_model, yield_stress_function):
        """
        Args:
            elastic_model: LinearElasticModel with C, mu, lam properties
            yield_stress_function: Callable R(p) returning yield stress
        """
        super().__init__()
        self.elastic_model = elastic_model
        self.yield_stress = yield_stress_function
    
    @property
    def gradient_names(self):
        return ("Strain",)
    
    @property
    def flux_names(self):
        return ("Stress",)
    
    @property
    def internal_state_variables(self):
        """
        Only cumulated plastic strain p is needed.
        
        Note: We do NOT store ε^p because it's not needed for isotropic
        hardening (only p appears in R(p)).
        """
        return {
            "p": 1,  # Cumulated equivalent plastic strain
        }
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _deviatoric(self, sig):
        """
        Deviatoric part of stress tensor (Voigt notation).
        
        sig = [s11, s22, s33, s12, s13, s23]
        dev(sig) = sig - (1/3)tr(sig)I
        """
        p = (sig[0] + sig[1] + sig[2]) / 3.0
        return jnp.array([
            sig[0] - p,
            sig[1] - p,
            sig[2] - p,
            sig[3],
            sig[4],
            sig[5],
        ], dtype=sig.dtype)
    
    def _equivalent_stress(self, sig):
        """
        von Mises equivalent stress: σ_eq = √(3/2 s:s)
        
        For Voigt notation: s:s = s1² + s2² + s3² + 2(s12² + s13² + s23²)
        """
        s = self._deviatoric(sig)
        s_dot_s = (
            s[0]**2 + s[1]**2 + s[2]**2 + 
            2.0 * (s[3]**2 + s[4]**2 + s[5]**2)
        )
        return jnp.sqrt(3.0 / 2.0 * s_dot_s)
    
    # ========================================================================
    # MAIN CONSTITUTIVE UPDATE
    # ========================================================================
    
    @tangent_AD
    def constitutive_update(self, eps, state, dt):
        """
        Return mapping algorithm for von Mises plasticity.
        
        Algorithm:
        1. Elastic predictor
        2. Check yield criterion
        3. If elastic: return trial stress
        4. If plastic: solve 1D equation for Δp, update stress
        
        Args:
            eps: Current total strain (6-component Voigt)
            state: State dictionary
            dt: Time step (not used in rate-independent plasticity)
        
        Returns:
            sig: Updated stress
            state: Updated state dictionary
        """
        
        # ====================================================================
        # 1. EXTRACT OLD STATE
        # ====================================================================
        
        eps_old = state["Strain"]
        deps = eps - eps_old
        p_old = state["p"][0]  # Extract scalar from length-1 array
        sig_old = state["Stress"]
        
        # ====================================================================
        # 2. ELASTIC PREDICTOR
        # ====================================================================
        
        C = self.elastic_model.C
        mu = self.elastic_model.mu
        
        sig_el = sig_old + C @ deps  # Trial stress
        
        # ====================================================================
        # 3. CHECK YIELD CRITERION
        # ====================================================================
        
        # Equivalent stress of elastic predictor
        sig_eq_el = self._equivalent_stress(sig_el)
        
        # Clip to avoid division by zero later
        sig_eq_el = jnp.clip(sig_eq_el, a_min=1e-8)
        
        # Yield stress at old plastic strain
        sig_Y_old = self.yield_stress(p_old)
        
        # Yield criterion: f = σ_eq - R(p)
        yield_criterion = sig_eq_el - sig_Y_old
        
        # ====================================================================
        # 4. FLOW DIRECTION (von Mises specific)
        # ====================================================================
        
        # For von Mises, the flow direction is known from elastic predictor:
        # n = (3/2) s_el / σ_eq,el
        # This is a key simplification that makes explicit return mapping possible
        
        s_el = self._deviatoric(sig_el)
        n_el = s_el / sig_eq_el  # Already clipped above
        
        # ====================================================================
        # 5. PLASTIC STRAIN INCREMENT FUNCTION
        # ====================================================================
        
        def deps_p(dp, yield_criterion):
            """
            Compute plastic strain increment as function of Δp.
            
            For von Mises: Δε^p = (3/2) n_el * Δp
            
            Uses jax.lax.cond to handle elastic vs plastic cases.
            """
            
            def deps_p_elastic(dp):
                # Elastic: no plastic strain
                return jnp.zeros(6, dtype=eps.dtype)
            
            def deps_p_plastic(dp):
                # Plastic: Δε^p = (3/2) * n * Δp
                # For von Mises: n = s_el/σ_eq,el
                return 3.0 / 2.0 * n_el * dp
            
            return jax.lax.cond(
                yield_criterion < 0.0,
                deps_p_elastic,
                deps_p_plastic,
                dp,
            )
        
        # ====================================================================
        # 6. RESIDUAL EQUATION FOR Δp
        # ====================================================================
        
        def r(dp):
            """
            Residual equation to solve for Δp.
            
            Elastic case: r(Δp) = Δp → Δp = 0 (trivial)
            Plastic case: r(Δp) = σ_eq,el - 3μΔp - R(p_old + Δp) = 0
            
            The plastic case enforces the consistency condition:
            σ_eq,n+1 - R(p_n+1) = 0
            
            Using the von Mises property: σ_eq,n+1 = σ_eq,el - 3μΔp
            """
            
            def r_elastic(dp):
                # Trivial equation: dp = 0
                # Newton solver will find dp = 0
                return dp
            
            def r_plastic(dp):
                # Consistency condition: f(σ_n+1, p_n+1) = 0
                # σ_eq,n+1 - R(p_n+1) = 0
                # (σ_eq,el - 3μΔp) - R(p_old + Δp) = 0
                return sig_eq_el - 3.0 * mu * dp - self.yield_stress(p_old + dp)
            
            return jax.lax.cond(
                yield_criterion < 0.0,
                r_elastic,
                r_plastic,
                dp,
            )
        
        # ====================================================================
        # 7. SOLVE FOR Δp USING NEWTON SOLVER
        # ====================================================================
        
        # JAXNewton is a fully differentiable Newton-Raphson solver
        # It automatically computes the jacobian via AD
        newton = JAXNewton(r)
        
        # Solve starting from dp = 0
        dp, res = newton.solve(0.0)
        
        # ====================================================================
        # 8. UPDATE STRESS
        # ====================================================================
        
        # Compute plastic strain increment
        delta_eps_p = deps_p(dp, yield_criterion)
        
        # Update stress: σ = σ_el - C : Δε^p
        # For isotropic elasticity: σ = σ_el - 2μ Δε^p
        sig = sig_el - 2.0 * mu * delta_eps_p
        
        # ====================================================================
        # 9. UPDATE STATE DICTIONARY
        # ====================================================================
        
        state["Strain"] = eps
        state["p"] = jnp.array([p_old + dp])  # Wrap scalar in array
        state["Stress"] = sig
        
        return sig, state
```

## Example Usage

### Linear Hardening

```python
from dolfinx_materials.material.jax import LinearElasticModel

# Elastic properties
E = 200000.0  # MPa
nu = 0.3
elastic = LinearElasticModel(E, nu)

# Hardening function: R(p) = σ₀ + H·p
sigma_0 = 250.0  # MPa
H = 10000.0  # MPa

def yield_stress_linear(p):
    return sigma_0 + H * p

# Create material
material = vonMisesIsotropicHardening(elastic, yield_stress_linear)
```

### Exponential Saturation

```python
# Hardening function: R(p) = σ₀ + (σ∞ - σ₀)(1 - exp(-b·p))
sigma_0 = 250.0  # MPa
sigma_inf = 450.0  # MPa
b = 50.0

def yield_stress_saturation(p):
    return sigma_0 + (sigma_inf - sigma_0) * (1.0 - jnp.exp(-b * p))

material = vonMisesIsotropicHardening(elastic, yield_stress_saturation)
```

### Power Law

```python
# Hardening function: R(p) = σ₀ + K·p^n
sigma_0 = 250.0  # MPa
K = 500.0  # MPa
n = 0.5

def yield_stress_power(p):
    return sigma_0 + K * jnp.power(p, n)

material = vonMisesIsotropicHardening(elastic, yield_stress_power)
```

## Key Features

### Why von Mises is Special

1. **Explicit return mapping**: Don't need to solve system, just 1D equation
2. **Flow direction from trial**: n = s_el/σ_eq,el (not updated iteratively)
3. **Direct relation**: σ_eq,n+1 = σ_eq,el - 3μΔp
4. **Much faster**: ~10x faster than general yield surface

### Advantages

- Minimal state variables (only p)
- Fast convergence (1D Newton)
- Robust and stable
- Compatible with tangent_AD

### Limitations

- Only for von Mises yield surface
- Only isotropic elasticity
- Only isotropic hardening
- For kinematic hardening or other surfaces, use general template

## Testing

```python
# Uniaxial tension test
import jax.numpy as jnp

# Initialize state
state = {
    "Strain": jnp.zeros(6),
    "Stress": jnp.zeros(6),
    "p": jnp.array([0.0])
}

# Apply strain increment
eps = jnp.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])  # 1% tension
dt = 1.0  # Not used

sig, state = material.constitutive_update(eps, state, dt)

print(f"Stress: {sig}")
print(f"Plastic strain: {state['p'][0]}")
```

## Performance Notes

Typical Newton iterations for plasticity:
- Elastic step: 1 iteration (trivial solution)
- Plastic step: 2-4 iterations (quadratic convergence)
- Total cost: Very low (1D problem)

Compare to general yield surface:
- System size: 1D vs 7D
- Iterations: ~3 vs ~5-10
- Speed: ~10x faster
