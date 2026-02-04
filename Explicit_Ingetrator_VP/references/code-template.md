# Complete Code Template for Explicit Integration Scheme

This file provides a fully annotated template for implementing explicit integration schemes for viscoplastic constitutive models with damage.

## Full Implementation Template

```python
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from dolfinx_materials.material.jax import JAXMaterial, tangent_AD


@dataclass
class ModelParams:
    """
    Material parameters for the constitutive model.
    Customize based on the specific model equations provided.
    """
    # Elastic properties
    E: float       # Young's modulus [MPa]
    nu: float      # Poisson's ratio [-]

    # Viscoplastic parameters (Perzyna-type)
    k: float       # Initial yield stress [MPa]
    K_visc: float  # Viscosity parameter [MPa·s^(1/n)]
    n_visc: float  # Viscosity exponent [-]

    # Isotropic hardening (if applicable)
    b: float       # Saturation rate [-]
    R1: float      # Saturation value [MPa]

    # Kinematic hardening (if applicable)
    a: float       # Kinematic modulus [MPa]
    c: float       # Dynamic recovery [-]

    # Damage parameters (if applicable)
    Dc: float      # Critical damage [-]
    eps_R: float   # Critical plastic strain [-]
    eps_D: float   # Damage threshold strain [-]


class ConstitutiveMaterial(JAXMaterial):
    """
    Explicit integration scheme 

    """
    
    def __init__(self, elastic_model, params: ModelParams):
        super().__init__()
        self.elastic_model = elastic_model
        self.params = params
        self.dt = 0.0 

    @property
    def gradient_names(self):
        """Input gradients - typically total strain"""
        return ("Strain",)

    @property
    def flux_names(self):
        """Output fluxes - typically Cauchy stress"""
        return ("Stress",)

    @property
    def internal_state_variables(self):
        """
        All history variables per integration point.
        Each key maps to the dimension of that variable:
        - Scalars: 1
        - Voigt tensors (3D): 6
        
        CRITICAL: All variables used in constitutive_update must be declared here.
        """
        return {
            # Primary internal variables
            "p": 1,      # Equivalent plastic strain (scalar)
            "D": 1,      # Damage variable (scalar, 0 to Dc)
            "R": 1,      # Isotropic hardening (scalar)
            "eps_p": 6,  # Plastic strain tensor (Voigt)
            "X": 6,      # Backstress tensor (Voigt)
            "eps_e": 6,  # Elastic strain tensor (Voigt)
            
            # Diagnostic/output variables
            "fy": 1,         # Yield function value
            "p_dot": 1,      # Plastic strain rate
            "dp": 1,         # Plastic strain increment
            "D_dot": 1,      # Damage rate
            "R_dot": 1,      # Isotropic hardening rate
            "X_dot": 6,      # Backstress rate (Voigt)
            "eps_I_dot": 6,  # Inelastic strain rate (Voigt)
            "is_plastic": 1  # Flag indicating plastic loading
        }

    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _deviatoric(self, sig):
        """
        Extract deviatoric part of Voigt stress tensor.
        
        Input: sig = [s11, s22, s33, s12, s13, s23]
        Output: sig_dev = sig - (1/3)*tr(sig)*I
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
    
    def _J2(self, sig_dev):
        """
        J2 invariant with numerical regularization.
        
        J2 = sqrt(3/2 * s_dev : s_dev)
        
        The regularization prevents gradient issues at s_dev = 0:
        - Physical value: sqrt(max(1.5*s:s, 0))
        - Gradient computed from: sqrt(1.5*s:s + eps_reg)
        
        This uses JAX's stop_gradient to achieve this behavior.
        """
        s = sig_dev
        
        # Compute s : s with Voigt convention (factor of 2 for shear terms)
        s_colon_s = (
            s[0] * s[0] +
            s[1] * s[1] +
            s[2] * s[2] +
            2.0 * (s[3] * s[3] + s[4] * s[4] + s[5] * s[5])
        )
        
        val = 1.5 * s_colon_s
        val_pos = jnp.maximum(val, 0.0)

        # Physical value (what we actually want)
        J2_phys = jnp.sqrt(val_pos)

        # Regularized value (for gradient computation)
        eps_reg = 1e-16
        J2_reg = jnp.sqrt(val_pos + eps_reg)

        # Return physical value but with regularized gradient
        return jax.lax.stop_gradient(J2_phys - J2_reg) + J2_reg

    def _hydrostatic(self, sig):
        """Hydrostatic (mean normal) stress."""
        return (sig[0] + sig[1] + sig[2]) / 3.0

    def _equiv_stress(self, sig):
        """Huber-Mises equivalent stress from total stress."""
        sig_dev = self._deviatoric(sig)
        return self._J2(sig_dev)
    
    # ========================================================================
    # MAIN CONSTITUTIVE UPDATE
    # ========================================================================
    
    @tangent_AD
    def constitutive_update(self, eps, state, dt):
        """
        Explicit integration scheme for one time step.
        
        Algorithm structure:
        1. Extract old state 
        2. Elastic predictor 
        3. Check plasticity 
        4. Elastic or plastic branch 
        5. Update state variables 
        
        Args:
            eps: Current total strain (6-component Voigt)
            state: Dictionary of all state variables
            dt: Time step size
            
        Returns:
            sig_new: Updated stress
            state: Updated state dictionary
        """
        
        # ====================================================================
        # 1. EXTRACT OLD STATE VARIABLES 
        # ====================================================================
        
        eps_old = state["Strain"]     # Total strain at t-1
        deps = eps - eps_old          # Total strain increment
        sig_old = state["Stress"]     # Stress at t-1

        eps_p_old = state["eps_p"]    # Plastic strain at t-1 (Voigt)
        eps_e_old = state["eps_e"]    # Elastic strain at t-1 (Voigt)
        X_old = state["X"]            # Backstress at t-1 (Voigt)

        p_old = state["p"][0]         # Equivalent plastic strain (scalar)
        D_old = state["D"][0]         # Damage (scalar)
        R_old = state["R"][0]         # Isotropic hardening (scalar)
        
        # Rates from previous step (used in plastic update initialization)
        p_dot = state["p_dot"][0] 
        eps_I_dot = state["eps_I_dot"]
     
        # ====================================================================
        # 2. ELASTIC PREDICTOR 
        # ====================================================================
        
        # Trial stress assuming purely elastic increment
        C = self.elastic_model.C  # Elastic stiffness tensor 
        sig_trial = sig_old + (1.0 - D_old) * (C @ deps)

        # ====================================================================
        # 3. YIELD FUNCTION EVALUATION 
        # ====================================================================
        
        params = self.params
        
        # Effective stress (accounting for backstress)
        sig_eff = sig_trial - X_old
        sig_eff_dev = self._deviatoric(sig_eff)
        J2_eff = self._J2(sig_eff_dev)

        # Correct for damage
        denom_D = 1.0 - D_old 
        sigma_eff = J2_eff / denom_D

        # Yield function: f = sigma_eff - (R + k)
        # f < 0: elastic, f >= 0: plastic
        fy = sigma_eff - (R_old + params.k) ## this yield function will change according to user's query

        # ====================================================================
        # 4. PREPARE FOR ELASTIC/PLASTIC BRANCHING
        # ====================================================================
        
        # Initialize variables that will be updated
        D_dot = 0.0
        R_dot = 0.0
        X_dot = jnp.zeros((6,), dtype=eps.dtype)
        R_new = 0.0
        X_new = jnp.zeros((6,), dtype=eps.dtype)

        # Pack all variables into operand tuple for jax.lax.cond
        # Both branches must accept the same operand signature
        operand = (eps, eps_old, sig_old, eps_p_old, eps_e_old, X_old,
                   p_old, D_old, R_old, deps, sig_trial, fy, dt, 
                   p_dot, D_dot, R_dot, X_dot, eps_I_dot, R_new, X_new)

        # ====================================================================
        # 5. ELASTIC BRANCH (Algorithm 1, line 9-12)
        # ====================================================================
        
        def _elastic_update(operand):
            """
            Elastic loading: no plastic deformation, no hardening evolution.
            Simply use trial stress and update elastic strain.
            """
            (eps, eps_old, sig_old, eps_p_old, eps_e_old, X_old,
             p_old, D_old, R_old, deps, sig_trial, fy, dt, 
             p_dot, D_dot, R_dot, X_dot, eps_I_dot, R_new, X_new) = operand
        
            # Accept trial stress (no plastic correction needed)
            sig_new = sig_trial
            
            # Backstress remains unchanged
            X_new = X_old

            # All strain increment is elastic
            eps_e_new = eps_e_old + deps
            eps_p_new = eps_p_old  
            eps = eps_e_new + eps_p_new

            # No evolution of plastic variables
            p_new = p_old
            D_new = D_old
            R_new = R_old
        
            # All rates are zero
            p_dot = 0.0
            dp = 0.0
            D_dot = 0.0
            R_dot = 0.0
            X_dot = jnp.zeros_like(X_new)
            eps_I_dot = jnp.zeros((6,), dtype=eps.dtype)

            # Flag indicating no plastic deformation
            is_plastic_out = jnp.array(0.0, dtype=eps.dtype)

            # Return all variables (must match plastic branch signature)
            return (sig_new, eps_e_new, X_new, eps, eps_p_new, 
                    p_new, D_new, R_new, fy, p_dot, dp, D_dot, 
                    R_dot, X_dot, eps_I_dot, is_plastic_out)
        
        # ====================================================================
        # 6. PLASTIC BRANCH (Algorithm 1, line 14-41)
        # ====================================================================
        
        def _plastic_update(operand):
            """
            Plastic loading: compute plastic strain increment and evolve
            all internal variables (hardening, damage).
            """
            (eps, eps_old, sig_old, eps_p_old, eps_e_old, X_old,
             p_old, D_old, R_old, deps, sig_trial, fy, dt, 
             p_dot, D_dot, R_dot, X_dot, eps_I_dot, R_new, X_new) = operand

            # ================================================================
            # 6a. Update P and damage
            # ================================================================
            
            # Equivalent plastic strain increment
            dp = p_dot * dt 
            p_new = p_old + dp 

            ## Update damage
            D_new = D_old + D_dot * dt

            # ================================================================
            # 6b. UPDATE STRAINS
            # ================================================================
            # Inelastic strain increment
            delta_eps_I = eps_I_dot * dt

            # Total plastic strain
            eps_p_new = eps_p_old + delta_eps_I
            
            # Elastic strain increment (total minus inelastic)
            delta_eps_e = deps - delta_eps_I
            eps_e_new = eps_e_old + delta_eps_e

            # Total strain (should equal input eps)
            eps = eps_e_new + eps_p_new

            # ================================================================
            # 6c. UPDATE STRESS
            # ================================================================

            # Stress update with evolving damage
            sig_new = sig_old + (1.0 - D_new) * (C @ delta_eps_e)

            # ================================================================
            # 6d. UPDATE X and R
            # ================================================================
            
            X_new = X_old + X_dot * dt
            R_new = R_old + R_dot * dt

       
            # ================================================================
            # 6e. COMPUTE P dot
            # ================================================================
            
            sig_eff_1 = sig_new - X_new
            sig_eff_dev_1 = self._deviatoric(sig_eff_1)
            J2_eff_1 = self._J2(sig_eff_dev_1)

            # Effective stress corrected for damage
            denom_D = 1.0 - D_old 
            sigma_eff_2 = J2_eff_1 / denom_D
            fy_1 = sigma_eff_2 - (R_new + params.k)
            x = fy_1 / params.K_visc
            bracket = 0.5 * (x + jnp.abs(x))  

            p_dot = jnp.power(bracket, params.n_visc)
            
            # ================================================================
            # 6f. COMPUTE INELASTIC STRAIN INCREMENT (E_dot)
            # ================================================================
            sig_eff_p = sig_new - X_new
            sig_eff_dev_p = self._deviatoric(sig_eff_p)
            J2_eff_p = self._J2(sig_eff_dev_p)
            
            inv_J2 = jnp.where(J2_eff_p > 0.0, 1.0 / J2_eff_p, 0.0)

            flow_dir = sig_eff_dev_p * inv_J2      # normalized direction
            # Inelastic strain rate: eps_I_dot = (3/2) * p_dot * n
            eps_I_dot = 1.5 * p_dot * flow_dir
            
            # ================================================================
            # 6g. UPDATE HARDENING VARIABLES 
            # ================================================================
            
            # Isotropic hardening rate: R_dot = b(R1 - R) * p_dot
            R_dot = params.b * (params.R1 - R_new) * p_dot
         
            
            # Kinematic hardening rate: X_dot = (2/3)*a*eps_I_dot - c*X*p_dot
            X_dot = (2.0 / 3.0) * params.a * eps_I_dot - params.c * X_new * p_dot
    

            # ================================================================
            # 6h. UPDATE Damage parameters
            # ================================================================
            # For models WITHOUT damage evolution during the step:
            # sig_new = sig_old + (1.0 - D_old) * C @ (deps - delta_eps_I)
            
            # For models WITH damage evolution:
            # First compute damage rate and new damage
            
            # Damage rate (Lemaitre damage)
            sigma_eq = self._equiv_stress(sig_new)  # Using trial stress
            sigma_H = self._hydrostatic(sig_new)

            nu = params.nu
            bracket_damage = (
                (2.0 / 3.0) * (1.0 + nu) * sigma_eq * sigma_eq +
                3.0 * (1.0 - 2.0 * nu) * sigma_H * sigma_H
            )

            factor_D = params.Dc / (params.eps_R - params.eps_D + 1e-12)
            D_dot = factor_D * bracket_damage * p_dot

       

            # Flag indicating plastic deformation occurred
            is_plastic_out = jnp.array(1.0, dtype=eps.dtype)

            # Return all variables (must match elastic branch signature)
            return (sig_new, eps_e_new, X_new, eps, eps_p_new,
                    p_new, D_new, R_new, fy, p_dot, dp, D_dot,
                    R_dot, X_dot, eps_I_dot, is_plastic_out)
        
        # ====================================================================
        # 7. CONDITIONAL EXECUTION 
        # ====================================================================
        
        # Determine which branch to execute based on yield function
        is_plastic = fy > 0.0

        # Execute appropriate branch (JAX will compile both)
        (sig_new, eps_e_new, X_new, eps, eps_p_new, p_new, D_new, R_new, 
         fy, p_dot, dp, D_dot, R_dot, X_dot, eps_I_dot, is_plastic_out) = jax.lax.cond(
            is_plastic,
            _plastic_update,
            _elastic_update,
            operand,
        ) 

        # ====================================================================
        # 8. UPDATE STATE DICTIONARY (Algorithm 1, line 33-41)
        # ====================================================================
        
        # CRITICAL: Update ALL state variables, even if unchanged
        state["Strain"] = eps
        state["Stress"] = sig_new

        state["eps_p"] = eps_p_new
        state["eps_e"] = eps_e_new
        state["X"] = X_new
        state["eps_I_dot"] = eps_I_dot

        # Scalars must be wrapped in arrays
        state["p"] = jnp.array([p_new])
        state["R"] = jnp.array([R_new])
        state["D"] = jnp.array([D_new])
        state["fy"] = jnp.array([fy])
        state["p_dot"] = jnp.array([p_dot])
        state["dp"] = jnp.array([dp])
        state["D_dot"] = jnp.array([D_dot])
        state["R_dot"] = jnp.array([R_dot])
        state["is_plastic"] = jnp.array([is_plastic_out])

        return sig_new, state
```

## Key Points

1. **All branches must have same signature**: Both `_elastic_update` and `_plastic_update` must accept the same `operand` tuple and return the same number of values.

2. **Update all state variables**: Even if a variable doesn't change, it must be assigned in the state dictionary.

3. **Scalar wrapping**: JAX requires consistent shapes. Scalars are stored as length-1 arrays.

4. **Numerical stability**: Use regularization for square roots, safe division with `jnp.where`, and clipping for physical bounds.

5. **Pure functions**: The inner update functions must not have side effects or depend on external state.

6. **Algorithm correspondence**: Comments reference Algorithm 1 line numbers to maintain traceability to the source paper.