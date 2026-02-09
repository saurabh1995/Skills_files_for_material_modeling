# Complete Code Template for Explicit Integration Scheme

This file provides a fully annotated template for implementing explicit integration schemes for viscoplastic constitutive models with damage.

## Full Implementation Template

## This is an implementation of Perzyna's viscoplastic model with implicit integration w.r.t article Viscoplasticity for instabilities due to strain softening and strain-rate softening. 
## The class PerzynaImplicitMaterial_PurePlasticity refers to constant yield function. Yield function is equal to yield stress value from material parameter.

```python
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from jax import debug as jdbg
from jax_newton_solver import JAXNewton
from dolfinx_materials.material.jax import JAXMaterial, tangent_AD

@dataclass
class PerzynaImplicitMaterial_PurePlasticity(JAXMaterial):
    E: float  # Young's modulus
    nu: float  # Poisson's ratio
    sig_y: float  # Yield stress
    eta: float  # Viscosity parameter
    N : float # Constant of eq 15

class PerzynaImplicitModel_PurePlasticity(JAXMaterial):

    def __init__(self, elastic_model, params: PerzynaImplicitMaterial_PurePlasticity):
        super().__init__()
        self.elastic_model = elastic_model
        self.params = params
        self.dt = 0.0

    @property
    def gradient_names(self):
        
        return ("Strain",)

    @property
    def flux_names(self):
    
        return ("Stress",)

    @property
    def internal_state_variables(self):
        """
        Extra history variables per Gauss point, beyond Strain/Stress.
        Shapes are vector lengths for each key.
        """
        return {
           
            "eps_p": 6, 
            "eps_e" : 6,
            "fy": 1,
            "dlam": 1,

        }

 
    def _deviatoric(self, sig):
        """
        Deviatoric part of a 6-component Voigt stress:
        sig = [s11, s22, s33, s12, s23, s13].
        """
        p = (sig[0] + sig[1] + sig[2]) / 3.0
        return jnp.array(
            [
                sig[0] - p,
                sig[1] - p,
                sig[2] - p,
                sig[3],
                sig[4],
                sig[5],
            ],
            dtype=sig.dtype,
        )
    def _J2(self, sig_dev):

        """
        J2 = sqrt(3/2 * s_dev : s_dev), with Voigt convention.
        - Use a slightly regularized version only for the derivative to avoid
          AD problems at s:s = 0 (sqrt kink).
        """
        s = sig_dev
        
        s_colon_s = (
            s[0] * s[0]
            + s[1] * s[1]
            + s[2] * s[2]
            + 2.0 * (s[3] * s[3] + s[4] * s[4] + s[5] * s[5])
        )
        #s_colon_s = jnp.dot(s, s) 

        val = 1.5 * s_colon_s
        val_pos = jnp.maximum(val, 0.0)

        # Physical value (what you would have without any regularisation)
        J2_phys = jnp.sqrt(val_pos)

        # Regularised value used only for the gradient
        eps_reg = 1e-16
        J2_reg = jnp.sqrt(val_pos + eps_reg)

        # Return a value whose forward eval = J2_phys,
        # but whose derivative behaves like J2_reg
        return jax.lax.stop_gradient(J2_phys - J2_reg) + J2_reg


    def _hydrostatic(self, sig):
        """
        Hydrostatic stress (mean normal stress).
        """
        return (sig[0] + sig[1] + sig[2]) / 3.0

    def _equiv_stress(self, sig):
        """
        Huber–Mises equivalent stress from total stress (no backstress).
        """
        sig_dev = self._deviatoric(sig)
        return self._J2(sig_dev)
    
    def df_dsigma(self, sig):

        '''
        This is only for ∂f/∂σ not for f/s or s/σ
        '''
        s11, s22, s33, s12, s23, s13 = sig
        p = (s11 + s22 + s33) / 3.0

        s_dev = jnp.array([
            s11 - p,
            s22 - p,
            s33 - p,
            s12,
            s23,
            s13
        ])

        sigma_eq = jnp.sqrt(1.5 * (
            (s11-p)**2 + (s22-p)**2 + (s33-p)**2
            + 2*(s12**2 + s23**2 + s13**2)
        ))

        return (3.0 / (2.0 * sigma_eq)) * s_dev
    
    def d2f_dsigma2(self, sig):

        '''
        This is only for ∂²f/∂σ²
        '''

        s11, s22, s33, s12, s23, s13 = sig
        p = (s11 + s22 + s33) / 3.0

        s_dev = jnp.array([
            s11 - p,
            s22 - p,
            s33 - p,
            s12,
            s23,
            s13
        ])

        sigma_eq = jnp.sqrt(1.5 * (
            (s11-p)**2 + (s22-p)**2 + (s33-p)**2
            + 2*(s12**2 + s23**2 + s13**2)
        ))

        P_dev = jnp.array([
            [ 2/3, -1/3, -1/3, 0, 0, 0],
            [-1/3,  2/3, -1/3, 0, 0, 0],
            [-1/3, -1/3,  2/3, 0, 0, 0],
            [ 0,    0,    0,    1, 0, 0],
            [ 0,    0,    0,    0, 1, 0],
            [ 0,    0,    0,    0, 0, 1]
        ])

        term1 = (3.0 / (2.0 * sigma_eq)) * P_dev
        term2 = (3.0 / (2.0 * sigma_eq**3)) * jnp.outer(s_dev, s_dev)

        return term1 - term2


    
    @tangent_AD
    def constitutive_update(self, eps, state, dt):

        eps_old = state["Strain"]
        sig_old = state["Stress"]
        eps_p_old = state["eps_p"]
        eps_e_old = state["eps_e"]
        dlam = state["dlam"][0]


        params = self.params

        C =self.elastic_model.C

        sig_trial = sig_old + C @ (eps - eps_old)

        J2 = self._equiv_stress(sig_trial)

        fy = J2 - params.sig_y

        #print(state.keys())

       
        #dlam = state.get("dlam", jnp.array(0.0, dtype=eps.dtype))

        operand = (eps, eps_old, sig_old, eps_p_old, eps_e_old,  dt, fy, dlam)

        def elastic_update(operand):
            (eps, eps_old, sig_old, eps_p_old, eps_e_old,  dt, fy, dlam) = operand
            sig_new = sig_trial
            eps_p_new = eps_p_old
            eps_e_new = eps_e_old + (eps - eps_old)
            #dlam = 0.0
          

            return sig_new, eps_p_new, eps_e_new, fy, dlam
        
        def plastic_update(operand):
            (eps, eps_old, sig_old, eps_p_old, eps_e_old,  dt, fy, dlam) = operand

            #dlam = jnp.array(0.0, dtype=eps.dtype)
            #dlam = 0.0
            sig0 = sig_old + C @ ((eps - eps_old) - dlam * self.df_dsigma(sig_old)) 


            sig_eq0 = self._equiv_stress(sig0)
            f0 = sig_eq0 - params.sig_y
            phi0 = jnp.where(f0 > 0.0, (f0 / params.sig_y)** params.N, 0.0)
            

            r0 = dlam - params.eta * dt * phi0

            def R_plastic(dlam):
                J2 = self._equiv_stress(sig0)
                f = J2 - params.sig_y   # iso hardening via kappa
                phi = jnp.where(f > 0.0, (f / params.sig_y)**params.N, 0.0)        # example power law
                return dlam - params.eta * dt * phi

           

            newton = JAXNewton()
            newton.set_residual(R_plastic)
            dlam, _ = newton.solve(0.0)




            n = self.df_dsigma(sig_old)
            deps_p_new = eps_p_old + dlam * n

            sig_new = sig_old + C @ ((eps - eps_old) - deps_p_new)
            eps_p_new = deps_p_new*dt
            

            #eps_p_new = eps_p_old + dlam * self.df_dsigma(sig_new)
           
            eps_e_new =  (eps - eps_p_new)

            return sig_new, eps_p_new, eps_e_new,  fy, dlam
        
        is_plastic = fy > 0.0

        sig_new, eps_p_new, eps_e_new, fy,dlam = jax.lax.cond(
            is_plastic,
            plastic_update,
            elastic_update,
            operand,
        )
        

        
        state["Strain"] = eps
        state["Stress"] = sig_new
        state["eps_p"] = eps_p_new
        state["eps_e"] = eps_e_new
        state["fy"] = jnp.array([fy])
        state["dlam"] = jnp.array([dlam])
     
        


        return sig_new, state
```

## This class PerzynaImplicitMaterial_Hardening is an example when yield function is not constant
```python
@dataclass
class PerzynaImplicitMaterial_Hardening(JAXMaterial):
    E: float  # Young's modulus
    nu: float  # Poisson's ratio
    sig_y: float  # Yield stress
    eta: float  # Viscosity parameter
    N : float # Constant of eq 15
    H : float ## 

class PerzynaImplicitModel_Hardening(JAXMaterial):

    def __init__(self, elastic_model, yield_function, params: PerzynaImplicitMaterial_Hardening):
        super().__init__()
        self.elastic_model = elastic_model
        self.yield_function = yield_function
        self.params = params
        self.dt = 0.0

    @property
    def gradient_names(self):
        
        return ("Strain",)

    @property
    def flux_names(self):
    
        return ("Stress",)

    @property
    def internal_state_variables(self):
        """
        Extra history variables per Gauss point, beyond Strain/Stress.
        Shapes are vector lengths for each key.
        """
        return {
           
            "eps_p": 6, 
            "eps_e" : 6,
            "fy": 1,
            "dlam": 1,

        }

 
    def _deviatoric(self, sig):
        """
        Deviatoric part of a 6-component Voigt stress:
        sig = [s11, s22, s33, s12, s23, s13].
        """
        p = (sig[0] + sig[1] + sig[2]) / 3.0
        return jnp.array(
            [
                sig[0] - p,
                sig[1] - p,
                sig[2] - p,
                sig[3],
                sig[4],
                sig[5],
            ],
            dtype=sig.dtype,
        )
    def _J2(self, sig_dev):

        """
        J2 = sqrt(3/2 * s_dev : s_dev), with Voigt convention.
        - Use a slightly regularized version only for the derivative to avoid
          AD problems at s:s = 0 (sqrt kink).
        """
        s = sig_dev
        
        s_colon_s = (
            s[0] * s[0]
            + s[1] * s[1]
            + s[2] * s[2]
            + 2.0 * (s[3] * s[3] + s[4] * s[4] + s[5] * s[5])
        )
        #s_colon_s = jnp.dot(s, s) 

        val = 1.5 * s_colon_s
        val_pos = jnp.maximum(val, 0.0)

        # Physical value (what you would have without any regularisation)
        J2_phys = jnp.sqrt(val_pos)

        # Regularised value used only for the gradient
        eps_reg = 1e-16
        J2_reg = jnp.sqrt(val_pos + eps_reg)

        # Return a value whose forward eval = J2_phys,
        # but whose derivative behaves like J2_reg
        return jax.lax.stop_gradient(J2_phys - J2_reg) + J2_reg


    def _hydrostatic(self, sig):
        """
        Hydrostatic stress (mean normal stress).
        """
        return (sig[0] + sig[1] + sig[2]) / 3.0

    def _equiv_stress(self, sig):
        """
        Huber–Mises equivalent stress from total stress (no backstress).
        """
        sig_dev = self._deviatoric(sig)
        return self._J2(sig_dev)
    
    def df_dsigma(self, sig):
        s11, s22, s33, s12, s23, s13 = sig
        p = (s11 + s22 + s33) / 3.0

        s_dev = jnp.array([
            s11 - p,
            s22 - p,
            s33 - p,
            s12,
            s23,
            s13
        ])

        sigma_eq = jnp.sqrt(1.5 * (
            (s11-p)**2 + (s22-p)**2 + (s33-p)**2
            + 2*(s12**2 + s23**2 + s13**2)
        ))

        return (3.0 / (2.0 * sigma_eq)) * s_dev
    
    def d2f_dsigma2(self, sig):
        s11, s22, s33, s12, s23, s13 = sig
        p = (s11 + s22 + s33) / 3.0

        s_dev = jnp.array([
            s11 - p,
            s22 - p,
            s33 - p,
            s12,
            s23,
            s13
        ])

        sigma_eq = jnp.sqrt(1.5 * (
            (s11-p)**2 + (s22-p)**2 + (s33-p)**2
            + 2*(s12**2 + s23**2 + s13**2)
        ))

        P_dev = jnp.array([
            [ 2/3, -1/3, -1/3, 0, 0, 0],
            [-1/3,  2/3, -1/3, 0, 0, 0],
            [-1/3, -1/3,  2/3, 0, 0, 0],
            [ 0,    0,    0,    1, 0, 0],
            [ 0,    0,    0,    0, 1, 0],
            [ 0,    0,    0,    0, 0, 1]
        ])

        term1 = (3.0 / (2.0 * sigma_eq)) * P_dev
        term2 = (3.0 / (2.0 * sigma_eq**3)) * jnp.outer(s_dev, s_dev)

        return term1 - term2


    
    @tangent_AD
    def constitutive_update(self, eps, state, dt):

        eps_old = state["Strain"]
        sig_old = state["Stress"]
        eps_p_old = state["eps_p"]
        eps_e_old = state["eps_e"]
        dlam = state["dlam"][0]


        params = self.params

        C =self.elastic_model.C

        sig_trial = sig_old + C @ (eps - eps_old)

        J2 = self._equiv_stress(sig_trial)

        yield_func = self.yield_function(dlam*dt)

        fy = J2 - yield_func


        operand = (eps, eps_old, sig_old, eps_p_old, eps_e_old,  dt, fy, dlam)

        def elastic_update(operand):
            (eps, eps_old, sig_old, eps_p_old, eps_e_old,  dt, fy, dlam) = operand
            sig_new = sig_trial
            eps_p_new = eps_p_old
            eps_e_new = eps_e_old + (eps - eps_old)
            #dlam = 0.0
          

            return sig_new, eps_p_new, eps_e_new, fy, dlam
        
        def plastic_update(operand):
            (eps, eps_old, sig_old, eps_p_old, eps_e_old,  dt, fy, dlam) = operand

            #dlam = jnp.array(0.0, dtype=eps.dtype)
            #dlam = 0.0
            sig0 = sig_old + C @ ((eps - eps_old) - dlam * self.df_dsigma(sig_old)) 


            sig_eq0 = self._equiv_stress(sig0)
            lam_new = dlam*dt + dlam
            f0 = sig_eq0 - self.yield_function(lam_new)
            phi0 = jnp.where(f0 > 0.0, (f0 / params.sig_y)** params.N, 0.0)
            

            r0 = dlam - params.eta * dt * phi0
            #jdbg.print("dlam_old = {dlam_old}\n", dlam_old=dlam)


            def R_plastic(dlam):
                sig_eq = self._equiv_stress(sig0)
                lam_new = dlam*dt + dlam
                f = sig_eq - self.yield_function(lam_new)
                phi = jnp.where(f0 > 0.0, (f / params.sig_y)** params.N, 0.0)
                return dlam - params.eta * dt * phi  

            newton = JAXNewton()
            newton.set_residual(R_plastic)
            dlam, _ = newton.solve(0.0)




            n = self.df_dsigma(sig_old)
            deps_p_new = eps_p_old + dlam * n

            sig_new = sig_old + C @ ((eps - eps_old) - deps_p_new)
            eps_p_new = deps_p_new*dt
            

            #eps_p_new = eps_p_old + dlam * self.df_dsigma(sig_new)
           
            eps_e_new =  (eps - eps_p_new)
            '''
            jdbg.print("sig0 = {sig0}\n", sig0=sig0)
            jdbg.print("r0 = {r0}\n", r0=r0)
            jdbg.print("dlam = {dlam}\n", dlam=dlam)
            jdbg.print("phi0 = {phi0}\n", phi0=phi0)
            jdbg.print("sig_new = {sig_new}\n", sig_new=sig_new)
            jdbg.print("dt = {dt}\n", dt=dt)'''
           
            return sig_new, eps_p_new, eps_e_new,  fy, dlam
        
        is_plastic = fy > 0.0

        sig_new, eps_p_new, eps_e_new, fy,dlam = jax.lax.cond(
            is_plastic,
            plastic_update,
            elastic_update,
            operand,
        )
        

        
        state["Strain"] = eps
        state["Stress"] = sig_new
        state["eps_p"] = eps_p_new
        state["eps_e"] = eps_e_new
        state["fy"] = jnp.array([fy])
        state["dlam"] = jnp.array([dlam])
     
        


        return sig_new, state

```
