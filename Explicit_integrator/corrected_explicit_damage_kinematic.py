"""
Corrected explicit integration scheme for a viscoplastic material model with damage and kinematic hardening.

The implementation follows the forward‑Euler pattern described in the CLAUDE.md guidelines:
1. Use rates from the *previous* time step to compute increments.
2. Update all state variables first.
3. Compute new yield function and new rates from the fully updated state.
4. Return the updated state and new rates for the next step.

The class expects a `params` object exposing the material constants and an `elastic_model` providing the elastic stiffness matrix `C`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import Dict, Any


class ExplicitDamageKinematicIntegrator:
    """Explicit forward‑Euler integrator for viscoplasticity with damage and kinematic hardening.

    The integrator keeps an internal state dictionary with the following keys:
    - "Strain" : cumulative strain (Voigt, 6‑array)
    - "Stress" : current stress (Voigt, 6‑array)
    - "eps_I" : inelastic strain (Voigt, 6‑array)
    - "eps_e" : elastic strain (Voigt, 6‑array)
    - "X"     : back‑stress (Voigt, 6‑array)
    - "p"     : equivalent plastic strain (scalar array of shape (1,))
    - "D"     : damage (scalar array of shape (1,))
    - "R"     : isotropic hardening (scalar array of shape (1,))
    - "fy"    : yield function value (scalar array of shape (1,))
    - "p_dot" : plastic strain rate for the *next* step (scalar array of shape (1,))
    - "dp"    : plastic strain increment of the current step (scalar array of shape (1,))
    - "D_dot" : damage rate for the *next* step (scalar array of shape (1,))
    - "R_dot" : isotropic hardening rate for the *next* step (scalar array of shape (1,))
    - "X_dot" : back‑stress rate for the *next* step (Voigt, 6‑array)
    - "eps_I_dot" : inelastic strain rate for the *next* step (Voigt, 6‑array)
    - "is_plastic" : plasticity flag (scalar array of shape (1,))
    """

    def __init__(self, params: Any, elastic_model: Any):
        self.params = params
        self.C = elastic_model.C  # Elastic stiffness matrix (6x6)
        # Kinematic hardening elastic matrix
        self.C_kin = params.C_kin

    # ---------------------------------------------------------------------
    # Helper functions
    # ---------------------------------------------------------------------
    @staticmethod
    def _deviatoric(stress: jnp.ndarray) -> jnp.ndarray:
        """Return the deviatoric part of a Voigt stress tensor.

        Parameters
        ----------
        stress : jnp.ndarray, shape (6,)
            Stress in Voigt notation [σxx, σyy, σzz, σxy, σyz, σzx].
        """
        trace = stress[0] + stress[1] + stress[2]
        mean = trace / 3.0
        dev = stress - jnp.array([mean, mean, mean, 0.0, 0.0, 0.0])
        return dev

    @staticmethod
    def _J2(dev: jnp.ndarray) -> jnp.ndarray:
        """Compute the second invariant of the deviatoric stress in Voigt.

        The formula accounts for the shear factor of 2 in Voigt.
        """
        J2 = 0.5 * (
            dev[0] ** 2
            + dev[1] ** 2
            + dev[2] ** 2
            + 2.0 * (dev[3] ** 2 + dev[4] ** 2 + dev[5] ** 2)
        )
        return J2

    @staticmethod
    def _equiv_stress(J2: jnp.ndarray) -> jnp.ndarray:
        """Equivalent (von Mises) stress from J2."""
        return jnp.sqrt(3.0 * J2)

    # ---------------------------------------------------------------------
    # Elastic and plastic update kernels
    # ---------------------------------------------------------------------
    def _elastic_update(self, operand):
        """Return unchanged state for elastic branch.
        The operand contains all the state variables and rates; they are simply passed through.
        """
        return operand

    def _plastic_update(self, operand):
        """Explicit forward‑Euler plastic update.

        The operand is a tuple containing the full state and the rates from the previous step.
        """
        (
            eps, eps_old, sig_old, eps_I_old, eps_e_old, X_old,
            p_old, D_old, R_old, deps, sig_trial, sig_eff, sig_eff_dev, J2_eff, fy, dt,
            p_dot_old, eps_I_dot_old, D_dot_old, R_dot_old, X_dot_old,
        ) = operand

        # Step 1: use previous rates to compute increments
        dp = p_dot_old * dt
        p_new = p_old + dp

        D_new = D_old + D_dot_old * dt
        # Clamp damage to [0, Dc]
        D_new = jnp.clip(D_new, 0.0, self.params.Dc)

        delta_eps_I = eps_I_dot_old * dt
        eps_I_new = eps_I_old + delta_eps_I

        # Step 2: update strains
        delta_eps_e = deps - delta_eps_I
        eps_e_new = eps_e_old + delta_eps_e
        eps = eps_e_new + eps_I_new

        # Step 3: update stress with new damage
        sig_new = sig_old + (1.0 - D_new) * (self.C @ delta_eps_e)

        # Step 4: update hardening
        R_new = R_old + R_dot_old * dt
        X_new = X_old + X_dot_old * dt

        # Step 5: compute new yield function
        sig_eff_new = sig_new - X_new
        sig_eff_dev_new = self._deviatoric(sig_eff_new)
        J2_eff_new = self._J2(sig_eff_dev_new)
        denom_D_new = 1.0 - D_new
        sigma_eff_new = J2_eff_new / denom_D_new
        fy_new = sigma_eff_new - (R_new + self.params.sigma_0)

        # Step 6: compute new plastic strain rate
        x = fy_new / self.params.K
        bracket = 0.5 * (x + jnp.abs(x))
        p_dot_new = jnp.power(bracket, self.params.m)

        # Step 7: compute new flow direction
        inv_J2 = jnp.where(J2_eff_new > 0.0, 1.0 / J2_eff_new, 0.0)
        flow_dir = sig_eff_dev_new * inv_J2
        eps_I_dot_new = 1.5 * p_dot_new * flow_dir

        # Step 8: compute new hardening and damage rates
        sigma_eq = self._equiv_stress(J2_eff)  # use previous J2_eff for stability
        denom_D_safe = jnp.maximum(1.0 - D_old, 1e-12)
        D_dot_new = (self.params.alpha / denom_D_safe) * (sigma_eq / self.params.sigma_0) * p_dot_new

        R_dot_new = self.params.c1 * (self.params.R_inf - R_new) * p_dot_new
        X_dot_new = (2.0 / 3.0) * self.C_kin @ eps_I_dot_new - self.params.gamma * X_new * p_dot_new

        is_plastic_out = jnp.array([1.0])

        return (
            sig_new, eps_e_new, eps_I_new, X_new, eps,
            p_new, D_new, R_new,
            fy_new, p_dot_new, dp, D_dot_new, R_dot_new, X_dot_new, eps_I_dot_new, is_plastic_out,
        )

    # ---------------------------------------------------------------------
    # Public update method
    # ---------------------------------------------------------------------
    def update(self, state: Dict[str, jnp.ndarray], deps: jnp.ndarray, dt: float) -> Dict[str, jnp.ndarray]:
        """Perform one integration step.

        Parameters
        ----------
        state : dict
            Current state dictionary as described in ``internal_state_variables``.
        deps : jnp.ndarray, shape (6,)
            Incremental strain for the current time step.
        dt : float
            Time step size.
        """
        # Extract current values
        eps = state["Strain"]
        sig_old = state["Stress"]
        eps_I_old = state["eps_I"]
        eps_e_old = state["eps_e"]
        X_old = state["X"]
        p_old = state["p"][0]
        D_old = state["D"][0]
        R_old = state["R"][0]
        fy = state["fy"][0]
        dt = jnp.asarray(dt, dtype=eps.dtype)

        # Compute trial stress
        sig_trial = sig_old + self.C @ deps

        # Effective stress (trial minus old backstress)
        sig_eff = sig_trial - X_old
        sig_eff_dev = self._deviatoric(sig_eff)
        J2_eff = self._J2(sig_eff_dev)

        # Yield function for trial
        denom_D_old = 1.0 - D_old
        sigma_eff = J2_eff / denom_D_old
        fy_trial = sigma_eff - (R_old + self.params.sigma_0)

        # Plasticity flag
        is_plastic = jnp.where(fy_trial > 0.0, 1.0, 0.0)

        # Rates from previous step
        p_dot_old = state["p_dot"][0]
        eps_I_dot_old = state["eps_I_dot"]
        D_dot_old = state["D_dot"][0]
        R_dot_old = state["R_dot"][0]
        X_dot_old = state["X_dot"]

        # Operand for conditional branching
        operand = (
            eps, eps_old, sig_old, eps_I_old, eps_e_old, X_old,
            p_old, D_old, R_old, deps, sig_trial, sig_eff, sig_eff_dev, J2_eff, fy_trial, dt,
            p_dot_old, eps_I_dot_old, D_dot_old, R_dot_old, X_dot_old,
        )

        # Conditional execution
        (
            sig_new, eps_e_new, eps_I_new, X_new, eps_new,
            p_new, D_new, R_new,
            fy_new, p_dot_new, dp_new, D_dot_new, R_dot_new, X_dot_new, eps_I_dot_new, is_plastic_out,
        ) = jax.lax.cond(
            is_plastic,
            self._plastic_update,
            self._elastic_update,
            operand,
        )

        # Assemble updated state
        new_state = {
            "Strain": eps_new,
            "Stress": sig_new,
            "eps_I": eps_I_new,
            "eps_e": eps_e_new,
            "X": X_new,
            "p": jnp.array([p_new]),
            "D": jnp.array([D_new]),
            "R": jnp.array([R_new]),
            "fy": jnp.array([fy_new]),
            "p_dot": jnp.array([p_dot_new]),  # For next step
            "dp": jnp.array([dp_new]),
            "D_dot": jnp.array([D_dot_new]),
            "R_dot": jnp.array([R_dot_new]),
            "X_dot": X_dot_new,
            "eps_I_dot": eps_I_dot_new,
            "is_plastic": jnp.array([is_plastic_out]),
        }
        return new_state

    # ---------------------------------------------------------------------
    # State initialization helper
    # ---------------------------------------------------------------------
    @staticmethod
    def initial_state(stress: jnp.ndarray, strain: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """Create an initial state dictionary.

        Parameters
        ----------
        stress : jnp.ndarray, shape (6,)
            Initial stress (usually zeros).
        strain : jnp.ndarray, shape (6,)
            Initial strain (usually zeros).
        """
        zero_6 = jnp.zeros(6)
        zero_1 = jnp.array([0.0])
        return {
            "Strain": strain,
            "Stress": stress,
            "eps_I": zero_6,
            "eps_e": zero_6,
            "X": zero_6,
            "p": zero_1,
            "D": zero_1,
            "R": zero_1,
            "fy": zero_1,
            "p_dot": zero_1,
            "dp": zero_1,
            "D_dot": zero_1,
            "R_dot": zero_1,
            "X_dot": zero_6,
            "eps_I_dot": zero_6,
            "is_plastic": zero_1,
        }

# ---------------------------------------------------------------------
# Example parameter dataclass (to be adapted by the user)
# ---------------------------------------------------------------------
from dataclasses import dataclass

@dataclass
class MaterialParams:
    E: float
    nu: float
    sigma_0: float
    K: float
    m: float
    c1: float
    R_inf: float
    C_kin: jnp.ndarray
    gamma: float
    alpha: float
    Dc: float

    @property
    def C(self) -> jnp.ndarray:
        """Elastic stiffness matrix in Voigt notation for isotropic material.
        Returns a 6×6 matrix.
        """
        E = self.E
        nu = self.nu
        lmbda = (E * nu) / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        C = jnp.array(
            [
                [lmbda + 2 * mu, lmbda, lmbda, 0, 0, 0],
                [lmbda, lmbda + 2 * mu, lmbda, 0, 0, 0],
                [lmbda, lmbda, lmbda + 2 * mu, 0, 0, 0],
                [0, 0, 0, mu, 0, 0],
                [0, 0, 0, 0, mu, 0],
                [0, 0, 0, 0, 0, mu],
            ],
            dtype=jnp.float32,
        )
        return C

# ---------------------------------------------------------------------
# End of module
# ---------------------------------------------------------------------
