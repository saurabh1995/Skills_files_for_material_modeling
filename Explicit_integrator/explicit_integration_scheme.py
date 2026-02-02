# -*- coding: utf-8 -*-
"""
Explicit integration scheme for a Lemaitre‑Chaboche material model.

The implementation follows the forward‑Euler explicit time integration
scheme described in the user’s request.  The file contains a single
``ExplicitIntegrator`` class which exposes a ``constitutive_update``
method that can be called each time step.

The code is deliberately straightforward – it uses only JAX primitives
and no external dependencies beyond the standard ``dataclasses``
module.  The implementation is safe for inclusion in a larger
simulation framework.

Note that the module is **not** meant to be executed directly;
it is intended to be imported by the surrounding skill runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

# ``tangent_AD`` is provided by the explicit‑integration skill
# framework.  It injects a JAX‑autograd wrapper that records the
# Jacobian of the constitutive update for linearisation.
from .tangent import tangent_AD


# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------
@dataclass
class ExplicitParams:
    """Material parameters for the explicit integration scheme."""

    # Elasticity
    E: float = 115000.0  # Young's modulus [MPa]
    nu: float = 0.33     # Poisson's ratio

    # Viscoplasticity (Perzyna law)
    sigma0: float = 220.0   # Initial yield stress [MPa]
    K_visc: float = 15.0    # Viscous modulus [MPa·s^(1/m)]
    m: float = 6.0          # Perzyna exponent

    # Isotropic hardening (exponential saturation)
    c1: float = 800.0       # Hardening rate
    R_inf: float = 12000.0  # Saturation stress

    # Kinematic hardening (Armstrong‑Frederick)
    C_kin: float = 8500.0   # Kinematic modulus [MPa]
    gamma: float = 250.0     # Back‑stress decay coefficient

    # Damage model
    alpha: float = 0.05     # Damage scaling
    Dc: float = 0.25        # Critical damage


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def elastic_modulus(E: float, nu: float) -> jnp.ndarray:
    """Return the isotropic stiffness matrix ``C`` in Voigt notation.

    Parameters
    ----------
    E : float
        Young's modulus [MPa].
    nu : float
        Poisson's ratio.

    Returns
    -------
    jnp.ndarray
        6×6 stiffness matrix.
    """
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    return jnp.array(
        [
            [lam + 2 * mu, lam, lam, 0, 0, 0],
            [lam, lam + 2 * mu, lam, 0, 0, 0],
            [lam, lam, lam + 2 * mu, 0, 0, 0],
            [0, 0, 0, mu, 0, 0],
            [0, 0, 0, 0, mu, 0],
            [0, 0, 0, 0, 0, mu],
        ],
        dtype=jnp.float32,
    )


def deviatoric(sig: jnp.ndarray) -> jnp.ndarray:
    """Return the deviatoric part of a stress tensor in Voigt notation.

    Parameters
    ----------
    sig : jnp.ndarray
        Stress tensor in Voigt notation (6 components).

    Returns
    -------
    jnp.ndarray
        Deviatoric stress tensor.
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


def J2(sig_dev: jnp.ndarray) -> jnp.ndarray:
    """Compute the second deviatoric invariant with a small
    regularisation to keep the gradient well‑defined.

    Parameters
    ----------
    sig_dev : jnp.ndarray
        Deviatoric stress tensor.

    Returns
    -------
    jnp.ndarray
        J2 value.
    """
    s = sig_dev
    s_dot_s = (
        s[0] ** 2
        + s[1] ** 2
        + s[2] ** 2
        + 2.0 * (s[3] ** 2 + s[4] ** 2 + s[5] ** 2)
    )
    val = 1.5 * s_dot_s
    val_pos = jnp.maximum(val, 0.0)
    eps_reg = 1e-16
    J2_reg = jnp.sqrt(val_pos + eps_reg)
    return jax.lax.stop_gradient(jnp.sqrt(val_pos) - J2_reg) + J2_reg


def equivalent_stress(sig: jnp.ndarray) -> jnp.ndarray:
    """Return the von Mises equivalent stress."""
    return jnp.sqrt(3.0 * J2(deviatoric(sig)))


def mc_cauley_bracket(x: jnp.ndarray) -> jnp.ndarray:
    """Smooth ``max(x, 0)`` for Perzyna rate law.

    Using ``0.5 * (x + |x|)`` yields the same result but keeps the
    function differentiable.
    """
    return 0.5 * (x + jnp.abs(x))


# -----------------------------------------------------------------------------
# Explicit integrator
# -----------------------------------------------------------------------------
class ExplicitIntegrator:
    """Explicit forward‑Euler integrator for the Lemaitre‑Chaboche model.

    The integrator expects the caller to maintain a state dictionary
    containing the variables listed in ``internal_state_variables``.
    """

    def __init__(self, params: ExplicitParams):
        self.params = params
        self.C = elastic_modulus(params.E, params.nu)

    # ``internal_state_variables`` is part of the skill API – it tells the
    # surrounding framework which entries must be initialised.
    @property
    def internal_state_variables(self):  # pragma: no cover
        return {
            "p": 1,  # Equivalent plastic strain
            "R": 1,  # Isotropic hardening
            "X": 6,  # Backstress tensor (Voigt)
            "eps_p": 6,  # Plastic strain tensor
            "eps_e": 6,  # Elastic strain tensor
            "D": 1,  # Damage variable
            "Stress": 6,  # Stress tensor
            "Strain": 6,  # Total strain tensor
        }

    @tangent_AD
    def constitutive_update(self, eps_new: jnp.ndarray, state: dict, dt: float):
        """Update the material state for a single time step.

        Parameters
        ----------
        eps_new : jnp.ndarray
            Total strain at the new time step (6‑component Voigt).
        state : dict
            Current state dictionary.  The function reads and updates the
            entries listed in ``internal_state_variables``.
        dt : float
            Time‑step size.

        Returns
        -------
        tuple
            Updated stress tensor and the updated state dictionary.
        """
        # Unpack old state
        eps_old = state["Strain"]
        deps = eps_new - eps_old

        sig_old = state["Stress"]
        eps_p_old = state["eps_p"]
        eps_e_old = state["eps_e"]
        p_old = state["p"][0]
        R_old = state["R"][0]
        X_old = state["X"]
        D_old = state["D"][0]

        # Elastic predictor
        sig_trial = sig_old + self.C @ deps

        # Effective stress used in the yield function
        sig_eff = sig_trial - X_old
        J2_eff = J2(deviatoric(sig_eff))
        denom_D = 1.0 - D_old
        f = J2_eff / denom_D - (R_old + self.params.sigma0)

        # Plasticity decision
        is_plastic = f > 0.0

        # Initialise with elastic values
        sig_new = sig_trial
        eps_e_new = eps_e_old + deps
        eps_p_new = eps_p_old
        p_new = p_old
        R_new = R_old
        X_new = X_old
        D_new = D_old

        def plastic_update(_):
            # Perzyna viscoplastic strain rate
            x = f / self.params.K_visc
            bracket = mc_cauley_bracket(x)
            p_dot = jnp.power(bracket, self.params.m)
            dp = p_dot * dt
            p_new = p_old + dp

            # Flow direction (normal to yield surface)
            sig_eff_dev = deviatoric(sig_eff)
            inv_J2 = jnp.where(J2_eff > 0.0, 1.0 / J2_eff, 0.0)
            flow_dir = sig_eff_dev * inv_J2

            # Inelastic strain rate and increment
            eps_I_dot = 1.5 * p_dot * flow_dir
            delta_eps_I = eps_I_dot * dt

            # Strain update
            eps_p_new = eps_p_old + delta_eps_I
            delta_eps_e = deps - delta_eps_I
            eps_e_new = eps_e_old + delta_eps_e
            eps_new = eps_e_new + eps_p_new

            # Stress update with damage scaling
            sig_new = sig_old + (1.0 - D_new) * self.C @ delta_eps_e

            # Isotropic hardening (exponential saturation)
            R_dot = self.params.c1 * (self.params.R_inf - R_old) * p_dot
            R_new = R_old + R_dot * dt

            # Kinematic hardening (Armstrong‑Frederick)
            X_dot = (
                (2.0 / 3.0) * self.params.C_kin * eps_I_dot
                - self.params.gamma * X_old * p_dot
            )
            X_new = X_old + X_dot * dt

            # Damage evolution
            sigma_eq = equivalent_stress(sig_new)
            D_dot = self.params.alpha / denom_D * sigma_eq / self.params.sigma0 * p_dot
            D_new = D_old + D_dot * dt
            D_new = jnp.clip(D_new, 0.0, self.params.Dc)

            return sig_new, eps_e_new, eps_new, eps_p_new, p_new, R_new, X_new, D_new

        # Conditionally execute plastic update
        sig_new, eps_e_new, eps_new, eps_p_new, p_new, R_new, X_new, D_new = jax.lax.cond(
            is_plastic,
            plastic_update,
            lambda _: (sig_new, eps_e_new, eps_new, eps_p_new, p_new, R_new, X_new, D_new),
            operand=None,
        )

        # Update state dictionary
        state["Strain"] = eps_new
        state["Stress"] = sig_new
        state["eps_p"] = eps_p_new
        state["eps_e"] = eps_e_new
        state["p"] = jnp.array([p_new])
        state["R"] = jnp.array([R_new])
        state["X"] = X_new
        state["D"] = jnp.array([D_new])

        return sig_new, state

# -----------------------------------------------------------------------------
# End of file
# -----------------------------------------------------------------------------
