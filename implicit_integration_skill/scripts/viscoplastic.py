import jax
import jax.numpy as jnp

# ------------------------------
# Material parameters (SI units)
# ------------------------------
E  = 210_000.0   # MPa
nu = 0.3
sigma0 = 300.0    # MPa
H = -5000.0       # MPa
gamma = 0.001     # MPa^{-N} s^{-1}
N = 2.0
DT = 1e-6         # s

# ------------------------------
# Elasticity tensor in Voigt notation (6x6)
# ------------------------------

def elasticity_tensor(E, nu):
    """Return the 6x6 isotropic stiffness matrix in MPa."""
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame
    mu = E / (2 * (1 + nu))
    C = jnp.array([
        [lam + 2*mu, lam, lam, 0, 0, 0],
        [lam, lam + 2*mu, lam, 0, 0, 0],
        [lam, lam, lam + 2*mu, 0, 0, 0],
        [0, 0, 0, mu, 0, 0],
        [0, 0, 0, 0, mu, 0],
        [0, 0, 0, 0, 0, mu],
    ])
    return C

C = elasticity_tensor(E, nu)

# ------------------------------
# Helper functions
# ------------------------------

def deviatoric(stress):
    """Return deviatoric part of stress vector (Voigt)."""
    trace = stress[0] + stress[1] + stress[2]
    mean = trace / 3.0
    # Stress deviator: s = sigma - mean*I
    s = stress.copy()
    s = s.at[0].set(stress[0] - mean)
    s = s.at[1].set(stress[1] - mean)
    s = s.at[2].set(stress[2] - mean)
    # shear components unchanged
    return s


def J2(stress):
    """Return the second deviatoric invariant J2 = sqrt(3/2 * s:s)."""
    s = deviatoric(stress)
    val = jnp.dot(s, s)  # s:s
    return jnp.sqrt(1.5 * val)


def flow_direction(stress):
    """∂f/∂σ, regularised if J2 == 0."""
    s = deviatoric(stress)
    J2_val = J2(stress)
    # Avoid division by zero: return zero vector if J2 is very small
    def zero_flow():
        return jnp.zeros_like(stress)
    def normal_flow():
        return (1.5 * s) / J2_val
    return jax.lax.cond(J2_val > 1e-12, normal_flow, zero_flow)


def yield_function(stress, kappa):
    """f(σ, κ) = J2(σ) - σ_y(κ)."""
    J2_val = J2(stress)
    sigma_y = sigma0 + H * kappa
    return J2_val - sigma_y


def overstress(f):
    """φ(f) = (f/σ0)^N for f>0, else 0."""
    def pos():
        return (f / sigma0) ** N
    def neg():
        return 0.0
    return jax.lax.cond(f > 0.0, pos, neg)

# ------------------------------
# Residual and Newton solver for Δλ
# ------------------------------

def residual_and_grad(delta_lambda, stress_trial, kappa_trial):
    """Compute residual R(Δλ) and its gradient w.r.t. Δλ.
    stress_trial: σ_trial from elastic predictor.
    kappa_trial: κ_trial (equal to κ_n + Δλ? We will update κ accordingly.)
    """
    # Flow direction based on current guess of stress (depends on Δλ)
    def sigma_new(dl):
        n = flow_direction(stress_trial - dl * (C @ flow_direction(stress_trial)))
        return stress_trial - dl * (C @ n)
    # However, the flow direction depends on σ_new, so we must use self-consistency.
    # For simplicity we use fixed-point: approximate n using stress_trial.
    n_fixed = flow_direction(stress_trial)
    sigma_new = stress_trial - delta_lambda * (C @ n_fixed)
    f_new = yield_function(sigma_new, kappa_trial)
    phi = overstress(f_new)
    R = delta_lambda - gamma * phi * DT
    # Use jax.grad to get derivative
    R_grad = jax.grad(lambda dl: residual_and_grad(dl, stress_trial, kappa_trial)[0])(delta_lambda)
    return R, R_grad

# To avoid circular dependency in residual, we implement a simple fixed-point residual instead

def residual_func(delta_lambda, stress_trial, kappa_trial):
    """Residual R(Δλ) = Δλ - γ φ(f_new) Δt."""
    # Flow direction from trial stress (first guess)
    n = flow_direction(stress_trial)
    sigma_new = stress_trial - delta_lambda * (C @ n)
    f_new = yield_function(sigma_new, kappa_trial)
    phi = overstress(f_new)
    return delta_lambda - gamma * phi * DT

# Newton–Raphson inner loop

def solve_delta_lambda(stress_trial, kappa_trial, tol=1e-8, max_iter=25):
    delta_lambda = 0.0
    for i in range(max_iter):
        R = residual_func(delta_lambda, stress_trial, kappa_trial)
        if jnp.abs(R) < tol:
            return delta_lambda, True
        # derivative via autograd
        dR_dl = jax.grad(lambda dl: residual_func(dl, stress_trial, kappa_trial))(delta_lambda)
        # Guard against zero derivative
        if jnp.abs(dR_dl) < 1e-12:
            return delta_lambda, False
        delta_lambda = delta_lambda - R / dR_dl
    return delta_lambda, False

# ------------------------------
# Constitutive update routine
# ------------------------------

def constitutive_update(
    stress_n, eps_n, eps_np1, epsp_n, kappa_n
):
    """Return updated stress, plastic strain, and kappa for the new time step.
    Inputs are in MPa, MPa, MPa, MPa, dimensionless respectively.
    """
    # Elastic predictor
    eps_plastic_increment_guess = eps_np1 - eps_n - epsp_n  # Not used directly
    eps_diff = eps_np1 - eps_n
    stress_trial = stress_n + C @ eps_diff
    # Check elastic/plastic branching
    f_trial = yield_function(stress_trial, kappa_n)
    def elastic_case():
        return stress_trial, epsp_n, kappa_n
    def plastic_case():
        # Solve for Δλ
        delta_lambda, converged = solve_delta_lambda(stress_trial, kappa_n)
        n = flow_direction(stress_trial)
        stress_new = stress_trial - delta_lambda * (C @ n)
        epsp_new = epsp_n + delta_lambda * n
        kappa_new = kappa_n + delta_lambda  # κ increments with Δλ
        return stress_new, epsp_new, kappa_new
    return jax.lax.cond(f_trial > 0.0, plastic_case, elastic_case)

# Example usage (not executed here)
# stress_n = jnp.array([0., 0., 0., 0., 0., 0.])
# eps_n   = jnp.zeros(6)
# eps_np1 = jnp.array([1e-4, 0., 0., 0., 0., 0.])
# epsp_n  = jnp.zeros(6)
# kappa_n = 0.0
# stress_new, epsp_new, kappa_new = constitutive_update(stress_n, eps_n, eps_np1, epsp_n, kappa_n)

