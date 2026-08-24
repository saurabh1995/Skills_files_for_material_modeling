---
name: large-strain-elastoplastic-implicit
description: Generate JAX-compatible large-strain rate-independent elastoplastic material models using the 15-component Kinematics interface, Cauchy stress push-forward predictor, and implicit radial-return plastic corrector. Use for large-strain von Mises/J2 models with generic isotropic or other user-specified hardening laws. Do not use for full multiplicative finite-strain plasticity with Fe/Fp evolution unless explicitly requested.
---

# Large-Strain Elastoplastic Implicit Material Skill

Use this skill to generate `dolfinx_materials` JAX material classes for the large-strain algorithm used in this project. The generated model must reproduce the existing incremental Cauchy-stress push-forward algorithm, not a full multiplicative finite-strain plasticity formulation.

This skill is independent from the small-strain elastoplastic skill. Reuse only the general return-mapping ideas: trial state, yield check, plastic branch, Newton residual, state write-back, and JAX-compatible `lax.cond`.

Do not generate a test file unless the user separately asks for one.

## Scope Boundary

The default generated model is large-strain J2/von Mises elastoplasticity with user-specified hardening. Keep the kinematic predictor unchanged for all generated variants.

If the user requests a non-J2 yield surface, do not silently reuse the `3.0 * mu * dlam * n_tr` radial-return correction. Ask for the intended large-strain correction rule or derive it explicitly from the supplied flow rule before coding.

## Target Algorithm

The material receives one registered gradient named `"Kinematics"` with 15 components:

```text
Kinematics[0:6]   = incremental Green-Lagrange strain vector
Kinematics[6:15]  = incremental deformation gradient F_inc, row-major 3x3
```

The strain vector uses engineering Voigt ordering compatible with the stress vector:

```text
[E11, E22, E33, 2E12, 2E13, 2E23]
```

The stress flux is always Cauchy stress in 6-component Voigt notation:

```text
[sigma11, sigma22, sigma33, sigma12, sigma13, sigma23]
```

The finite-strain update is the project-specific incremental algorithm:

```text
deps      = Kinematics[0:6]
F_inc     = reshape(Kinematics[6:15], (3, 3))
J         = det(F_inc)
delta_sig = C @ deps

sigma_tr = (1/J) F_inc delta_sig F_inc^T
         + (1/J) F_inc sigma_old F_inc^T

s_tr      = dev(sigma_tr)
sigma_eq  = sqrt(3/2 s_tr:s_tr)
f_tr      = sigma_eq - sigma_y(old internal variables)
```

If `f_tr < 0`, accept `sigma_tr` elastically. If `f_tr >= 0`, solve the rate-independent consistency residual:

```text
R(dlam) = f(sigma_new(dlam), updated internal variables(dlam)) = 0
sigma_new = sigma_tr - 3 mu dlam n_tr
n_tr = dev(sigma_tr) / sigma_eq_tr
```

Keep `n_tr` fixed at the trial state inside the Newton residual. This is a semi-implicit radial-return update in Cauchy stress space.

## Required Class Interface

Generated classes must inherit from `JAXMaterial` and use `@tangent_AD`.

Required imports:

```python
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from dolfinx_materials.material.jax import JAXMaterial, JAXNewton, tangent_AD
```

The parameter dataclass contains parameters only. Put no constitutive logic in it.

Required properties:

```python
@property
def gradient_names(self):
    return ("Kinematics",)

@property
def gradients(self):
    return {"Kinematics": 15}

@property
def flux_names(self):
    return ("Stress",)

@property
def fluxes(self):
    return {"Stress": 6}

@property
def tangent_blocks(self):
    return {("Stress", "Kinematics"): (6, 15)}
```

The minimum state is:

```python
@property
def internal_state_variables(self):
    return {
        "p": 1,
        "dlam": 1,
        "fy": 1,
    }
```

Add extra internal variables only when the user-provided equations require them, for example scalar isotropic hardening variables or tensor backstresses. Do not add `F_p`, `F_e`, `eps_p`, or `eps_e` for this algorithm unless the user explicitly changes the target formulation.

## Mandatory Helpers

Always include tensor/Voigt helpers using the exact ordering above:

```python
def _voigt_to_tensor(self, v):
    return jnp.array([
        [v[0], v[3], v[4]],
        [v[3], v[1], v[5]],
        [v[4], v[5], v[2]],
    ], dtype=v.dtype)

def _tensor_to_voigt(self, A):
    return jnp.array([
        A[0, 0],
        A[1, 1],
        A[2, 2],
        A[0, 1],
        A[0, 2],
        A[1, 2],
    ], dtype=A.dtype)

def _push_forward_stress(self, sig_vec, F_inc, J):
    sig_T = self._voigt_to_tensor(sig_vec)
    sig_pf_T = (F_inc @ sig_T @ F_inc.T) / J
    return self._tensor_to_voigt(sig_pf_T)
```

Always compute:

```python
C = self.elastic_model.C
mu = self._shear_modulus()
```

Do not manually reconstruct the elastic stiffness matrix when an elastic model is supplied.

Use a regularized equivalent stress:

```python
def _equivalent_stress(self, sig):
    s = self._deviatoric(sig)
    s_dot_s = (
        s[0]**2 + s[1]**2 + s[2]**2
        + 2.0 * (s[3]**2 + s[4]**2 + s[5]**2)
    )
    return jnp.sqrt(jnp.maximum(1.5 * s_dot_s, 1.0e-24))
```

## Constitutive Update Template

Use this structure for generated models. Adapt only parameter names, hardening laws, extra state variables, and the residual contents required by the user equations.

```python
@tangent_AD
def constitutive_update(self, gradients, state, dt):
    _ = dt

    eps = gradients[:6]
    F_inc_vec = gradients[6:15]
    deps = eps

    p_old = state["p"][0]
    sig_old = state["Stress"]

    C = self.elastic_model.C
    mu = self._shear_modulus()

    F_inc = F_inc_vec.reshape((3, 3))
    J = jnp.linalg.det(F_inc)

    delta_sig_trial = C @ deps
    sig_tr = (
        self._push_forward_stress(delta_sig_trial, F_inc, J)
        + self._push_forward_stress(sig_old, F_inc, J)
    )

    s_tr = self._deviatoric(sig_tr)
    sigma_eq_tr = self._equivalent_stress(sig_tr)
    sigma_y_old = self.yield_stress(p_old)
    yield_criterion = sigma_eq_tr - sigma_y_old
    n_tr = s_tr / jnp.clip(sigma_eq_tr, a_min=1.0e-8)

    operand = (p_old, sig_old, sig_tr, yield_criterion)

    def elastic_update(operand):
        p_old, sig_old, sig_tr, yield_criterion = operand
        sig_new = sig_tr
        p_new = p_old
        dlam = 0.0
        return sig_new, p_new, dlam, yield_criterion

    def plastic_update(operand):
        p_old, sig_old, sig_tr, yield_criterion = operand

        def R_plastic(dlam):
            sig_new = sig_tr - 3.0 * mu * dlam * n_tr
            p_new = p_old + dlam
            sigma_y_new = self.yield_stress(p_new)
            sigma_eq_new = self._equivalent_stress(sig_new)
            return sigma_eq_new - sigma_y_new

        newton = JAXNewton()
        newton.set_residual(R_plastic)
        dlam, _ = newton.solve(0.0)

        p_new = p_old + dlam
        sig_new = sig_tr - 3.0 * mu * dlam * n_tr
        return sig_new, p_new, dlam, yield_criterion

    is_plastic = yield_criterion >= 0.0
    sig_new, p_new, dlam, fy = jax.lax.cond(
        is_plastic, plastic_update, elastic_update, operand
    )

    state["Kinematics"] = gradients
    state["Stress"] = sig_new
    state["p"] = jnp.array([p_new])
    state["dlam"] = jnp.array([dlam])
    state["fy"] = jnp.array([fy])

    return sig_new, state
```

## Hardening Laws

Keep hardening general. Implement the user-provided hardening as a method such as:

```python
def yield_stress(self, p):
    ...
```

Examples:

```python
# Perfect plasticity
return self.params.sigma_0

# Linear isotropic hardening
return self.params.sigma_0 + self.params.H * p

# Voce saturation
return self.params.sigma_0 + (self.params.sigma_inf - self.params.sigma_0) * (
    1.0 - jnp.exp(-self.params.b * p)
)

# Combined linear + saturation
return self.params.sigma_0 + self.params.H * p + (
    self.params.sigma_inf - self.params.sigma_0
) * (1.0 - jnp.exp(-self.params.b * p))
```

Always solve the general Newton residual for `dlam`, even when a closed-form linear-hardening expression exists. Do not switch to a closed-form return unless the user explicitly requests that optimization.

For extra scalar hardening variables, update them implicitly inside `R_plastic` and recompute them after Newton with the converged `dlam`. For tensor backstresses, keep tensor shapes consistent in both `lax.cond` branches and evaluate the yield function with the correct relative stress. In this large-strain Cauchy-stress algorithm, a stress-like backstress from the old state must be pushed forward with the same incremental map as `sigma_old` before forming the trial relative stress:

```python
X_old_pf = self._push_forward_stress(X_old, F_inc, J)
X_old_pf = self._deviatoric(X_old_pf)
eta_tr = self._deviatoric(sig_tr - X_old_pf)
```

Use `X_old_pf` in both elastic and plastic branches. In the elastic branch, write the transported value back as the new backstress; in the plastic branch, use it as the old backstress entering the kinematic-hardening update. Keep the stored backstress deviatoric after transport and after the hardening update.

## JAX Rules

Follow these rules strictly:

- Put `JAXNewton()` and `newton.solve(0.0)` inside `plastic_update`.
- Call `newton.set_residual(R_plastic)` before `solve`.
- Do not put `jax.lax.cond` inside `R_plastic`.
- Both `elastic_update` and `plastic_update` must return the same number of values with matching shapes.
- Return `dlam = 0.0` as a scalar in the elastic branch; wrap it only when writing to `state`.
- Unpack scalar state variables with `[0]` and write them back as `jnp.array([value])`.
- Recompute all final state variables after Newton using the converged `dlam`; do not reuse values from inside `R_plastic`.
- Use `yield_criterion >= 0.0` for the plastic branch condition.
- Keep `n_tr` fixed from the trial state throughout the Newton solve.
- Avoid ordinary `print()` inside JAX-traced functions. Use `jax.debug.print` only when explicitly debugging.

## Large-Strain Pitfalls

Avoid these errors:

- Do not use `gradient_names = ("Strain",)` or a 6-component gradient for this algorithm.
- Do not compute `deps = eps - state["Strain"]`; the supplied first 6 kinematic components are already the incremental Green-Lagrange strain vector for the current update.
- Do not use the small-strain predictor `sig_tr = sig_old + C @ deps`.
- Do not forget to push forward both the old stress and the elastic trial increment.
- If the model has a stress-like tensor backstress, do not use `state["X"]` directly with the pushed-forward trial Cauchy stress. Push the old backstress forward with `F_inc` and `J`, then use the transported deviatoric backstress in the relative stress and branch updates.
- Do not reshape `F_inc` in column-major order; use `F_inc_vec.reshape((3, 3))` for the row-major vector from the notebook.
- Do not change the stress measure. The material returns Cauchy stress.
- Do not add finite-strain multiplicative-plasticity state variables unless the user asks for a different formulation.
- Do not declare `"fy"` in `internal_state_variables` without writing `state["fy"]`.
- Be careful with `J = det(F_inc)`. This algorithm assumes a physically admissible positive determinant.

## FEniCSx Coupling Notes

When generating companion usage code or explaining integration with `QuadratureMap`, register the gradient as:

```python
qmap.register_gradient("Kinematics", kinematics)
```

The kinematics vector should be assembled from:

```text
strain_vec = incremental Green-Lagrange strain from F_inc
F_inc_vec  = row-major vector of F_inc = F_{n+1} F_n^{-1}
```

For the weak form, convert returned Cauchy stress to a tensor and then to first Piola-Kirchhoff stress outside the material:

```text
sigma = Cauchy stress tensor from qmap.fluxes["Stress"]
P = J_total * sigma * inv(F_total).T
```

That conversion belongs in the FEniCSx problem, not inside the material model.
