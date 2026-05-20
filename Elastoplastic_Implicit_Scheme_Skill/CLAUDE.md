# Notes For Future Generation: `von_mises_linear_isotropic_implicit.py`

This file records the concrete failures encountered while generating and testing
`von_mises_linear_isotropic_implicit.py` and
`test_von_mises_linear_isotropic_implicit.py`.

## Target Model

- File: `Elastoplasticity_implicit/von_mises_linear_isotropic_implicit.py`
- Model: small-strain von Mises plasticity
- Hardening: linear isotropic
- Internal variable stored by model: `alpha`
- Governing plastic consistency equation:

```text
sigma_eq_tr - 3*mu*delta_lambda - (sigma_0 + H*(alpha_old + delta_lambda)) = 0
```

## Errors That Happened

### Error 1: `JAXNewton` object had no attribute `r`

Observed traceback:

```text
AttributeError: 'JAXNewton' object has no attribute 'r'
```

Meaning:

- The installed `dolfinx_materials` build did not match the assumed `JAXNewton`
  constructor behavior.
- Code generation assumed `newton = JAXNewton(residual)` was always valid.
- In the tested environment, `solve()` expected `self.r` and the object was not
  initialized in the expected way.

Rule:

- Do not assume `JAXNewton` API compatibility across environments without
  verifying the installed implementation.

### Error 2: `TypeError` inside `jax_newton_solver.py`

Observed traceback:

```text
TypeError: unsupported operand type(s) for *: 'function' and 'DynamicJaxprTracer'
```

Meaning:

- In the tested environment, the first argument passed to `JAXNewton(...)` was
  effectively being treated as solver parameters rather than as the residual.
- That caused `params.rtol` logic in the Newton solver to receive a function
  object instead of numeric tolerance data.

Rule:

- For this model, do not use `JAXNewton` unless the exact installed API has been
  inspected and confirmed.

## Correct Fix For This Model

For von Mises plasticity with linear isotropic hardening, do not use Newton at
all. Use the closed-form radial-return update.

Plastic branch:

```text
delta_lambda = f_tr / (3*mu + H)
```

with

```text
f_tr = sigma_eq_tr - (sigma_0 + H*alpha_old)
```

Elastic branch:

```text
delta_lambda = 0
```

Implementation rule:

- Use `jax.lax.cond(...)` for elastic/plastic branching.
- Keep the update JAX-compatible.
- Avoid solver dependency for this specific linear hardening model.

## Required Pattern For This File

Use this structure in future generations:

```python
f_tr = sig_eq_tr - self.yield_stress(alpha_old)

def elastic_branch(_):
    return jnp.array(0.0, dtype=eps.dtype)

def plastic_branch(_):
    return f_tr / (3.0 * mu + self.params.H)

delta_lambda = jax.lax.cond(f_tr < 0.0, elastic_branch, plastic_branch, operand=None)
delta_eps_p = 1.5 * n_tr * delta_lambda
sig = sig_tr - 2.0 * mu * delta_eps_p
state["alpha"] = jnp.array([alpha_old + delta_lambda], dtype=eps.dtype)
```

## Test-Side Notes

- The generated test uses algebraic checkers derived from the user equations.
- The test maps the model state variable `alpha` to checker variable `p`.
- The target model interface still requires `constitutive_update(eps, state, dt)`.
- The constitutive equations remain rate-independent; passing `dt=0.0` at the
  test call site is acceptable for this model interface.

## Do Not Repeat

- Do not assume every implicit elastoplastic model should use `JAXNewton`.
- Do not assume the local `dolfinx_materials` solver API matches template code.
- For linear isotropic von Mises return mapping, prefer the exact closed-form
  solution over Newton.

Project note: the `von_mises_linear_isotropic_implicit.py` Newton/API failures
and the closed-form replacement rule are intentionally documented in this
`CLAUDE.md` section for reuse in future generations.

---

# Notes For Future Generation: `j2_armstrong_frederick_implicit.py`

This section records the concrete mistakes corrected in
`j2_armstrong_frederick_implicit_corrected.py` relative to the originally
generated `j2_armstrong_frederick_implicit.py`.

## Target Model

- File: `j2_armstrong_frederick_implicit.py`
- Model: small-strain J2 plasticity
- Hardening: Armstrong-Frederick kinematic hardening
- Internal variables stored by model: `p`, `X`
- Governing plastic consistency equation:

```text
R(delta_lambda) = sigma_eq(dev(sigma_new - X_new)) - sigma_y = 0
```

with

```text
sigma_new = sigma_tr - 3*mu*delta_lambda*n_tr
X_new = (X_old + C_k*delta_lambda*n_tr) / (1 + gamma*delta_lambda)
```

## Errors That Happened

### Error 1: Incorrect relative stress inside the plastic residual

Original wrong pattern:

```python
eta_new = self._deviatoric(sig_new) - X_new
```

Correct pattern:

```python
eta_new = self._deviatoric(sig_new - X_new)
```

Meaning:

- The yield function is defined from the deviatoric part of the relative stress,
  not from `dev(sig)` minus the full backstress afterwards.
- Applying `_deviatoric` before subtracting `X_new` is only equivalent if `X_new`
  is already guaranteed to be purely deviatoric.
- Future generations should not rely on that assumption in the residual.

Rule:

- For J2 plasticity with backstress, always form the effective stress as
  `eta = dev(sig - X)` before evaluating the equivalent stress.
- In code, write `self._deviatoric(sig_new - X_new)`, not
  `self._deviatoric(sig_new) - X_new`.

### Error 2: Replacing `JAXNewton` with a custom Newton solver unnecessarily

Original generated pattern:

- Introduced a custom `_newton_solve_scalar(...)` helper.
- Avoided `JAXNewton` based on the older von Mises notes.

Corrected pattern:

```python
newton = JAXNewton()
newton.set_residual(residual)
dlam, _ = newton.solve(0.0)
```

Meaning:

- The earlier `JAXNewton` incompatibility notes were specific to a different
  generation pattern and constructor usage.
- For this Armstrong-Frederick model, the local environment works with the
  `JAXNewton()` plus `set_residual(...)` API.
- The custom local Newton helper should not be introduced by default when the
  standard solver API is usable.

Rule:

- For implicit elastoplastic models in this repo, prefer:
  `newton = JAXNewton(); newton.set_residual(residual); dlam, _ = newton.solve(0.0)`.
- Only fall back to a custom Newton solver if this exact API is shown to fail in
  the current target model.

## Correct Fix For This Model

Use the standard `JAXNewton` pattern and compute the residual from the
deviatoric relative stress:

```python
def residual(dlam):
    sig_new = sig_tr - 3.0 * mu * dlam * n_tr
    X_new = (X_old + self.params.C_k * dlam * n_tr) / (
        1.0 + self.params.gamma * dlam
    )
    eta_new = self._deviatoric(sig_new - X_new)
    sigma_eq_new = self._equivalent_stress_from_relative(eta_new)
    return sigma_eq_new - self.params.sigma_y

newton = JAXNewton()
newton.set_residual(residual)
dlam, _ = newton.solve(0.0)
```

## Do Not Repeat

- Do not compute the effective stress in the residual as
  `dev(sig_new) - X_new`.
- Do not add a custom Newton solver when `JAXNewton()` with
  `set_residual(...)` works for the target model.
- Do not over-generalize the older von Mises `JAXNewton` failure notes to every
  implicit elastoplastic model in this repo.
