# Skill Files for Generating Material Models

A Skills repository for generating constitutive material model integration schemes for FEniCSx-JAX workflows for constitutive laws.

## What is it about?

The repository collects reusable AI-assistant skills for deriving and implementing material models from governing equations. The emphasis is on mechanics workflows where the assistant must first derive the integration algorithm, then generate JAX-compatible Python code that follows the `dolfinx_materials` style.



## Contents

```text
.
|-- .claude-plugin/
|   `-- plugin.json
|-- Elastoplastic_Implicit_Scheme_Skill/
|   |-- SKILL.md
|   |-- CLAUDE.md
|   `-- references/
|-- Viscoplastic_Explicit_Scheme_Skill/
|   |-- SKILL.md
|   |-- CLAUDE.md
|   |-- explicit-integration-scheme.skill
|   `-- references/
`-- Viscoplastic_Implicit_Scheme_Skill/
    |-- SKILL.md
    `-- references/
```

## Skills

### Elastoplastic implicit scheme

Location: `Elastoplastic_Implicit_Scheme_Skill/`

Generates implicit return-mapping algorithms for rate-independent elastoplastic constitutive models. The skill is built around a derivation-first workflow:

1. Write the continuous governing equations.
2. Build the elastic predictor and yield check.
3. Express the stress update in terms of the plastic multiplier increment.
4. Derive implicit scalar and tensor hardening updates.
5. Form the scalar consistency residual.
6. Generate complete JAX-compatible code using a Newton solver.

Supported model families include von Mises plasticity, Drucker-Prager, Mohr-Coulomb-style custom yield surfaces, isotropic hardening, kinematic hardening, and mixed hardening.

Key references:

- `references/von-mises-template.md` - mandatory implementation template for von Mises return mapping.
- `references/general-template.md` - guidance for non-von-Mises yield surfaces.
- `references/hardening-laws.md` - common hardening law forms.
- `references/yield-surfaces.md` - yield surface reference material.

### Viscoplastic explicit scheme

Location: `Viscoplastic_Explicit_Scheme_Skill/`

Generates explicit time integration schemes for constitutive material models in the style of Lemaitre-Chaboche and Perzyna viscoplasticity. The skill enforces a two-phase explicit update:

1. Advance state variables using rates stored from the previous time step.
2. Re-evaluate all rates at the newly updated state and store them for the next time step.

This is intended for FEniCSx-JAX-compatible material implementations with elastic predictor, yield check, plastic/viscoplastic corrector logic, and explicit rate storage.

Key references:

- `references/code-template.md` - FEniCSx-JAX implementation template.
- `references/equation-mapping.md` - mapping from constitutive equations to code variables.
- `references/example-models.md` - example material model patterns.
- `references/plane-stress-extension.md` - plane-stress extension guidance.

### Viscoplastic implicit scheme

Location: `Viscoplastic_Implicit_Scheme_Skill/`

Generates fully implicit backward-Euler integration schemes for viscoplastic constitutive models. The workflow converts differential equations into a scalar residual in terms of the plastic multiplier increment and solves it with Newton iteration.

Covered model families include:

- Lemaitre-Chaboche viscoplasticity.
- Perzyna overstress models.
- Duvaut-Lions relaxation models.
- Consistency-type rate-dependent models.
- User-defined viscoplastic models with isotropic and/or kinematic hardening.

Key references:

- `references/code-template.md` - complete code generation template.
- `references/quick-reference.md` - model identification tree, residual snippets, and debugging notes.
- `references/box-algorithms.md` - mathematical algorithm boxes for supported model types.

## Typical Workflow

Use the skills when you have a constitutive model defined by equations and need an implementation-ready integration algorithm.

A typical assistant workflow is:

1. Identify the model class: elastoplastic, explicit viscoplastic, or implicit viscoplastic.
2. Read the corresponding `SKILL.md`.
3. Follow the required derivation steps before writing code.
4. Use the reference templates in that skill directory.
5. Generate FEniCSx-JAX-compatible material code.
6. Verify that all internal state variables are declared, updated, and written back.

## Usage

Step 1 - Clone the repository.

Step 2 - In Codex CLI, Claude Code, or Copilot, attach the required skill file for the material model you want to generate.

Step 3 - Use an example prompt like:

```text
Refer to Elastoplasticity/SKILL.md and create a new file for generating elastoplasticity with an implicit scheme using FEniCSx-JAX-compatible code for the following equations:

Strain decomposition:
(Equation)

Trial stress:
(Equation)

Relative stress:
(Equation)

Equivalent stress:
(Equation)

Yield function:
(Equation)

Plastic flow:
(Equation)

Backstress evolution:
(Equation)

Stress update:
(Equation)
```

Step 4 - Test the generated material file in your FEniCSx environment.

## Expected Inputs

The skills work best when the model definition includes:

- Strain decomposition.
- Elastic law.
- Yield or overstress function.
- Flow rule.
- Isotropic hardening laws.
- Kinematic hardening laws or backstress evolution equations.
- Damage evolution equations, if present.
- Time integration preference, if known.

## Generated Code Style

Generated implementations are expected to follow these conventions:

- Use JAX and `jax.numpy`.
- Use `dolfinx_materials.material.jax` material classes and decorators.
- Store all internal state variables explicitly.
- Use Voigt 6-vector tensor conventions where required by the templates.
- Use `jax.lax.cond` for elastic/plastic branching.
- Use Newton solves for implicit residuals.
- Keep derivation and code aligned so each residual term can be traced back to the equations.

## Plugin Metadata

The plugin metadata is stored in `.claude-plugin/plugin.json`:

```json
{
  "name": "fenicsx-mechanics-suite",
  "description": "Unified plugin containing explicit/implicit constitutive integration skills for FEniCSx.",
  "version": "1.0.0",
  "author": {
    "name": "Ameya"
  }
}
```

## Notes

This repository is a skill and reference collection. It is not a standalone Python package. The generated material-model code is intended to be copied into, or generated directly inside, a FEniCSx/dolfinx-materials project and then tested with the target simulation setup.

## About the author 

Dr.-Ing Saurabh Tandale is a PostDoc at Institute of general Mechanics, RWTH Aachen University, whose work develops neural-network-enhanced FEM, self-learning mechanics solvers, and neuromorphic or FPGA-based AI methods to make nonlinear simulations faster and more energy-efficient. 

Email : tandale@iam.rwth-aachen.de 

