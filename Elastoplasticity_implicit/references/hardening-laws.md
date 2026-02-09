# Common Hardening Laws

All functions return R(p), the yield strength as function of cumulated plastic strain p.

## Linear Hardening

```python
def yield_stress_linear(p, sigma_0, H):
    """
    R(p) = σ₀ + H·p
    
    Parameters:
    - σ₀: Initial yield stress
    - H: Hardening modulus
    """
    return sigma_0 + H * p
```

**Characteristics:**
- Constant hardening rate
- Simple, commonly used
- Good for small strains

## Exponential Saturation (Voce Law)

```python
def yield_stress_voce(p, sigma_0, sigma_inf, b):
    """
    R(p) = σ₀ + (σ∞ - σ₀)(1 - exp(-b·p))
    
    Parameters:
    - σ₀: Initial yield stress
    - σ∞: Saturation yield stress
    - b: Saturation rate
    """
    return sigma_0 + (sigma_inf - sigma_0) * (1.0 - jnp.exp(-b * p))
```

**Characteristics:**
- Saturates to σ∞
- Common for metals
- Good for large strains

## Power Law (Hollomon)

```python
def yield_stress_power(p, sigma_0, K, n):
    """
    R(p) = σ₀ + K·p^n
    
    Parameters:
    - σ₀: Initial yield stress
    - K: Strength coefficient
    - n: Strain hardening exponent (0 < n < 1)
    """
    return sigma_0 + K * jnp.power(p, n)
```

**Characteristics:**
- Continuously increasing
- Models strain hardening
- n ~ 0.2-0.5 for metals

## Swift Law

```python
def yield_stress_swift(p, K, eps_0, n):
    """
    R(p) = K(ε₀ + p)^n
    
    Parameters:
    - K: Strength coefficient
    - ε₀: Pre-strain
    - n: Hardening exponent
    """
    return K * jnp.power(eps_0 + p, n)
```

**Characteristics:**
- Accounts for pre-strain
- Used in metal forming
- Good fit for large strains

## Ludwik Law

```python
def yield_stress_ludwik(p, sigma_0, K, n):
    """
    R(p) = σ₀ + K·p^n
    
    (Same as power law but different parameterization)
    """
    return sigma_0 + K * jnp.power(p, n)
```

## Combined Linear-Saturation

```python
def yield_stress_combined(p, sigma_0, H, sigma_inf, b):
    """
    R(p) = σ₀ + H·p + (σ∞ - σ₀)(1 - exp(-b·p))
    
    Combines linear hardening with saturation
    """
    linear = H * p
    saturation = (sigma_inf - sigma_0) * (1.0 - jnp.exp(-b * p))
    return sigma_0 + linear + saturation
```

**Characteristics:**
- Linear + saturation effects
- More flexible fitting
- Common for advanced models

## Johnson-Cook (Temperature/Rate Dependent)

```python
def yield_stress_johnson_cook(p, eps_dot, T, 
                                sigma_0, B, n, C, m, T_ref, eps_dot_ref):
    """
    R(p, ε̇, T) = (σ₀ + B·p^n)(1 + C·ln(ε̇/ε̇₀))(1 - ((T-T_ref)/(T_melt-T_ref))^m)
    
    Includes strain rate and temperature effects
    """
    hardening = sigma_0 + B * jnp.power(p, n)
    rate_effect = 1.0 + C * jnp.log(eps_dot / eps_dot_ref)
    temp_effect = 1.0 - jnp.power((T - T_ref) / (T_melt - T_ref), m)
    return hardening * rate_effect * temp_effect
```

**Note:** For rate-independent plasticity, omit rate and temperature terms.

## Comparison Table

| Law | Parameters | Behavior | Use Case |
|-----|------------|----------|----------|
| Linear | 2 | Constant rate | Small strain, simple |
| Voce | 3 | Saturation | Metals, large strain |
| Power | 3 | Increasing | General metals |
| Swift | 3 | Pre-strain aware | Metal forming |
| Combined | 4 | Linear + saturation | Complex behavior |
| Johnson-Cook | 7 | Rate/temp dependent | High strain rate |

## Choosing Hardening Law

**For metals:**
- Small strain (< 5%): Linear
- Large strain (> 20%): Voce or Swift
- Intermediate: Power law

**For calibration:**
- 1-2 parameters: Limited data
- 3-4 parameters: Standard testing
- 5+ parameters: Extensive testing

## Derivatives for Newton Solver

The Newton solver needs dR/dp. JAX computes this automatically via AD:

```python
# No manual derivatives needed!
yield_stress = lambda p: sigma_0 + H * p

# JAX automatically computes dR/dp in Newton solver
# via jax.jacfwd or jax.grad
```

## Example: Creating Custom Hardening

```python
def my_custom_hardening(p, params):
    """
    Custom hardening law with arbitrary complexity
    """
    sigma_0, H1, H2, b1, b2 = params
    
    # Multi-stage hardening
    term1 = H1 * p * jnp.exp(-b1 * p)
    term2 = H2 * (1.0 - jnp.exp(-b2 * p))
    
    return sigma_0 + term1 + term2

# Use in material
yield_stress = lambda p: my_custom_hardening(p, my_params)
material = vonMisesIsotropicHardening(elastic, yield_stress)
```

## Calibration Tips

1. **Uniaxial tension test**: Provides σ vs ε curve
2. **Convert to R(p)**: For uniaxial, σ = R(ε_p)
3. **Fit parameters**: Use least squares or optimization
4. **Verify**: Check R'(p) > 0 (hardening not softening)

## Typical Parameter Ranges

**Structural steel:**
- σ₀: 250-400 MPa
- H: 1,000-10,000 MPa (linear)
- σ∞: 400-600 MPa (Voce)
- b: 10-100 (Voce)

**Aluminum:**
- σ₀: 100-300 MPa
- H: 500-5,000 MPa
- n: 0.2-0.3 (power law)

**Copper:**
- σ₀: 50-150 MPa
- H: 1,000-5,000 MPa
- Saturation behavior common
