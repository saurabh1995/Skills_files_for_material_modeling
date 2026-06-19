# Common Yield Surfaces

## von Mises (J₂ Plasticity)

```python
def equivalent_stress_von_mises(sig):
    """
    σ̄ = √(3/2 s:s)
    
    For Voigt: s:s = s₁² + s₂² + s₃² + 2(s₁₂² + s₁₃² + s₂₃²)
    """
    s = deviatoric(sig)
    return jnp.sqrt(1.5 * (s[0]**2 + s[1]**2 + s[2]**2 + 
                           2.0*(s[3]**2 + s[4]**2 + s[5]**2)))
```

**Use:** Metals, ductile materials, pressure-independent

## Tresca

```python
def equivalent_stress_tresca(sig):
    """
    σ̄ = max(|σ₁ - σ₂|, |σ₂ - σ₃|, |σ₃ - σ₁|) / 2
    
    Requires principal stresses
    """
    # Convert Voigt to 3x3 tensor
    sig_tensor = voigt_to_tensor(sig)
    # Compute principal stresses
    principals = jnp.linalg.eigvalsh(sig_tensor)
    # Tresca criterion
    return (jnp.max(principals) - jnp.min(principals)) / 2.0
```

**Use:** Upper bound for von Mises, theoretical studies

## Drucker-Prager

```python
def equivalent_stress_drucker_prager(sig, alpha, k):
    """
    σ̄ = √J₂ + α·I₁ - k
    
    where:
    - J₂ = (1/2)s:s
    - I₁ = tr(σ)
    - α = friction parameter
    - k = cohesion
    """
    s = deviatoric(sig)
    J2 = 0.5 * jnp.sum(s**2)
    I1 = sig[0] + sig[1] + sig[2]
    return jnp.sqrt(J2) + alpha * I1 - k
```

**Use:** Geomaterials, soils, concrete, pressure-dependent yield

## Mohr-Coulomb

```python
def equivalent_stress_mohr_coulomb(sig, phi, c):
    """
    f = (σ₁ - σ₃) + (σ₁ + σ₃)sin(φ) - 2c·cos(φ)
    
    where:
    - φ = friction angle
    - c = cohesion
    - σ₁, σ₃ = max/min principal stresses
    """
    sig_tensor = voigt_to_tensor(sig)
    principals = jnp.linalg.eigvalsh(sig_tensor)
    sig1 = jnp.max(principals)
    sig3 = jnp.min(principals)
    
    return (sig1 - sig3) + (sig1 + sig3)*jnp.sin(phi) - 2*c*jnp.cos(phi)
```

**Use:** Soils, granular materials, rocks

## Hosford

```python
def equivalent_stress_hosford(sig, a):
    """
    σ̄ = (|σ₁-σ₂|ᵃ + |σ₂-σ₃|ᵃ + |σ₃-σ₁|ᵃ)^(1/a)
    
    Special cases:
    - a = 2: von Mises
    - a → ∞: Tresca
    - a = 6, 8: Aluminum
    """
    sig_tensor = voigt_to_tensor(sig)
    principals = jnp.linalg.eigvalsh(sig_tensor)
    s1, s2, s3 = principals[0], principals[1], principals[2]
    
    return jnp.power(
        jnp.abs(s1 - s2)**a + 
        jnp.abs(s2 - s3)**a + 
        jnp.abs(s3 - s1)**a,
        1.0/a
    )
```

**Use:** Anisotropic yield, FCC metals, calibration to experiments

## Hill48 (Anisotropic)

```python
def equivalent_stress_hill48(sig, F, G, H, L, M, N):
    """
    σ̄² = F(σ₂₂-σ₃₃)² + G(σ₃₃-σ₁₁)² + H(σ₁₁-σ₂₂)² + 
          2L·σ₂₃² + 2M·σ₁₃² + 2N·σ₁₂²
    
    Anisotropic von Mises
    """
    return jnp.sqrt(
        F*(sig[1]-sig[2])**2 + G*(sig[2]-sig[0])**2 + H*(sig[0]-sig[1])**2 +
        2*L*sig[5]**2 + 2*M*sig[4]**2 + 2*N*sig[3]**2
    )
```

**Use:** Sheet metal forming, rolled products, textured materials

## Gurson (Porous Plasticity)

```python
def equivalent_stress_gurson(sig, f, q1, q2, sigma_0):
    """
    Φ = (σ_eq/σ₀)² + 2q₁f·cosh(q₂·σ_H/(2σ₀)) - 1 - (q₁f)²
    
    where:
    - f = void volume fraction
    - σ_H = hydrostatic stress
    - q₁, q₂ = Tvergaard parameters
    """
    sig_eq = equivalent_stress_von_mises(sig)
    sig_H = (sig[0] + sig[1] + sig[2]) / 3.0
    
    term1 = (sig_eq / sigma_0)**2
    term2 = 2*q1*f * jnp.cosh(q2 * sig_H / (2*sigma_0))
    term3 = 1 + (q1*f)**2
    
    return term1 + term2 - term3
```

**Use:** Ductile fracture, void growth, porous metals

## Comparison Table

| Surface | Type | Pressure | Corners | Complexity |
|---------|------|----------|---------|------------|
| von Mises | Smooth | Independent | No | Low |
| Tresca | Singular | Independent | 6 | Medium |
| Drucker-Prager | Smooth | Dependent | No | Low |
| Mohr-Coulomb | Singular | Dependent | 6 | High |
| Hosford | Smooth | Independent | No | Medium |
| Hill48 | Smooth | Independent | No | Medium |
| Gurson | Smooth | Dependent | No | High |

## Notes on Singular Surfaces

Tresca and Mohr-Coulomb have corners (non-differentiable points).
Requires special handling:
- Smoothing at corners
- Multi-surface plasticity
- Vertex treatment

For practical implementation, use smoothed versions or Drucker-Prager approximation.
