# Algorithm Boxes from Wang et al. (1997)

This reference contains the three main stress-update algorithms from "Viscoplasticity for instabilities due to strain softening and strain-rate softening" by Wang, Sluys, and de Borst.

## Box 1: One-Step Euler Stress-Update Algorithm for Perzyna Model

```
Given: Δεₜ₊Δₜ = B Δaₜ₊Δₜ

1. Compute trial stress:
   σᵗʳⁱᵃˡ = σₜ + Dᵉ Δεₜ₊Δₜ

2. Check yield condition:
   IF f(σᵗʳⁱᵃˡ, κₜ) ≤ 0:  → ELASTIC STATE
      σₜ₊Δₜ = σᵗʳⁱᵃˡ
      
   ELSE:  → PLASTIC STATE
   
3. Compute viscoplastic tangent contributions:
   Gₜ = γ[∂φ/∂σ nᵀ + φ ∂²f/∂σ²]ₜ
   
4. Compute hardening rate contribution:
   hₜ = γ[∂φ/∂κ n]ₜ
   
5. Compute algorithmic tangent:
   D♯ = [Dᵉ⁻¹ + θΔt Gₜ]⁻¹
   
6. Compute pseudo-force:
   Δq = D♯(ε̇ᵛᵖₜ Δt + θΔt hₜ Δκ⁽ᴵ⁾)
   
7. Update stress:
   σₜ₊Δₜ = σₜ + D♯ Δεₜ₊Δₜ - Δq
   
8. Evaluate yield function:
   f = f(σₜ₊Δₜ, κₜ + Δκ⁽ᴵ⁾)
   
9. Compute viscoplastic strain rate:
   ε̇ᵛᵖ = γφ(σₜ₊Δₜ, κₜ + Δκ) ∂f/∂σ
   
10. Compute internal variable rate:
    κ̇ₜ₊Δₜ = √(2/3) ε̇ᵛᵖ : ε̇ᵛᵖ
    
11. Update internal variable:
    Δκ⁽ᴵ⁺¹⁾ = [(1-θ) κ̇ₜ + θ κ̇ₜ₊Δₜ] Δt

Notes:
- θ = integration parameter (0 = explicit, 1 = fully implicit)
- (I) = global iteration counter
- φ = overstress function, e.g., φ = (f/σ₀)ᴺ
```

## Box 2: Iterative Implicit Stress-Update Algorithm for Perzyna Model

```
Given: Δεₜ₊Δₜ = B Δaₜ₊Δₜ

1. Compute trial stress:
   σᵗʳⁱᵃˡ = σₜ + Dᵉ Δεₜ₊Δₜ

2. Check yield condition:
   IF f(σᵗʳⁱᵃˡ, κₜ) ≤ 0:  → ELASTIC STATE
      σₜ₊Δₜ = σᵗʳⁱᵃˡ
      
   ELSE:  → PLASTIC STATE

3. Initialize:
   Δλ⁽⁰⁾ = 0
   σₜ₊Δₜ⁽⁰⁾ = σₜ + Dᵉ[Δε - Δλ⁽⁰⁾ ∂f/∂σ]
   r⁽⁰⁾ = φ(σₜ₊Δₜ⁽⁰⁾, κₜ + Δλ⁽⁰⁾) - Δλ⁽⁰⁾/(γΔt)

4. Local iteration loop:

   a) Compute pseudo-elastic stiffness:
      H = [Dᵉ⁻¹ + Δλ⁽ⁱ⁾ ∂²f/∂σ²]⁻¹
   
   b) Compute denominator:
      a = [∂φ/∂σ]ᵀ H [Δλ⁽ⁱ⁾ ∂²f/∂σ∂κ + ∂f/∂σ] + 1/(γΔt) - ∂φ/∂κ
   
   c) Update plastic multiplier:
      Δλ⁽ⁱ⁺¹⁾ = Δλ⁽ⁱ⁾ + r⁽ⁱ⁾/a
   
   d) Update stress:
      σₜ₊Δₜ⁽ⁱ⁺¹⁾ = σₜ + Dᵉ[Δε - Δλ⁽ⁱ⁺¹⁾ ∂f/∂σ]
   
   e) Update residual:
      r⁽ⁱ⁺¹⁾ = φ(σₜ₊Δₜ⁽ⁱ⁺¹⁾, κₜ + Δλ⁽ⁱ⁺¹⁾) - Δλ⁽ⁱ⁺¹⁾/(γΔt)
   
   f) Check convergence:
      IF |r⁽ⁱ⁺¹⁾| < tolerance:
         EXIT loop
      ELSE:
         GO TO step 4a

Notes:
- This is a fully implicit algorithm (θ = 1)
- Uses local Newton-Raphson iteration
- Converges quadratically near solution
```

## Box 3: One-Step Implicit Stress-Update Algorithm for Duvaut-Lions Model

```
Given: Δεₜ₊Δₜ = B Δaₜ₊Δₜ

1. Compute trial stress:
   σᵗʳⁱᵃˡ = σₜ + Dᵉ Δεₜ₊Δₜ

2. Check yield condition:
   IF f(σᵗʳⁱᵃˡ, κₜ) ≤ 0:  → ELASTIC STATE
      σₜ₊Δₜ = σᵗʳⁱᵃˡ
      
   ELSE:  → PLASTIC STATE

3. Compute backbone (inviscid) stress using return mapping:
   σ̄ₜ₊Δₜ = σ̄ₜ + D♯¹ Δε
   
   where D♯¹ = Dᵉ - (Dᵉ n n̄ᵀ Dᵉ)/(h + n̄ᵀ Dᵉ n̄)
   and n̄ = ∂f/∂σ̄

4. Update backbone internal variable:
   κ̄ₜ₊Δₜ = κ̄ₜ + Δκ̄ₜ₊Δₜ Δt

5. Compute algorithmic tangent:
   D♯ = τ/(τ + θΔt) [Dᵉ + (θΔt/τ) D♯¹]

6. Compute pseudo-force:
   Δq = τΔt/(τ + θΔt) [(1-θ) Dᵉ ε̇ᵛᵖₜ + (θ/τ)(σₜ - σ̄ₜ)]

7. Update stress:
   σₜ₊Δₜ = σₜ + D♯ Δε - Δq

8. Compute viscoplastic strain rate:
   ε̇ᵛᵖₜ₊Δₜ = (1/τ) Dᵉ⁻¹(σₜ₊Δₜ - σ̄ₜ₊Δₜ)

9. Compute internal variable rate:
   κ̇ₜ₊Δₜ = √[(ε̇ᵛᵖₜ₊Δₜ)ᵀ A ε̇ᵛᵖₜ₊Δₜ]
   
   where A = diag[2/3, 2/3, 2/3, 1/3, 1/3, 1/3]

10. Update internal variable:
    κₜ₊Δₜ = κ̄ₜ₊Δₜ - τ κ̇ₜ₊Δₜ

Notes:
- τ = relaxation time (viscosity parameter)
- σ̄ = backbone (inviscid) stress
- Two-step process: first compute inviscid response, then viscoplastic relaxation
```

## Box 4: Fully Implicit Stress-Update Algorithm for Consistency Model

```
Given: Δεₜ₊Δₜ = B Δaₜ₊Δₜ

1. Compute trial stress:
   σᵗʳⁱᵃˡ = σₜ + Dᵉ Δεₜ₊Δₜ

2. Check yield condition:
   IF f(σᵗʳⁱᵃˡ, κₜ, κ̇ₜ) ≤ 0:  → ELASTIC STATE
      σₜ₊Δₜ = σᵗʳⁱᵃˡ
      
   ELSE:  → PLASTIC STATE

3. Initialize:
   Δλ⁽⁰⁾ = 0
   κ̇ₜ₊Δₜ⁽⁰⁾ = Δλ⁽⁰⁾/Δt
   σₜ₊Δₜ⁽⁰⁾ = σₜ + Dᵉ[Δε - Δλ⁽⁰⁾ ∂f/∂σ]
   f⁽⁰⁾ = f(σₜ₊Δₜ⁽⁰⁾, κₜ + Δλ⁽⁰⁾, κ̇ₜ₊Δₜ⁽⁰⁾)

4. Local iteration loop:

   a) Compute pseudo-elastic stiffness:
      H⁽ⁱ⁾ = [Dᵉ⁻¹ + Δλ⁽ⁱ⁾ ∂²f/∂σ²]⁻¹
   
   b) Compute denominator:
      b = [∂f/∂σ]ᵀ H [∂f/∂σ + Δλ⁽ⁱ⁾ ∂²f/∂σ∂κ + (Δλ⁽ⁱ⁾/Δt) ∂²f/∂σ∂κ̇]
          - ∂f/∂κ - (1/Δt) ∂f/∂κ̇
   
   c) Update plastic multiplier:
      Δλ⁽ⁱ⁺¹⁾ = Δλ⁽ⁱ⁾ + f⁽ⁱ⁾/b
   
   d) Update stress:
      σₜ₊Δₜ⁽ⁱ⁺¹⁾ = σₜ + Dᵉ[Δε - Δλ⁽ⁱ⁺¹⁾ ∂f/∂σ]
   
   e) Update yield function:
      f⁽ⁱ⁺¹⁾ = f(σₜ₊Δₜ⁽ⁱ⁺¹⁾, κₜ + Δλ⁽ⁱ⁺¹⁾, κ̇ₜ₊Δₜ⁽ⁱ⁺¹⁾)
      
      where κ̇ₜ₊Δₜ⁽ⁱ⁺¹⁾ = Δλ⁽ⁱ⁺¹⁾/Δt
   
   f) Check convergence:
      IF |f⁽ⁱ⁺¹⁾| < tolerance:
         EXIT loop
      ELSE:
         GO TO step 4a

Notes:
- Rate-dependent yield surface: f = f(σ, κ, κ̇)
- Enforces consistency condition: f = 0
- Can handle both H-type (strain softening) and S-type (strain-rate softening) instabilities
- ∂f/∂κ̇ ≠ 0 for rate-dependent yield surface
```

## Common Notation

- σ = Cauchy stress tensor (Voigt notation: [σ₁₁, σ₂₂, σ₃₃, σ₁₂, σ₂₃, σ₁₃])
- ε = total strain tensor
- εᵛᵖ = viscoplastic strain tensor
- Dᵉ = elastic stiffness matrix
- f = yield function
- κ = internal variable (often equivalent plastic strain)
- Δλ = plastic multiplier increment
- γ = fluidity parameter (Perzyna)
- φ = overstress function
- τ = relaxation time (Duvaut-Lions)
- n = ∂f/∂σ = flow direction
- h = -∂f/∂κ = hardening modulus
- Δt = time increment
- θ = integration parameter (0 = explicit, 1 = implicit)

## Yield Function Examples

### Von Mises (J₂ plasticity):
```
f = √(3J₂) - σᵧ(κ)

where J₂ = (1/2) s:s (second invariant of deviatoric stress)
      s = σ - (1/3)tr(σ)I (deviatoric stress)
```

### Von Mises with linear hardening:
```
f = √(3J₂) - (σᵧ + Hκ)
```

### Von Mises with rate dependence (consistency model):
```
f = √(3J₂) - (σᵧ + Hκ + mκ̇)

where m = viscosity parameter
```

## Overstress Function Examples

### Linear viscosity (N=1):
```
φ = γ(f/σ₀)  for f > 0
φ = 0        for f ≤ 0
```

### Power law viscosity:
```
φ = γ(f/σ₀)ᴺ  for f > 0
φ = 0         for f ≤ 0
```

### Exponential viscosity:
```
φ = γ exp(f/σ₀)  for f > 0
φ = 0            for f ≤ 0
```
