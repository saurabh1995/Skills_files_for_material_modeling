'''
Changes you need tomake for testing each file:

1.Import the correct material model at the start of the file.
2.Change material parameters 
3.Initial state, this will be same as material model's state variables.
4.Change n_steps, dt, fy_limit 
5.I have different set of eps_i values 



'''


from Lemaiture_Chaboche_model import PlaneStressLemaitreDamage, LemaitreChabocheParams, LemaitreChabocheMaterial
import jax.numpy as jnp
from dolfinx_materials.jax_materials import (
    vonMisesIsotropicHardening,
    LinearElasticIsotropic,
)

params = LemaitreChabocheParams(
    E=113066, nu=0.3,
    k=100, K_visc=11.45, n_visc=8.15,
    b=0.0, R1=0.0,
    a=98939.3, c=1533.41,
    Dc=0.0, eps_D=0.0, eps_R=0.1629
)
#mat = PlaneStressLemaitreDamage(params)
elasticity = LinearElasticIsotropic(E=params.E, nu=params.nu)
mat = LemaitreChabocheMaterial(elastic_model=elasticity, params=params)
#mat = LemaitreChabocheMaterial(params)


# Build initial state dict with zeros, correct shapes:
initial_state = {
    "Strain": jnp.zeros(6),
    "Stress": jnp.zeros(6),
    "p": jnp.zeros(1),
    "D": jnp.zeros(1),
    "R": jnp.zeros(1),
    "eps_p": jnp.zeros(6),
    "eps_e": jnp.zeros(6),
    "X": jnp.zeros(6),
    "fy": jnp.zeros(1),
    "p_dot": jnp.zeros(1),
    "dp": jnp.zeros(1),
    "eps_I_dot": jnp.zeros(6),
}

## Multiple strain sequence
# --- Plastic yield start test ---
import matplotlib.pyplot as plt
import jax.numpy as jnp
from Lemaiture_Chaboche_model import PlaneStressLemaitreDamage, LemaitreChabocheParams

# ... your params, mat, initial_state as before ...


def test_yield_start(initial_state):
    # Strain ramp in e11 from 0 to 0.01 (1% strain) in 200 steps
    state = initial_state
    n_steps = 500
    
    ## 3D cube values
    eps_11_i = 4.97872340e-06
    eps_22_i = -9.33510638e-07
    eps_33_i = -9.33510638e-07
    eps_12_i = 1.13686838e-22
    eps_23_i = 2.27373675e-22
    eps_13_i = 4.60044261e-22
    '''
    ## Saurabh's values 
    eps_11_i = -4.075468462975723891e-12
    eps_22_i = 4.256575537342220905e-11
    eps_33_i = -1.649583724733479877e-11
    eps_23_i = -1.606784058309870244e-10
    eps_13_i = 2.696120304608055775e-12
    eps_12_i = 2.229080511025814095e-10

    
    ## Saurabh's values at 10000
    eps_11_i = -8.15093693e-07 
    eps_22_i =  8.51315107e-06
    eps_33_i = -3.29916745e-06
    eps_23_i = -3.21356812e-05
    eps_13_i = 5.39224061e-07
    eps_12_i = 4.45816102e-05 

    ## Saurabh's values at 7000

    eps_11_i = -5.70565585e-07  
    eps_22_i =  5.95920575e-06
    eps_33_i = -2.30941721e-06
    eps_23_i = -2.24949768e-05
    eps_13_i = 3.77456843e-07
    eps_12_i = 3.12071272e-05 

    ## Saurabh's values at 5000

    eps_11_i = -4.07546846e-07  
    eps_22_i =  4.25657554e-06
    eps_33_i = -1.64958372e-06
    eps_23_i = -1.60678406e-05
    eps_13_i = 2.69612030e-07
    eps_12_i = 2.22908051e-05'''
    

    dt = 1e-7  

    fy_limit = 30.0  # stop test when overstress gets too large (for stability)
    #sig_max = 1.08658439e+04

    print("Plastic yield start test:")
    print(" step |   eps11    |   sig11    |    fy      |    p       |   dp      |  p_dot")

    # --- Arrays to store history ---
    steps = []
    eps11_hist = []
    sig11_hist = []
    eps11_tot_hist = []
    eps11_e_hist = []
    eps11_p_hist = []
    sig12_hist = []
    eps12_hist = [] 


    yield_found = False

    for i in range(0, n_steps + 1):
        
        eps11 = eps_11_i * i + eps_11_i
        eps_22 = eps_22_i * i   + eps_22_i
        eps_33 = eps_33_i * i  + eps_33_i
        eps_12 = eps_12_i * i + eps_12_i
        eps_23 = eps_23_i * i + eps_23_i
        eps_13 = eps_13_i * i + eps_13_i
        eps = jnp.array([eps11, eps_22, eps_33, eps_12, eps_13, eps_23])



        sig, state = mat.constitutive_update(eps, state, dt)



        # Guard against NaNs/Infs
        if not jnp.isfinite(state["fy"]).all():
            print(f"\nStopping: non-finite fy at step {i}")
            break

        fy    = float(state["fy"][0])
        p     = float(state["p"][0])
        dp    = float(state["dp"][0])
        p_dot = float(state["p_dot"][0])
        #sig11 = float(jnp.ravel(sig)[0])
        sig11 = float(state["Stress"][0])

        # total strain, elastic, plastic (11 component)
        eps_tot_11 = float(state["Strain"][0])
        eps_e_11   = float(state["eps_e"][0])
        eps_p_11   = float(state["eps_p"][0])

        sig12 = float(state["Stress"][5])  # σ12 (Voigt index 5)


        
        

        print(
            f"{i:4d} | {eps11:9.6e} | {sig11:9.6e} | {fy: .12e} | "
            f"{p: .12e} | {dp: .12e} | {p_dot: .12e} | {eps_e_11: .12e} | {eps_p_11: .12e}"
        )

        # store history
        steps.append(i)
        eps11_hist.append(eps11)
        sig11_hist.append(sig11)
        eps11_tot_hist.append(eps_tot_11)
        eps11_e_hist.append(eps_e_11)
        eps11_p_hist.append(eps_p_11)
        sig12_hist.append(sig12)

          # add near other histories

# inside loop
        eps12 = float(eps[5])          # imposed ε12 (Voigt index 5)
        eps12_hist.append(eps12)



        # Yield detection only for info
        if fy > 0.0 and not yield_found:
            print("\n*** Plastic yield detected ***")
            print(f"  -> first plastic step = {i}")
            print(f"  -> eps11  = {eps11}")
            print(f"  -> sig11  = {sig11} MPa")
            print(f"  -> fy     = {fy}")
            print(f"  -> p      = {p}")
            print(f"  -> dp     = {dp}")
            print(f"  -> p_dot  = {p_dot}")
            yield_found = True

        # Stop test once we're far into post-yield (avoid blow-up)
        if fy >= fy_limit:
            print(f"\nStopping: sig11 exceeded limit ({fy_limit}) at step {i}")
            break

    # ------------------------------
    # Plotting
    # ------------------------------
    # plotting
    plt.figure(figsize=(6, 4))
    plt.plot(eps12_hist, sig12_hist, marker="o", linewidth=1)
    plt.xlabel(r"$\varepsilon_{12}$")
    plt.ylabel(r"$\sigma_{12}$")
    plt.title("Shear Stress–Strain (Material Point)")
    plt.grid(True)


    # (1) Stress vs total strain (11-component)
    plt.figure(figsize=(6, 4))
    plt.plot(eps11_hist, sig11_hist, marker="o", linewidth=1)
    plt.xlabel(r"$\varepsilon_{11}$")
    plt.ylabel(r"$\sigma_{11}$")  # units as per your convention
    plt.title("Uniaxial Stress–Strain (Material Point)")
    plt.grid(True)

    # (2) eps, eps_e, eps_p vs step index
    plt.figure(figsize=(6, 4))
    plt.plot(steps, eps11_tot_hist, label=r"$\varepsilon_{11}$ (total)")
    plt.plot(steps, eps11_e_hist,   label=r"$\varepsilon_{11}^e$")
    plt.plot(steps, eps11_p_hist,   label=r"$\varepsilon_{11}^p$")
    plt.xlabel("Step")
    plt.ylabel("Strain")
    plt.title(r"Strain decomposition vs step")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    #test_elastic()
    test_yield_start(initial_state)