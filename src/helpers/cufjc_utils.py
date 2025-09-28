import numpy as np
from src.helpers.inv_langevin_func_utils import s_cn_inv_langevin_fjc_func

def arccos_arg_domain_cnstrnt_func(arccos_arg: float):
    if arccos_arg >= 1. - 1.e-14: arccos_arg = 1. - 1.e-14
    elif arccos_arg < -1. + 1.e-14: arccos_arg = -1. + 1.e-14
    return arccos_arg

def gamma_n_crit_cufjc_func(kappa_n: float, zeta_n_char: float) -> float:
    return 1. + np.sqrt(zeta_n_char/kappa_n)

def gamma_crit_cufjc_func(kappa_n: float, zeta_n_char: float) -> float:
    return (
        gamma_n_crit_cufjc_func(kappa_n, zeta_n_char)
        - np.sqrt(1./(zeta_n_char*kappa_n))
    )

def gamma_pade_to_bergstrom_crit_cufjc_func(kappa_n: float) -> float:
    return 1. / kappa_n**0.818706900266885 + 0.61757545643322586

def gamma_n_cufjc_func(
        gamma: np.ndarray | float,
        kappa_n: float,
        zeta_n_char: float,
        gamma_pade_to_bergstrom_crit: float,
        gamma_crit: float) -> np.ndarray | float:
    gamma = np.asarray([gamma])
    gamma_n = np.empty_like(gamma)

    for indx in np.ndindex(np.shape(gamma)):
        gamma_val = gamma[indx]
    
        # analytical solution (Pade approximant)
        if np.isclose(gamma_val, 0.0): gamma_n[indx] = 1.
        
        # Pade approximant
        elif gamma_val < gamma_pade_to_bergstrom_crit:
            alpha_tilde = 1.
            
            trm_i = -3. * (kappa_n+1.)
            trm_ii = -(2.*kappa_n+3.)
            beta_tilde_nmrtr = trm_i + gamma_val * trm_ii
            beta_tilde_dnmntr = kappa_n + 1.
            beta_tilde = beta_tilde_nmrtr / beta_tilde_dnmntr
            
            trm_i = 2. * kappa_n
            trm_ii = 4. * kappa_n + 6.
            trm_iii = kappa_n + 3.
            gamma_tilde_nmrtr = trm_i + gamma_val * (trm_ii+gamma_val*trm_iii)
            gamma_tilde_dnmntr = kappa_n + 1.
            gamma_tilde = gamma_tilde_nmrtr / gamma_tilde_dnmntr

            trm_i = 2.
            trm_ii = 2. * kappa_n
            trm_iii = kappa_n + 3.
            delta_tilde_nmrtr = (
                trm_i - gamma_val * (trm_ii+gamma_val*(trm_iii+gamma_val))
            )
            delta_tilde_dnmntr = kappa_n + 1.
            delta_tilde = delta_tilde_nmrtr / delta_tilde_dnmntr

            pi_tilde_nmrtr = 3. * alpha_tilde * gamma_tilde - beta_tilde**2
            pi_tilde_dnmntr = 3. * alpha_tilde**2
            pi_tilde = pi_tilde_nmrtr / pi_tilde_dnmntr

            rho_tilde_nmrtr = (
                2. * beta_tilde**3 - 9. * alpha_tilde * beta_tilde * gamma_tilde 
                + 27. * alpha_tilde**2 * delta_tilde
            )
            rho_tilde_dnmntr = 27. * alpha_tilde**3
            rho_tilde = rho_tilde_nmrtr / rho_tilde_dnmntr
            
            arccos_arg = 3. * rho_tilde / (2.*pi_tilde) * np.sqrt(-3./pi_tilde)
            arccos_arg = arccos_arg_domain_cnstrnt_func(arccos_arg)
            cos_arg = 1. / 3. * np.arccos(arccos_arg) - 2. * np.pi / 3.
            gamma_n[indx] = (
                2. * np.sqrt(-pi_tilde/3.) * np.cos(cos_arg)
                - beta_tilde / (3.*alpha_tilde)
            )
        
        # Bergstrom approximant
        elif gamma_val <= gamma_crit:
            sqrt_arg = gamma_val**2 - 2. * gamma_val + 1. + 4. / kappa_n
            gamma_n[indx] = (gamma_val+1.+np.sqrt(sqrt_arg)) / 2.
        
        # Bergstrom approximant
        else:
            alpha_tilde = 1.
            beta_tilde = -3.
            gamma_tilde = 3. - zeta_n_char**2 / kappa_n
            delta_tilde = zeta_n_char**2 / kappa_n * gamma_val - 1.

            pi_tilde_nmrtr = 3. * alpha_tilde * gamma_tilde - beta_tilde**2
            pi_tilde_dnmntr = 3. * alpha_tilde**2
            pi_tilde = pi_tilde_nmrtr / pi_tilde_dnmntr

            rho_tilde_nmrtr = (
                2. * beta_tilde**3 - 9. * alpha_tilde * beta_tilde * gamma_tilde
                + 27. * alpha_tilde**2 * delta_tilde
            )
            rho_tilde_dnmntr = 27. * alpha_tilde**3
            rho_tilde = rho_tilde_nmrtr / rho_tilde_dnmntr
            
            arccos_arg = 3. * rho_tilde / (2.*pi_tilde) * np.sqrt(-3./pi_tilde)
            arccos_arg = arccos_arg_domain_cnstrnt_func(arccos_arg)
            cos_arg = 1. / 3. * np.arccos(arccos_arg) - 2. * np.pi / 3.
            gamma_n[indx] = (
                2. * np.sqrt(-pi_tilde/3.) * np.cos(cos_arg) 
                - beta_tilde / (3.*alpha_tilde)
            )
    
    if np.shape(gamma_n) == (1,): return gamma_n[0]
    else: return gamma_n

def u_n_subcrit_cufjc_func(
        gamma_n: np.ndarray | float,
        kappa_n: float) -> np.ndarray | float:
    return 0.5 * kappa_n * (gamma_n-1.)**2

def u_n_supercrit_cufjc_func(
        gamma_n: np.ndarray | float,
        kappa_n: float,
        zeta_n_char: float) -> np.ndarray | float:
    return zeta_n_char - zeta_n_char**2 / (2.*kappa_n*(gamma_n-1.)**2)

def u_n_cufjc_func(
        gamma_n: np.ndarray | float,
        kappa_n: float,
        zeta_n_char: float,
        gamma_n_crit: float) -> np.ndarray | float:
    gamma_n = np.asarray([gamma_n])
    u_n = np.empty_like(gamma_n)

    for indx in np.ndindex(np.shape(gamma_n)):
        gamma_n_val = gamma_n[indx]
        if gamma_n_val <= gamma_n_crit:
            u_n[indx] = u_n_subcrit_cufjc_func(gamma_n_val, kappa_n)
        else:
            u_n[indx] = u_n_supercrit_cufjc_func(
                gamma_n_val, kappa_n, zeta_n_char)
    
    if np.shape(u_n) == (1,): return u_n[0]
    else: return u_n

def u_c_cufjc_func(
        gamma_n: np.ndarray | float,
        n: np.ndarray | float | int,
        kappa_n: float,
        zeta_n_char: float,
        gamma_n_crit: float) -> np.ndarray | float:
    return n * u_n_cufjc_func(gamma_n, kappa_n, zeta_n_char, gamma_n_crit)

def s_cn_cufjc_func(gamma_cufjc_n: np.ndarray | float) -> np.ndarray | float:
    """Nondimensional chain-level entropic free energy contribution
    per segment as calculated by the Jedynak R[9,2] inverse Langevin
    approximate.
        
    This function computes the nondimensional chain-level entropic
    free energy contribution per segment as calculated by the
    Jedynak R[9,2] inverse Langevin approximate as a function of the
    result of the equilibrium chain stretch minus the segment
    stretch plus one.
    
    """
    return s_cn_inv_langevin_fjc_func(gamma_cufjc_n)

def s_c_cufjc_func(
        gamma_cufjc_n: np.ndarray | float,
        n: np.ndarray | float | int) -> np.ndarray | float:
    """Gaussian end-to-end distance polymer chain conformation
    probability distribution.

    This function calculates the Gaussian end-to-end distance polymer
    chain conformation probability for a chain with a given number of
    segments and a given end-to-end distance.

    Args:
        r (np.ndarray | float | int): End-to-end chain distance.
        n (np.ndarray | float | int): Number of segments in the chain.
        b (float): Chain segment and/or cross-linker diameter.
    
    Note: If nu is an np.ndarray, then r must be a float or int.
    Likewise, if r is an np.ndarray, then nu must be a float or int.

    May need a s_cn_args: tuple[float] later
    
    Returns:
        np.ndarray | float: Gaussian end-to-end distance polymer chain
        conformation probability (distribution).
    
    """
    return n * s_cn_cufjc_func(gamma_cufjc_n)

def w_c_cufjc_func(
        gamma: np.ndarray | float,
        n: np.ndarray | float | int,
        kappa_n: float,
        zeta_n_char: float) -> np.ndarray | float:
    gamma_n_crit = gamma_n_crit_cufjc_func(kappa_n, zeta_n_char)
    gamma_crit = gamma_crit_cufjc_func(kappa_n, zeta_n_char)
    gamma_pade_to_bergstrom_crit = gamma_pade_to_bergstrom_crit_cufjc_func(
        kappa_n)
    gamma_n = gamma_n_cufjc_func(
        gamma, kappa_n, zeta_n_char, gamma_pade_to_bergstrom_crit, gamma_crit)
    gamma_cufjc_n = gamma - gamma_n + 1.

    return (
        u_c_cufjc_func(gamma_n, n, kappa_n, zeta_n_char, gamma_n_crit)
        + s_c_cufjc_func(gamma_cufjc_n, n)
    )