import numpy as np
import numpy.typing as npt
from src.helpers.utils import arccos_arg_cnstrnt_func
from src.helpers.inv_langevin import (
    s_cn_inv_langevin_fjc_func,
    inv_langevin_func,
    inv_langevin_prime_func
)
from src.helpers.boltzmann import p_boltzmann_func

def gamma_n_crit_func(kappa_n: float, zeta_n_char: float) -> float:
    """Critical segment stretch.

    This function returns the critical segment stretch.

    Args:
        kappa_n (float): Nondimensional segment stiffness.
        zeta_n_char (float): Nondimensional characteristic segment potential energy scale.
    
    Returns:
        float: Critical segment stretch.
    
    """
    return 1. + np.sqrt(zeta_n_char/kappa_n)

def gamma_crit_func(kappa_n: float, zeta_n_char: float) -> float:
    """Critical absolute/equilibrium chain stretch.

    This function returns the critical absolute/equilibrium chain
    stretch.

    Args:
        kappa_n (float): Nondimensional segment stiffness.
        zeta_n_char (float): Nondimensional characteristic segment potential energy scale.
    
    Returns:
        float: Critical absolute/equilibrium chain stretch.
    
    """
    return (
        gamma_n_crit_func(kappa_n, zeta_n_char)
        - np.sqrt(1./(kappa_n*zeta_n_char))
    )

def xi_c_crit_func(kappa_n: float, zeta_n_char: float) -> float:
    """Nondimensional critical chain force.

    This function returns the nondimensional critical chain force.

    Args:
        kappa_n (float): Nondimensional segment stiffness.
        zeta_n_char (float): Nondimensional characteristic segment potential energy scale.
    
    Returns:
        float: Nondimensional critical chain force.
    
    """
    return np.sqrt(kappa_n*zeta_n_char)

def gamma_pade_to_bergstrom_crit_func(kappa_n: float) -> float:
    """Pade-to-Bergstrom critical absolute/equilibrium chain stretch.

    This function returns the Pade-to-Bergstrom critical
    absolute/equilibrium chain stretch as determined via a scipy
    optimize curve_fit analysis.

    Args:
        kappa_n (float): Nondimensional segment stiffness.
    
    Returns:
        float: Pade-to-Bergstrom critical absolute/equilibrium chain
        stretch.
    
    """
    return 1. / kappa_n**0.818706900266885 + 0.61757545643322586

def gamma_n_pade_to_bergstrom_crit_func(
        kappa_n: float,
        zeta_n_char: float,
        gamma_pade_to_bergstrom_crit: float,
        gamma_crit: float) -> float:
    """Pade-to-Bergstrom critical segment stretch.

    This function returns the Pade-to-Bergstrom critical segment
    stretch.

    Args:
        kappa_n (float): Nondimensional segment stiffness.
        zeta_n_char (float): Nondimensional characteristic segment potential energy scale.
        gamma_pade_to_bergstrom_crit (float): Pade-to-Bergstrom critical absolute/equilibrium chain stretch.
        gamma_crit (float): Critical absolute/equilibrium chain stretch.
    
    Returns:
        float: Pade-to-Bergstrom critical segment stretch.
    
    """
    return (
        gamma_n_func(
            gamma_pade_to_bergstrom_crit, kappa_n, zeta_n_char,
            gamma_pade_to_bergstrom_crit, gamma_crit)
    )

def subcrit_gamma_n_pade_approx_func(
        gamma: npt.ArrayLike,
        kappa_n: float) -> npt.ArrayLike:
    """Sub-critical chain state segment stretch as derived via the Pade
    approximant for the inverse Langevin function.
    
    This function computes the sub-critical chain state segment stretch
    (as derived via the Pade approximant for the inverse Langevin
    function).

    Args:
        gamma (npt.ArrayLike): Absolute/Equilibrium chain stretch.
        kappa_n (float): Nondimensional segment stiffness.
    
    Returns:
        npt.ArrayLike: Sub-critical chain state segment stretch as
        derived via the Pade approximant for the inverse Langevin
        function.
    
    """
    gamma_isscalar = np.isscalar(gamma)
    if gamma_isscalar: gamma = np.asarray([gamma])
    gamma_n = np.empty_like(gamma)

    for indx in np.ndindex(np.shape(gamma)):
        gamma_val = gamma[indx]

        # Analytical solution (Pade approximant)
        if np.isclose(gamma_val, 0.): gamma_n[indx] = 1.
    
        # Pade approximant
        else:
            alpha_tilde = 1.
            
            trm_i = -3. * (kappa_n+1.)
            trm_ii = -(2.*kappa_n+3.)
            beta_tilde_nmrtr = trm_i + gamma_val * trm_ii
            beta_tilde_dnmntr = kappa_n + 1.
            beta_tilde = beta_tilde_nmrtr / beta_tilde_dnmntr
            
            trm_i = 2. * kappa_n
            trm_ii = 4. * kappa_n + 6.
            trm_iii = kappa_n + 3.
            gamma_tilde_nmrtr = (
                trm_i + gamma_val * (trm_ii+gamma_val*trm_iii)
            )
            gamma_tilde_dnmntr = kappa_n + 1.
            gamma_tilde = gamma_tilde_nmrtr / gamma_tilde_dnmntr

            trm_i = 2.
            trm_ii = 2. * kappa_n
            trm_iii = kappa_n + 3.
            delta_tilde_nmrtr = (
                trm_i
                - gamma_val
                * (trm_ii+gamma_val*(trm_iii+gamma_val))
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
            arccos_arg = arccos_arg_cnstrnt_func(arccos_arg)
            cos_arg = 1. / 3. * np.arccos(arccos_arg) - 2. * np.pi / 3.
            gamma_n[indx] = (
                2. * np.sqrt(-pi_tilde/3.) * np.cos(cos_arg)
                - beta_tilde / (3.*alpha_tilde)
            )
    
    return gamma_n.item() if gamma_isscalar else gamma_n

def subcrit_gamma_n_bergstrom_approx_func(
        gamma: npt.ArrayLike,
        kappa_n: float) -> npt.ArrayLike:
    """Sub-critical chain state segment stretch as derived via the
    Bergstrom approximant for the inverse Langevin function.
    
    This function computes the sub-critical chain state segment stretch
    (as derived via the Bergstrom approximant for the inverse Langevin
    function).

    Args:
        gamma (npt.ArrayLike): Absolute/Equilibrium chain stretch.
        kappa_n (float): Nondimensional segment stiffness.
    
    Returns:
        npt.ArrayLike: Sub-critical chain state segment stretch as
        derived via the Bergstrom approximant for the inverse Langevin
        function.
    
    """
    sqrt_arg = gamma**2 - 2. * gamma + 1. + 4. / kappa_n
    return (gamma+1.+np.sqrt(sqrt_arg)) / 2.

def supercrit_gamma_n_bergstrom_approx_func(
        gamma: npt.ArrayLike,
        kappa_n: float,
        zeta_n_char: float) -> npt.ArrayLike:
    """Super-critical chain state segment stretch as derived via the
    Bergstrom approximant for the inverse Langevin function.
    
    This function computes the super-critical chain state segment
    stretch (as derived via the Bergstrom approximant for the inverse
    Langevin function).

    Args:
        gamma (npt.ArrayLike): Absolute/Equilibrium chain stretch.
        kappa_n (float): Nondimensional segment stiffness.
        zeta_n_char (float): Nondimensional characteristic segment potential energy scale.
    
    Returns:
        npt.ArrayLike: Super-critical chain state segment stretch as
        derived via the Bergstrom approximant for the inverse Langevin
        function.
    
    """
    gamma_isscalar = np.isscalar(gamma)
    if gamma_isscalar: gamma = np.asarray([gamma])
    gamma_n = np.empty_like(gamma)

    for indx in np.ndindex(np.shape(gamma)):
        gamma_val = gamma[indx]
        
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
        arccos_arg = arccos_arg_cnstrnt_func(arccos_arg)
        cos_arg = 1. / 3. * np.arccos(arccos_arg) - 2. * np.pi / 3.
        gamma_n[indx] = (
            2. * np.sqrt(-pi_tilde/3.) * np.cos(cos_arg)
            - beta_tilde / (3.*alpha_tilde)
        )
    
    return gamma_n.item() if gamma_isscalar else gamma_n

def gamma_n_func(
        gamma: npt.ArrayLike,
        kappa_n: float,
        zeta_n_char: float,
        gamma_pade_to_bergstrom_crit: float,
        gamma_crit: float) -> npt.ArrayLike:
    """Segment stretch.

    This function returns the segment stretch.

    Args:
        gamma (npt.ArrayLike): Absolute/Equilibrium chain stretch.
        kappa_n (float): Nondimensional segment stiffness.
        zeta_n_char (float): Nondimensional characteristic segment potential energy scale.
        gamma_pade_to_bergstrom_crit (float): Pade-to-Bergstrom critical absolute/equilibrium chain stretch.
        gamma_crit (float): Critical absolute/equilibrium chain stretch.
    
    Returns:
        npt.ArrayLike: Segment stretch.
    
    """
    gamma_isscalar = np.isscalar(gamma)
    if gamma_isscalar: gamma = np.asarray([gamma])
    gamma_n = np.empty_like(gamma)

    for indx in np.ndindex(np.shape(gamma)):
        gamma_val = gamma[indx]
    
        # Sub-critical chain state segment stretch as derived via the
        # Pade approximant for the inverse Langevin function
        if gamma_val < gamma_pade_to_bergstrom_crit:
            gamma_n[indx] = subcrit_gamma_n_pade_approx_func(
                gamma_val, kappa_n)
        
        # Sub-critical chain state segment stretch as derived via the
        # Bergstrom approximant for the inverse Langevin function
        elif gamma_val <= gamma_crit:
            gamma_n[indx] = subcrit_gamma_n_bergstrom_approx_func(
                gamma_val, kappa_n)
        
        # Super-critical chain state segment stretch as derived via the
        # Bergstrom approximant for the inverse Langevin function
        else:
            gamma_n[indx] = supercrit_gamma_n_bergstrom_approx_func(
                gamma_val, kappa_n, zeta_n_char)    
    
    return gamma_n.item() if gamma_isscalar else gamma_n

def subcrit_gamma_pade_approx_func(
        gamma_n: npt.ArrayLike,
        kappa_n: float) -> npt.ArrayLike:
    """Sub-critical chain state absolute/equilibrium chain stretch as
    derived via the Pade approximant for the inverse Langevin function.
    
    This function computes the sub-critical chain state
    absolute/equilibrium chain stretch (as derived via the Pade
    approximant for the inverse Langevin function).

    Args:
        gamma_n (npt.ArrayLike): Segment stretch.
        kappa_n (float): Nondimensional segment stiffness.
    
    Returns:
        npt.ArrayLike: Sub-critical chain state absolute/equilibrium
        chain stretch as derived via the Pade approximant for the
        inverse Langevin function.
    
    """
    gamma_n_isscalar = np.isscalar(gamma_n)
    if gamma_n_isscalar: gamma_n = np.asarray([gamma_n])
    gamma = np.empty_like(gamma_n)

    for indx in np.ndindex(np.shape(gamma_n)):
        gamma_n_val = gamma_n[indx]
    
        # Analytical solution (Pade approximant)
        if np.isclose(gamma_n_val, 1.): gamma[indx] = 0.
        
        # Pade approximant
        else:
            alpha_tilde = 1.
            
            trm_i = kappa_n + 3.
            trm_ii = 1.
            beta_tilde = trm_i * (trm_ii-gamma_n_val)

            trm_i = 2. * kappa_n + 3.
            trm_ii = 2.
            trm_iii = 2. * kappa_n
            gamma_tilde = trm_i * (gamma_n_val**2-trm_ii*gamma_n_val) + trm_iii
            
            trm_i = kappa_n + 1.
            trm_ii = 3.
            trm_iii = 2.
            trm_iv = kappa_n
            trm_v = 1.
            delta_tilde = (
                trm_i * (trm_ii*gamma_n_val**2-gamma_n_val**3)
                - trm_iii * (trm_iv*gamma_n_val+trm_v)
            )
            
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
            arccos_arg = arccos_arg_cnstrnt_func(arccos_arg)
            cos_arg = 1. / 3. * np.arccos(arccos_arg) - 2. * np.pi / 3.
            gamma[indx] = (
                2. * np.sqrt(-pi_tilde/3.) * np.cos(cos_arg)
                - beta_tilde / (3.*alpha_tilde)
            )
    
    return gamma.item() if gamma_n_isscalar else gamma

def subcrit_gamma_bergstrom_approx_func(
        gamma_n: npt.ArrayLike,
        kappa_n: float) -> npt.ArrayLike:
    """Sub-critical chain state absolute/equilibrium chain stretch as
    derived via the Bergstrom approximant for the inverse Langevin
    function.
    
    This function computes the sub-critical chain state
    absolute/equilibrium chain stretch (as derived via the Bergstrom
    approximant for the inverse Langevin function).

    Args:
        gamma_n (npt.ArrayLike): Segment stretch.
        kappa_n (float): Nondimensional segment stiffness.
    
    Returns:
        npt.ArrayLike: Sub-critical chain state absolute/equilibrium
        chain stretch as derived via the Bergstrom approximant for the
        inverse Langevin function.
    
    """
    return gamma_n - 1. / (kappa_n*(gamma_n-1.))

def supercrit_gamma_bergstrom_approx_func(
        gamma_n: npt.ArrayLike,
        kappa_n: float,
        zeta_n_char: float) -> npt.ArrayLike:
    """Super-critical chain state absolute/equilibrium chain stretch as
    derived via the Bergstrom approximant for the inverse Langevin
    function.
    
    This function computes the super-critical chain state
    absolute/equilibrium chain stretch (as derived via the Bergstrom
    approximant for the inverse Langevin function).

    Args:
        gamma_n (npt.ArrayLike): Segment stretch.
        kappa_n (float): Nondimensional segment stiffness.
        zeta_n_char (float): Nondimensional characteristic segment potential energy scale.
    
    Returns:
        npt.ArrayLike: Super-critical chain state absolute/equilibrium
        chain stretch as derived via the Bergstrom approximant for the
        inverse Langevin function.
    
    """
    return gamma_n - kappa_n / zeta_n_char**2 * (gamma_n-1.)**3

def gamma_func(
        gamma_n: npt.ArrayLike,
        kappa_n: float,
        zeta_n_char: float,
        gamma_n_pade_to_bergstrom_crit: float,
        gamma_n_crit: float) -> npt.ArrayLike:
    """Absolute/Equilibrium chain stretch.

    This function returns the absolute/equilibrium chain stretch.

    Args:
        gamma_n (npt.ArrayLike): Segment stretch.
        kappa_n (float): Nondimensional segment stiffness.
        zeta_n_char (float): Nondimensional characteristic segment potential energy scale.
        gamma_n_pade_to_bergstrom_crit (float): Pade-to-Bergstrom critical segment stretch.
        gamma_n_crit (float): Critical segment stretch.
    
    Returns:
        npt.ArrayLike: Absolute/Equilibrium chain stretch.
    
    """
    gamma_n_isscalar = np.isscalar(gamma_n)
    if gamma_n_isscalar: gamma_n = np.asarray([gamma_n])
    gamma = np.empty_like(gamma_n)

    for indx in np.ndindex(np.shape(gamma_n)):
        gamma_n_val = gamma_n[indx]
    
        # Sub-critical chain state absolute/equilibrium chain stretch as
        # derived via the Pade approximant for the inverse Langevin
        # function
        if gamma_n_val < gamma_n_pade_to_bergstrom_crit:
            gamma[indx] = subcrit_gamma_pade_approx_func(
                gamma_n_val, kappa_n)
        
        # Sub-critical chain state absolute/equilibrium chain stretch as
        # derived via the Bergstrom approximant for the inverse Langevin
        # function
        elif gamma_n_val <= gamma_n_crit:
            gamma[indx] = subcrit_gamma_bergstrom_approx_func(
                gamma_n_val, kappa_n)
        
        # Super-critical chain state absolute/equilibrium chain stretch 
        # as derived via the Bergstrom approximant for the inverse
        # Langevin function
        else:
            gamma[indx] = supercrit_gamma_bergstrom_approx_func(
                gamma_n_val, kappa_n, zeta_n_char)
    
    return gamma.item() if gamma_n_isscalar else gamma

def u_n_har_func(
        gamma_n: npt.ArrayLike,
        kappa_n: float,
        zeta_n_char: float) -> npt.ArrayLike:
    """Nondimensional harmonic segment potential energy.
        
    This function returns the nondimensional harmonic segment potential
    energy.
    
    Args:
        gamma_n (npt.ArrayLike): Segment stretch.
        kappa_n (float): Nondimensional segment stiffness.
        zeta_n_char (float): Nondimensional characteristic segment potential energy scale.
    
    Returns:
        npt.ArrayLike: Nondimensional harmonic segment potential energy.
    
    """
    return 0.5 * kappa_n * (gamma_n-1.)**2 - zeta_n_char

def u_n_subcrit_func(
        gamma_n: npt.ArrayLike,
        kappa_n: float,
        zeta_n_char: float) -> npt.ArrayLike:
    """Nondimensional sub-critical chain state segment potential energy.
        
    This function returns the nondimensional sub-critical chain state
    segment potential energy.
    
    Args:
        gamma_n (npt.ArrayLike): Segment stretch.
        kappa_n (float): Nondimensional segment stiffness.
        zeta_n_char (float): Nondimensional characteristic segment potential energy scale.
    
    Returns:
        npt.ArrayLike: Nondimensional sub-critical chain state segment
        potential energy.
    
    """
    return u_n_har_func(gamma_n, kappa_n, zeta_n_char)

def u_n_supercrit_func(
        gamma_n: npt.ArrayLike,
        kappa_n: float,
        zeta_n_char: float) -> npt.ArrayLike:
    """Nondimensional super-critical chain state segment potential
    energy.
        
    This function returns the nondimensional super-critical chain state
    segment potential energy.
    
    Args:
        gamma_n (npt.ArrayLike): Segment stretch.
        kappa_n (float): Nondimensional segment stiffness.
        zeta_n_char (float): Nondimensional characteristic segment potential energy scale.
    
    Returns:
        npt.ArrayLike: Nondimensional super-critical chain state segment
        potential energy.
    
    """
    return -zeta_n_char**2 / (2.*kappa_n*(gamma_n-1.)**2)

def u_n_func(
        gamma_n: npt.ArrayLike,
        kappa_n: float,
        zeta_n_char: float,
        gamma_n_crit: float) -> npt.ArrayLike:
    """Nondimensional cuFJC segment potential energy.
        
    This function returns the nondimensional cuFJC segment potential
    energy.
    
    Args:
        gamma_n (npt.ArrayLike): Segment stretch.
        kappa_n (float): Nondimensional segment stiffness.
        zeta_n_char (float): Nondimensional characteristic segment potential energy scale.
        gamma_n_crit (float): Critical segment stretch.
    
    Returns:
        npt.ArrayLike: Nondimensional cuFJC segment potential energy.
    
    """
    gamma_n_isscalar = np.isscalar(gamma_n)
    if gamma_n_isscalar: gamma_n = np.asarray([gamma_n])
    u_n = np.empty_like(gamma_n)

    for indx in np.ndindex(np.shape(gamma_n)):
        gamma_n_val = gamma_n[indx]
        if gamma_n_val <= gamma_n_crit:
            u_n[indx] = u_n_subcrit_func(gamma_n_val, kappa_n, zeta_n_char)
        else:
            u_n[indx] = u_n_supercrit_func(
                gamma_n_val, kappa_n, zeta_n_char)

    return u_n.item() if gamma_n_isscalar else u_n

def u_n_prime_har_func(
        gamma_n: npt.ArrayLike,
        kappa_n: float) -> npt.ArrayLike:
    """Derivative of the nondimensional harmonic segment potential
    energy.
        
    This function returns the derivative of the nondimensional harmonic
    segment potential energy.
    
    Args:
        gamma_n (npt.ArrayLike): Segment stretch.
        kappa_n (float): Nondimensional segment stiffness.
    
    Returns:
        npt.ArrayLike: Derivative of the nondimensional harmonic segment
        potential energy.
    
    """
    return kappa_n * (gamma_n-1.)

def u_n_prime_subcrit_func(
        gamma_n: npt.ArrayLike,
        kappa_n: float) -> npt.ArrayLike:
    """Derivative of the nondimensional sub-critical chain state segment
    potential energy.
        
    This function returns the derivative of the nondimensional
    sub-critical chain state segment potential energy.
    
    Args:
        gamma_n (npt.ArrayLike): Segment stretch.
        kappa_n (float): Nondimensional segment stiffness.
    
    Returns:
        npt.ArrayLike: Derivative of the nondimensional sub-critical
        chain state segment potential energy.
    
    """
    return u_n_prime_har_func(gamma_n, kappa_n)

def u_n_prime_supercrit_func(
        gamma_n: npt.ArrayLike,
        kappa_n: float,
        zeta_n_char: float) -> npt.ArrayLike:
    """Derivative of the nondimensional super-critical chain state
    segment potential energy.
        
    This function returns the derivative of the nondimensional
    super-critical chain state segment potential energy.
    
    Args:
        gamma_n (npt.ArrayLike): Segment stretch.
        kappa_n (float): Nondimensional segment stiffness.
        zeta_n_char (float): Nondimensional characteristic segment potential energy scale.
    
    Returns:
        npt.ArrayLike: Derivative of the nondimensional super-critical
        chain state segment potential energy.
    
    """
    return zeta_n_char**2 / (kappa_n*(gamma_n-1.)**3)

def u_n_prime_func(
        gamma_n: npt.ArrayLike,
        kappa_n: float,
        zeta_n_char: float,
        gamma_n_crit: float) -> npt.ArrayLike:
    """Derivative of the nondimensional cuFJC segment potential energy.
        
    This function returns the derivative of the nondimensional cuFJC
    segment potential energy.
    
    Args:
        gamma_n (npt.ArrayLike): Segment stretch.
        kappa_n (float): Nondimensional segment stiffness.
        zeta_n_char (float): Nondimensional characteristic segment potential energy scale.
        gamma_n_crit (float): Critical segment stretch.
    
    Returns:
        npt.ArrayLike: Derivative of the nondimensional cuFJC segment
        potential energy.
    
    """
    gamma_n_isscalar = np.isscalar(gamma_n)
    if gamma_n_isscalar: gamma_n = np.asarray([gamma_n])
    u_n_prime = np.empty_like(gamma_n)

    for indx in np.ndindex(np.shape(gamma_n)):
        gamma_n_val = gamma_n[indx]
        if gamma_n_val <= gamma_n_crit:
            u_n_prime[indx] = u_n_prime_subcrit_func(gamma_n_val, kappa_n)
        else:
            u_n_prime[indx] = u_n_prime_supercrit_func(
                gamma_n_val, kappa_n, zeta_n_char)

    return u_n_prime.item() if gamma_n_isscalar else u_n_prime

def u_c_func(
        gamma_n: npt.ArrayLike,
        n: npt.ArrayLike,
        kappa_n: float,
        zeta_n_char: float,
        gamma_n_crit: float) -> npt.ArrayLike:
    """Nondimensional cuFJC chain potential energy.
        
    This function returns the nondimensional cuFJC chain potential
    energy.
    
    Args:
        gamma_n (npt.ArrayLike): Segment stretch.
        n (npt.ArrayLike): Number of chain segments.
        kappa_n (float): Nondimensional segment stiffness.
        zeta_n_char (float): Nondimensional characteristic segment potential energy scale.
        gamma_n_crit (float): Critical segment stretch.
    
    Returns:
        npt.ArrayLike: Nondimensional cuFJC chain potential energy.
    
    """
    return n * u_n_func(gamma_n, kappa_n, zeta_n_char, gamma_n_crit)

def u_c_prime_func(
        gamma_n: npt.ArrayLike,
        n: npt.ArrayLike,
        kappa_n: float,
        zeta_n_char: float,
        gamma_n_crit: float) -> npt.ArrayLike:
    """Derivative of the nondimensional cuFJC chain potential energy.
    
    This function returns the derivative of the nondimensional cuFJC
    chain potential energy.
    
    Args:
        gamma_n (npt.ArrayLike): Segment stretch.
        n (npt.ArrayLike): Number of chain segments.
        kappa_n (float): Nondimensional segment stiffness.
        zeta_n_char (float): Nondimensional characteristic segment potential energy scale.
        gamma_n_crit (float): Critical segment stretch.
    
    Returns:
        npt.ArrayLike: Derivative of the nondimensional cuFJC chain
        potential energy.
    
    """
    return n * u_n_prime_func(gamma_n, kappa_n, zeta_n_char, gamma_n_crit)

def u_n_tilde_subcrit_func(
        gamma_n: npt.ArrayLike,
        kappa_n: float,
        zeta_n_char: float) -> npt.ArrayLike:
    """Nondimensional shifted sub-critical chain state segment potential
    energy.
        
    This function returns the nondimensional shifted sub-critical chain
    state segment potential energy.
    
    Args:
        gamma_n (npt.ArrayLike): Segment stretch.
        kappa_n (float): Nondimensional segment stiffness.
        zeta_n_char (float): Nondimensional characteristic segment potential energy scale.
    
    Returns:
        npt.ArrayLike: Nondimensional shifted sub-critical chain state
        segment potential energy.
    
    """
    return u_n_subcrit_func(gamma_n, kappa_n, zeta_n_char) + zeta_n_char

def u_n_tilde_supercrit_func(
        gamma_n: npt.ArrayLike,
        kappa_n: float,
        zeta_n_char: float) -> npt.ArrayLike:
    """Nondimensional shifted super-critical chain state segment
    potential energy.
        
    This function returns the nondimensional shifted super-critical
    chain state segment potential energy.
    
    Args:
        gamma_n (npt.ArrayLike): Segment stretch.
        kappa_n (float): Nondimensional segment stiffness.
        zeta_n_char (float): Nondimensional characteristic segment potential energy scale.
    
    Returns:
        npt.ArrayLike: Nondimensional shifted super-critical chain state
        segment potential energy.
    
    """
    return u_n_supercrit_func(gamma_n, kappa_n, zeta_n_char) + zeta_n_char

def u_n_tilde_func(
        gamma_n: npt.ArrayLike,
        kappa_n: float,
        zeta_n_char: float,
        gamma_n_crit: float) -> npt.ArrayLike:
    """Nondimensional shifted cuFJC segment potential energy.
        
    This function returns the nondimensional shifted cuFJC segment
    potential energy.
    
    Args:
        gamma_n (npt.ArrayLike): Segment stretch.
        kappa_n (float): Nondimensional segment stiffness.
        zeta_n_char (float): Nondimensional characteristic segment potential energy scale.
        gamma_n_crit (float): Critical segment stretch.
    
    Returns:
        npt.ArrayLike: Nondimensional shifted cuFJC segment potential
        energy.
    
    """
    gamma_n_isscalar = np.isscalar(gamma_n)
    if gamma_n_isscalar: gamma_n = np.asarray([gamma_n])
    u_n_tilde = np.empty_like(gamma_n)

    for indx in np.ndindex(np.shape(gamma_n)):
        gamma_n_val = gamma_n[indx]
        if gamma_n_val <= gamma_n_crit:
            u_n_tilde[indx] = u_n_tilde_subcrit_func(
                gamma_n_val, kappa_n, zeta_n_char)
        else:
            u_n_tilde[indx] = u_n_tilde_supercrit_func(
                gamma_n_val, kappa_n, zeta_n_char)

    return u_n_tilde.item() if gamma_n_isscalar else u_n_tilde

def u_n_tilde_prime_subcrit_func(
        gamma_n: npt.ArrayLike,
        kappa_n: float) -> npt.ArrayLike:
    """Derivative of the nondimensional shifted sub-critical chain state
    segment potential energy.
        
    This function returns the derivative of the nondimensional shifted
    sub-critical chain state segment potential energy.
    
    Args:
        gamma_n (npt.ArrayLike): Segment stretch.
        kappa_n (float): Nondimensional segment stiffness.
    
    Returns:
        npt.ArrayLike: Derivative of the nondimensional shifted
        sub-critical chain state segment potential energy.
    
    """
    return u_n_prime_subcrit_func(gamma_n, kappa_n)

def u_n_tilde_prime_supercrit_func(
        gamma_n: npt.ArrayLike,
        kappa_n: float,
        zeta_n_char: float) -> npt.ArrayLike:
    """Derivative of the nondimensional shifted super-critical chain
    state segment potential energy.
        
    This function returns the derivative of the nondimensional shifted
    super-critical chain state segment potential energy.
    
    Args:
        gamma_n (npt.ArrayLike): Segment stretch.
        kappa_n (float): Nondimensional segment stiffness.
        zeta_n_char (float): Nondimensional characteristic segment potential energy scale.
    
    Returns:
        npt.ArrayLike: Derivative of the nondimensional shifted
        super-critical chain state segment potential energy.
    
    """
    return u_n_prime_supercrit_func(gamma_n, kappa_n, zeta_n_char)

def u_n_tilde_prime_func(
        gamma_n: npt.ArrayLike,
        kappa_n: float,
        zeta_n_char: float,
        gamma_n_crit: float) -> npt.ArrayLike:
    """Derivative of the nondimensional shifted cuFJC segment potential
    energy.
        
    This function returns the derivative of the nondimensional shifted
    cuFJC segment potential energy.
    
    Args:
        gamma_n (npt.ArrayLike): Segment stretch.
        kappa_n (float): Nondimensional segment stiffness.
        zeta_n_char (float): Nondimensional characteristic segment potential energy scale.
        gamma_n_crit (float): Critical segment stretch.
    
    Returns:
        npt.ArrayLike: Derivative of the nondimensional shifted cuFJC
        segment potential energy.
    
    """
    gamma_n_isscalar = np.isscalar(gamma_n)
    if gamma_n_isscalar: gamma_n = np.asarray([gamma_n])
    u_n_tilde_prime = np.empty_like(gamma_n)

    for indx in np.ndindex(np.shape(gamma_n)):
        gamma_n_val = gamma_n[indx]
        if gamma_n_val <= gamma_n_crit:
            u_n_tilde_prime[indx] = u_n_tilde_prime_subcrit_func(
                gamma_n_val, kappa_n)
        else:
            u_n_tilde_prime[indx] = u_n_tilde_prime_supercrit_func(
                gamma_n_val, kappa_n, zeta_n_char)

    return u_n_tilde_prime.item() if gamma_n_isscalar else u_n_tilde_prime

def u_c_tilde_func(
        gamma_n: npt.ArrayLike,
        n: npt.ArrayLike,
        kappa_n: float,
        zeta_n_char: float,
        gamma_n_crit: float) -> npt.ArrayLike:
    """Nondimensional shifted cuFJC chain potential energy.
        
    This function returns the nondimensional shifted cuFJC chain
    potential energy.
    
    Args:
        gamma_n (npt.ArrayLike): Segment stretch.
        n (npt.ArrayLike): Number of chain segments.
        kappa_n (float): Nondimensional segment stiffness.
        zeta_n_char (float): Nondimensional characteristic segment potential energy scale.
        gamma_n_crit (float): Critical segment stretch.
    
    Returns:
        npt.ArrayLike: Nondimensional shifted cuFJC chain potential
        energy.
    
    """
    return n * u_n_tilde_func(gamma_n, kappa_n, zeta_n_char, gamma_n_crit)

def u_c_tilde_prime_func(
        gamma_n: npt.ArrayLike,
        n: npt.ArrayLike,
        kappa_n: float,
        zeta_n_char: float,
        gamma_n_crit: float) -> npt.ArrayLike:
    """Derivative of the nondimensional shifted cuFJC chain potential
    energy.
        
    This function returns the derivative of the nondimensional shifted
    cuFJC chain potential energy.
    
    Args:
        gamma_n (npt.ArrayLike): Segment stretch.
        n (npt.ArrayLike): Number of chain segments.
        kappa_n (float): Nondimensional segment stiffness.
        zeta_n_char (float): Nondimensional characteristic segment potential energy scale.
        gamma_n_crit (float): Critical segment stretch.
    
    Returns:
        npt.ArrayLike: Derivative of the nondimensional shifted cuFJC
        chain potential energy.
    
    """
    return (
        n
        * u_n_tilde_prime_func(gamma_n, kappa_n, zeta_n_char, gamma_n_crit)
    )

def s_cn_func(
        gamma: npt.ArrayLike,
        kappa_n: float,
        zeta_n_char: float,
        gamma_pade_to_bergstrom_crit: float,
        gamma_crit: float) -> npt.ArrayLike:
    """Nondimensional chain-level entropic free energy contribution
    per segment as calculated by the Jedynak R[9,2] inverse Langevin
    approximate.
    
    This function returns the nondimensional chain-level entropic
    free energy contribution per segment as calculated by the
    Jedynak R[9,2] inverse Langevin approximate.

    Args:
        gamma (npt.ArrayLike): Absolute/Equilibrium chain stretch.
        kappa_n (float): Nondimensional segment stiffness.
        zeta_n_char (float): Nondimensional characteristic segment potential energy scale.
        gamma_pade_to_bergstrom_crit (float): Pade-to-Bergstrom critical absolute/equilibrium segment stretch.
        gamma_crit (float): Critical absolute/equilibrium segment stretch.
    
    Returns:
        npt.ArrayLike: Nondimensional chain-level entropic free energy
        contribution per segment as calculated by the Jedynak R[9,2]
        inverse Langevin approximate.
    
    """
    gamma_n = gamma_n_func(
        gamma, kappa_n, zeta_n_char, gamma_pade_to_bergstrom_crit, gamma_crit)
    gamma_comp_n = gamma - gamma_n + 1.
    return s_cn_inv_langevin_fjc_func(gamma_comp_n)

def s_c_func(
        gamma: npt.ArrayLike,
        n: npt.ArrayLike,
        kappa_n: float,
        zeta_n_char: float,
        gamma_pade_to_bergstrom_crit: float,
        gamma_crit: float) -> npt.ArrayLike:
    """Nondimensional chain entropic free energy contribution as
    calculated by the Jedynak R[9,2] inverse Langevin approximate.
    
    This function returns the nondimensional chain entropic free energy
    contribution as calculated by the Jedynak R[9,2] inverse Langevin
    approximate.

    Args:
        gamma (npt.ArrayLike): Absolute/Equilibrium chain stretch.
        n (npt.ArrayLike): Number of chain segments.
        kappa_n (float): Nondimensional segment stiffness.
        zeta_n_char (float): Nondimensional characteristic segment potential energy scale.
        gamma_pade_to_bergstrom_crit (float): Pade-to-Bergstrom critical absolute/equilibrium segment stretch.
        gamma_crit (float): Critical absolute/equilibrium segment stretch.
    
    Returns:
        npt.ArrayLike: Nondimensional chain entropic free energy
        contribution as calculated by the Jedynak R[9,2] inverse
        Langevin approximate.
    
    """
    s_cn = s_cn_func(
        gamma, kappa_n, zeta_n_char, gamma_pade_to_bergstrom_crit,
        gamma_crit)
    return n * s_cn

def psi_cn_func(
        gamma: npt.ArrayLike,
        kappa_n: float,
        zeta_n_char: float,
        gamma_n_crit: float,
        gamma_pade_to_bergstrom_crit: float,
        gamma_crit: float) -> npt.ArrayLike:
    """Nondimensional chain-level Helmholtz free energy per segment.
    
    This function returns the nondimensional chain-level Helmholtz free
    energy per segment.

    Args:
        gamma (npt.ArrayLike): Absolute/Equilibrium chain stretch.
        kappa_n (float): Nondimensional segment stiffness.
        zeta_n_char (float): Nondimensional characteristic segment potential energy scale.
        gamma_n_crit (float): Critical segment stretch.
        gamma_pade_to_bergstrom_crit (float): Pade-to-Bergstrom critical absolute/equilibrium segment stretch.
        gamma_crit (float): Critical absolute/equilibrium segment stretch.
    
    Returns:
        npt.ArrayLike: Nondimensional chain-level Helmholtz free energy
        per segment.
    
    """
    gamma_n = gamma_n_func(
        gamma, kappa_n, zeta_n_char, gamma_pade_to_bergstrom_crit,
        gamma_crit)
    u_n = u_n_func(gamma_n, kappa_n, zeta_n_char, gamma_n_crit)
    s_cn = s_cn_func(
        gamma, kappa_n, zeta_n_char, gamma_pade_to_bergstrom_crit,
        gamma_crit)
    return u_n + s_cn

def psi_c_func(
        gamma: npt.ArrayLike,
        n: npt.ArrayLike,
        kappa_n: float,
        zeta_n_char: float,
        gamma_n_crit: float,
        gamma_pade_to_bergstrom_crit: float,
        gamma_crit: float) -> npt.ArrayLike:
    """Nondimensional chain Helmholtz free energy.
    
    This function returns the nondimensional chain Helmholtz free
    energy.

    Args:
        gamma (npt.ArrayLike): Absolute/Equilibrium chain stretch.
        n (npt.ArrayLike): Number of chain segments.
        kappa_n (float): Nondimensional segment stiffness.
        zeta_n_char (float): Nondimensional characteristic segment potential energy scale.
        gamma_n_crit (float): Critical segment stretch.
        gamma_pade_to_bergstrom_crit (float): Pade-to-Bergstrom critical absolute/equilibrium segment stretch.
        gamma_crit (float): Critical absolute/equilibrium segment stretch.

    Returns:
        npt.ArrayLike: Nondimensional chain Helmholtz free energy.
    
    """
    psi_cn = psi_cn_func(
        gamma, kappa_n, zeta_n_char, gamma_n_crit,
        gamma_pade_to_bergstrom_crit, gamma_crit)
    return n * psi_cn

def w_cn_func(
        gamma: npt.ArrayLike,
        kappa_n: float,
        zeta_n_char: float,
        gamma_n_crit: float,
        gamma_pade_to_bergstrom_crit: float,
        gamma_crit: float) -> npt.ArrayLike:
    """Nondimensional shifted chain-level Helmholtz free energy per
    segment.
    
    This function returns the nondimensional shifted chain-level
    Helmholtz free energy per segment.

    Args:
        gamma (npt.ArrayLike): Absolute/Equilibrium chain stretch.
        kappa_n (float): Nondimensional segment stiffness.
        zeta_n_char (float): Nondimensional characteristic segment potential energy scale.
        gamma_n_crit (float): Critical segment stretch.
        gamma_pade_to_bergstrom_crit (float): Pade-to-Bergstrom critical absolute/equilibrium segment stretch.
        gamma_crit (float): Critical absolute/equilibrium segment stretch.
    
    Returns:
        npt.ArrayLike: Nondimensional shifted chain-level Helmholtz free
        energy per segment.
    
    """
    gamma_n = gamma_n_func(
        gamma, kappa_n, zeta_n_char, gamma_pade_to_bergstrom_crit,
        gamma_crit)
    u_n_tilde = u_n_tilde_func(gamma_n, kappa_n, zeta_n_char, gamma_n_crit)
    s_cn = s_cn_func(
        gamma, kappa_n, zeta_n_char, gamma_pade_to_bergstrom_crit,
        gamma_crit)
    return u_n_tilde + s_cn

def w_c_func(
        gamma: npt.ArrayLike,
        n: npt.ArrayLike,
        kappa_n: float,
        zeta_n_char: float,
        gamma_n_crit: float,
        gamma_pade_to_bergstrom_crit: float,
        gamma_crit: float) -> npt.ArrayLike:
    """Nondimensional shifted chain Helmholtz free energy.
    
    This function returns the nondimensional shifted chain Helmholtz
    free energy.

    Args:
        gamma (npt.ArrayLike): Absolute/Equilibrium chain stretch.
        n (npt.ArrayLike): Number of chain segments.
        kappa_n (float): Nondimensional segment stiffness.
        zeta_n_char (float): Nondimensional characteristic segment potential energy scale.
        gamma_n_crit (float): Critical segment stretch.
        gamma_pade_to_bergstrom_crit (float): Pade-to-Bergstrom critical absolute/equilibrium segment stretch.
        gamma_crit (float): Critical absolute/equilibrium segment stretch.

    Returns:
        npt.ArrayLike: Nondimensional shifted chain Helmholtz free
        energy.
    
    """
    w_cn = w_cn_func(
        gamma, kappa_n, zeta_n_char, gamma_n_crit,
        gamma_pade_to_bergstrom_crit, gamma_crit)
    return n * w_cn

def xi_c_func(
        gamma: npt.ArrayLike,
        kappa_n: float,
        zeta_n_char: float,
        gamma_pade_to_bergstrom_crit: float,
        gamma_crit: float) -> npt.ArrayLike:
    """Nondimensional chain force.
    
    This function returns the nondimensional chain force.

    Args:
        gamma (npt.ArrayLike): Absolute/Equilibrium chain stretch.
        kappa_n (float): Nondimensional segment stiffness.
        zeta_n_char (float): Nondimensional characteristic segment potential energy scale.
        gamma_pade_to_bergstrom_crit (float): Pade-to-Bergstrom critical absolute/equilibrium segment stretch.
        gamma_crit (float): Critical absolute/equilibrium segment stretch.
    
    Returns:
        npt.ArrayLike: Nondimensional chain force.
    
    """
    gamma_n = gamma_n_func(
        gamma, kappa_n, zeta_n_char, gamma_pade_to_bergstrom_crit, gamma_crit)
    gamma_comp_n = gamma - gamma_n + 1.
    return inv_langevin_func(gamma_comp_n)

def xi_c_vec_func(
        gamma_vec: npt.NDArray[np.floating],
        gamma: float,
        kappa_n: float,
        zeta_n_char: float,
        gamma_pade_to_bergstrom_crit: float,
        gamma_crit: float) -> npt.NDArray[np.floating]:
    """Nondimensional chain force vector.
    
    This function returns the nondimensional chain force vector.

    Args:
        gamma_vec (npt.NDArray[np.floating]): Absolute/Equilibrium chain stretch vector.
        gamma (float): Absolute/Equilibrium chain stretch.
        kappa_n (float): Nondimensional segment stiffness.
        zeta_n_char (float): Nondimensional characteristic segment potential energy scale.
        gamma_pade_to_bergstrom_crit (float): Pade-to-Bergstrom critical absolute/equilibrium segment stretch.
        gamma_crit (float): Critical absolute/equilibrium segment stretch.
    
    Returns:
        npt.NDArray[np.floating]: Nondimensional chain force vector.
    
    """
    return (
        xi_c_func(gamma, kappa_n, zeta_n_char, gamma_pade_to_bergstrom_crit, gamma_crit)
        * gamma_vec / gamma
    )

def dw_c__dy_clnk_func(
        gamma_vec: npt.NDArray[np.floating],
        gamma: float,
        kappa_n: float,
        zeta_n_char: float,
        gamma_pade_to_bergstrom_crit: float,
        gamma_crit: float) -> npt.NDArray[np.floating]:
    """Nondimensional derivative of the polymer chain free energy with
    respect to the cross-link junction position for a chain in the
    cross-link structure RVE.
     
    This function returns the nondimensional derivative of the polymer
    chain free energy with respect to the cross-link junction position
    for a chain in the cross-link structure RVE.

    Args:
        gamma_vec (npt.NDArray[np.floating]): Absolute/Equilibrium chain stretch vector.
        gamma (float): Absolute/Equilibrium chain stretch.
        kappa_n (float): Nondimensional segment stiffness.
        zeta_n_char (float): Nondimensional characteristic segment potential energy scale.
        gamma_pade_to_bergstrom_crit (float): Pade-to-Bergstrom critical absolute/equilibrium segment stretch.
        gamma_crit (float): Critical absolute/equilibrium segment stretch.
    
    Returns:
        npt.NDArray[np.floating]: Nondimensional derivative of the
        polymer chain free energy with respect to the cross-link
        junction position for a chain in the cross-link structure RVE.
    
    """
    return (
        -xi_c_vec_func(
            gamma_vec, gamma, kappa_n, zeta_n_char, gamma_pade_to_bergstrom_crit,
            gamma_crit)
    )

def d2w_c__dy_clnk_dy_clnk_func(
        gamma_vec: npt.NDArray[np.floating],
        gamma: float,
        n: float | int,
        kappa_n: float,
        zeta_n_char: float,
        gamma_pade_to_bergstrom_crit: float,
        gamma_crit: float) -> npt.NDArray[np.floating]:
    """Nondimensional second derivative of the polymer chain free energy
    with respect to the cross-link junction position for a chain in the
    cross-link structure RVE.
     
    This function returns the nondimensional second derivative of the
    polymer chain free energy with respect to the cross-link junction
    position for a chain in the cross-link structure RVE.

    Args:
        gamma_vec (npt.NDArray[np.floating]): Absolute/Equilibrium chain stretch vector.
        gamma (float): Absolute/Equilibrium chain stretch.
        n (float | int): Number of chain segments.
        kappa_n (float): Nondimensional segment stiffness.
        zeta_n_char (float): Nondimensional characteristic segment potential energy scale.
        gamma_pade_to_bergstrom_crit (float): Pade-to-Bergstrom critical absolute/equilibrium segment stretch.
        gamma_crit (float): Critical absolute/equilibrium segment stretch.
    
    Returns:
        npt.NDArray[np.floating]: Nondimensional second derivative of
        the polymer chain free energy with respect to the cross-link
        junction position for a chain in the cross-link structure RVE.
    
    """
    gamma_n = gamma_n_func(
        gamma, kappa_n, zeta_n_char, gamma_pade_to_bergstrom_crit, gamma_crit)
    gamma_comp_n = gamma - gamma_n + 1.
    unit_gamma_vec = gamma_vec / gamma
    unit_gamma_vec_outer_prod = np.outer(unit_gamma_vec, unit_gamma_vec)
    return (
        (inv_langevin_prime_func(gamma_comp_n)*unit_gamma_vec_outer_prod+inv_langevin_func(gamma_comp_n)/gamma*(np.eye(3)-unit_gamma_vec_outer_prod))
        / n
    )

def u_n_tot_hat_func(
        gamma_n_hat: npt.ArrayLike,
        gamma_n: float,
        kappa_n: float,
        zeta_n_char: float,
        gamma_n_pade_to_bergstrom_crit: float,
        gamma_n_crit: float,
        gamma_pade_to_bergstrom_crit: float,
        gamma_crit: float) -> npt.ArrayLike:
    """Nondimensional total segment potential under an applied chain
    force.
    
    This function computes the nondimensional total segment potential
    under an applied chain force.

    Args:
        gamma_n_hat (npt.ArrayLike): Applied segment stretch.
        gamma_n (float): Segment stretch specifying a particular state in the energy landscape.
        kappa_n (float): Nondimensional segment stiffness.
        zeta_n_char (float): Nondimensional characteristic segment potential energy scale.
        gamma_n_pade_to_bergstrom_crit (float): Pade-to-Bergstrom critical segment stretch.
        gamma_n_crit (float): Critical segment stretch.
        gamma_pade_to_bergstrom_crit (float): Pade-to-Bergstrom critical absolute/equilibrium chain stretch.
        gamma_crit (float): Critical absolute/equilibrium chain stretch.
    
    Returns:
        npt.ArrayLike: Nondimensional total segment potential under an
        applied chain force.

    """
    u_n = u_n_func(gamma_n, kappa_n, zeta_n_char, gamma_n_crit)
    gamma_hat = gamma_func(
        gamma_n_hat, kappa_n, zeta_n_char, gamma_n_pade_to_bergstrom_crit,
        gamma_n_crit)
    xi_c_hat = xi_c_func(
        gamma_hat, kappa_n, zeta_n_char,
        gamma_pade_to_bergstrom_crit, gamma_crit)
    return u_n - gamma_n * xi_c_hat

def u_n_hat_func(
        gamma_n_hat: npt.ArrayLike,
        gamma_n: float,
        kappa_n: float,
        zeta_n_char: float,
        gamma_n_pade_to_bergstrom_crit: float,
        gamma_n_crit: float,
        gamma_pade_to_bergstrom_crit: float,
        gamma_crit: float) -> npt.ArrayLike:
    """Nondimensional total distorted segment potential under an applied
    chain force.
    
    This function computes the nondimensional total distorted segment
    potential under an applied chain force.

    Args:
        gamma_n_hat (npt.ArrayLike): Applied segment stretch.
        gamma_n (float): Segment stretch specifying a particular state in the energy landscape.
        kappa_n (float): Nondimensional segment stiffness.
        zeta_n_char (float): Nondimensional characteristic segment potential energy scale.
        gamma_n_pade_to_bergstrom_crit (float): Pade-to-Bergstrom critical segment stretch.
        gamma_n_crit (float): Critical segment stretch.
        gamma_pade_to_bergstrom_crit (float): Pade-to-Bergstrom critical absolute/equilibrium chain stretch.
        gamma_crit (float): Critical absolute/equilibrium chain stretch.
    
    Returns:
        npt.ArrayLike: Nondimensional total distorted segment potential
        under an applied chain force.
    
    """
    u_n = u_n_func(gamma_n, kappa_n, zeta_n_char, gamma_n_crit)
    gamma_hat = gamma_func(
        gamma_n_hat, kappa_n, zeta_n_char, gamma_n_pade_to_bergstrom_crit,
        gamma_n_crit)
    xi_c_hat = xi_c_func(
        gamma_hat, kappa_n, zeta_n_char,
        gamma_pade_to_bergstrom_crit, gamma_crit)
    return u_n - (gamma_n-gamma_n_hat) * xi_c_hat

def gamma_n_locmin_hat_func(
        gamma_n_hat: npt.ArrayLike,
        kappa_n: float,
        zeta_n_char: float,
        gamma_n_pade_to_bergstrom_crit: float,
        gamma_n_crit: float,
        gamma_pade_to_bergstrom_crit: float,
        gamma_crit: float) -> npt.ArrayLike:
    """Segment stretch corresponding to the local minimum of the
    nondimensional total (distorted) segment potential under an applied
    chain force.
    
    This function computes the segment stretch corresponding to the
    local minimum of the nondimensional total (distorted) segment
    potential under an applied chain force.

    Args:
        gamma_n_hat (npt.ArrayLike): Applied segment stretch.
        kappa_n (float): Nondimensional segment stiffness.
        zeta_n_char (float): Nondimensional characteristic segment potential energy scale.
        gamma_n_pade_to_bergstrom_crit (float): Pade-to-Bergstrom critical segment stretch.
        gamma_n_crit (float): Critical segment stretch.
        gamma_pade_to_bergstrom_crit (float): Pade-to-Bergstrom critical absolute/equilibrium chain stretch.
        gamma_crit (float): Critical absolute/equilibrium chain stretch.
    
    Returns:
        npt.ArrayLike: Segment stretch corresponding to the local
        minimum of the nondimensional total (distorted) segment
        potential under an applied chain force.
    
    """
    gamma_hat = gamma_func(
        gamma_n_hat, kappa_n, zeta_n_char, gamma_n_pade_to_bergstrom_crit,
        gamma_n_crit)
    xi_c_hat = xi_c_func(
        gamma_hat, kappa_n, zeta_n_char,
        gamma_pade_to_bergstrom_crit, gamma_crit)
    return 1. + xi_c_hat / kappa_n

def gamma_n_locmax_hat_func(
        gamma_n_hat: npt.ArrayLike,
        kappa_n: float,
        zeta_n_char: float,
        gamma_n_pade_to_bergstrom_crit: float,
        gamma_n_crit: float,
        gamma_pade_to_bergstrom_crit: float,
        gamma_crit: float) -> npt.ArrayLike:
    """Segment stretch corresponding to the local maximum of the
    nondimensional total (distorted) segment potential under an applied
    chain force.
    
    This function computes the segment stretch corresponding to the
    local maximum of the nondimensional total (distorted) segment
    potential under an applied chain force.

    Args:
        gamma_n_hat (npt.ArrayLike): Applied segment stretch.
        kappa_n (float): Nondimensional segment stiffness.
        zeta_n_char (float): Nondimensional characteristic segment potential energy scale.
        gamma_n_pade_to_bergstrom_crit (float): Pade-to-Bergstrom critical segment stretch.
        gamma_n_crit (float): Critical segment stretch.
        gamma_pade_to_bergstrom_crit (float): Pade-to-Bergstrom critical absolute/equilibrium chain stretch.
        gamma_crit (float): Critical absolute/equilibrium chain stretch.
    
    Returns:
        npt.ArrayLike: Segment stretch corresponding to the local
        maximum of the nondimensional total (distorted) segment
        potential under an applied chain force.
    
    """
    gamma_n_hat_isscalar = np.isscalar(gamma_n_hat)
    if gamma_n_hat_isscalar: gamma_n_hat = np.asarray([gamma_n_hat])
    gamma_n_locmax_hat = np.empty_like(gamma_n_hat)

    for indx in np.ndindex(np.shape(gamma_n_hat)):
        gamma_n_hat_val = gamma_n_hat[indx]
        if gamma_n_hat_val <= 1.: gamma_n_locmax_hat[indx] = np.inf
        else:
            gamma_hat_val = gamma_func(
                gamma_n_hat_val, kappa_n, zeta_n_char,
                gamma_n_pade_to_bergstrom_crit, gamma_n_crit)
            xi_c_hat_val = xi_c_func(
                gamma_hat_val, kappa_n, zeta_n_char,
                gamma_pade_to_bergstrom_crit, gamma_crit)
            cbrt_arg = zeta_n_char**2 / (kappa_n*xi_c_hat_val)
            gamma_n_locmax_hat[indx] = 1. + np.cbrt(cbrt_arg)

    return gamma_n_locmax_hat.item() if gamma_n_hat_isscalar else gamma_n_locmax_hat

def epsilon_n_sci_hat_func(
        gamma_n_hat: npt.ArrayLike,
        kappa_n: float,
        zeta_n_char: float,
        gamma_n_pade_to_bergstrom_crit: float,
        gamma_n_crit: float,
        gamma_pade_to_bergstrom_crit: float,
        gamma_crit: float) -> npt.ArrayLike:
    """Nondimensional segment scission energy.
    
    This function computes the nondimensional segment scission energy.

    Args:
        gamma_n_hat (npt.ArrayLike): Applied segment stretch.
        kappa_n (float): Nondimensional segment stiffness.
        zeta_n_char (float): Nondimensional characteristic segment potential energy scale.
        gamma_n_pade_to_bergstrom_crit (float): Pade-to-Bergstrom critical segment stretch.
        gamma_n_crit (float): Critical segment stretch.
        gamma_pade_to_bergstrom_crit (float): Pade-to-Bergstrom critical absolute/equilibrium chain stretch.
        gamma_crit (float): Critical absolute/equilibrium chain stretch.
    
    Returns:
        npt.ArrayLike: Nondimensional segment scission energy.
    
    """
    gamma_hat = gamma_func(
        gamma_n_hat, kappa_n, zeta_n_char, gamma_n_pade_to_bergstrom_crit,
        gamma_n_crit)
    return (
        w_cn_func(
            gamma_hat, kappa_n, zeta_n_char, gamma_n_crit,
            gamma_pade_to_bergstrom_crit, gamma_crit)
    )

def epsilon_cn_sci_hat_func(
        gamma_n_hat: npt.ArrayLike,
        kappa_n: float,
        zeta_n_char: float,
        gamma_n_pade_to_bergstrom_crit: float,
        gamma_n_crit: float,
        gamma_pade_to_bergstrom_crit: float,
        gamma_crit: float) -> npt.ArrayLike:
    """Nondimensional chain scission energy per segment.
    
    This function computes the nondimensional chain scission energy per
    segment.

    Args:
        gamma_n_hat (npt.ArrayLike): Applied segment stretch.
        kappa_n (float): Nondimensional segment stiffness.
        zeta_n_char (float): Nondimensional characteristic segment potential energy scale.
        gamma_n_pade_to_bergstrom_crit (float): Pade-to-Bergstrom critical segment stretch.
        gamma_n_crit (float): Critical segment stretch.
        gamma_pade_to_bergstrom_crit (float): Pade-to-Bergstrom critical absolute/equilibrium chain stretch.
        gamma_crit (float): Critical absolute/equilibrium chain stretch.
    
    Returns:
        npt.ArrayLike: Nondimensional chain scission energy per segment.
    
    """
    return (
        epsilon_n_sci_hat_func(
            gamma_n_hat, kappa_n, zeta_n_char,
            gamma_n_pade_to_bergstrom_crit, gamma_n_crit,
            gamma_pade_to_bergstrom_crit, gamma_crit)
    )

def e_n_sci_hat_func(
        gamma_n_hat: npt.ArrayLike,
        kappa_n: float,
        zeta_n_char: float,
        gamma_n_pade_to_bergstrom_crit: float,
        gamma_n_crit: float,
        gamma_pade_to_bergstrom_crit: float,
        gamma_crit: float) -> npt.ArrayLike:
    """Nondimensional segment scission activation energy barrier.
    
    This function computes the nondimensional segment scission
    activation energy barrier.

    Args:
        gamma_n_hat (npt.ArrayLike): Applied segment stretch.
        kappa_n (float): Nondimensional segment stiffness.
        zeta_n_char (float): Nondimensional characteristic segment potential energy scale.
        gamma_n_pade_to_bergstrom_crit (float): Pade-to-Bergstrom critical segment stretch.
        gamma_n_crit (float): Critical segment stretch.
        gamma_pade_to_bergstrom_crit (float): Pade-to-Bergstrom critical absolute/equilibrium chain stretch.
        gamma_crit (float): Critical absolute/equilibrium chain stretch.
    
    Returns:
        npt.ArrayLike: Nondimensional segment scission activation energy
        barrier.
    
    """
    gamma_n_hat_isscalar = np.isscalar(gamma_n_hat)
    if gamma_n_hat_isscalar: gamma_n_hat = np.asarray([gamma_n_hat])
    e_n_sci_hat = np.empty_like(gamma_n_hat)

    for indx in np.ndindex(np.shape(gamma_n_hat)):
        gamma_n_hat_val = gamma_n_hat[indx]
        if gamma_n_hat_val <= 1.: e_n_sci_hat[indx] = zeta_n_char
        elif gamma_n_hat_val <= gamma_n_crit:
            gamma_n_locmin_hat_val = gamma_n_locmin_hat_func(
                gamma_n_hat_val, kappa_n, zeta_n_char,
                gamma_n_pade_to_bergstrom_crit, gamma_n_crit,
                gamma_pade_to_bergstrom_crit, gamma_crit)
            gamma_n_locmax_hat_val = gamma_n_locmax_hat_func(
                gamma_n_hat_val, kappa_n, zeta_n_char,
                gamma_n_pade_to_bergstrom_crit, gamma_n_crit,
                gamma_pade_to_bergstrom_crit, gamma_crit)
            u_n_hat_gamma_n_locmin_hat_val = u_n_hat_func(
                gamma_n_hat_val, gamma_n_locmin_hat_val, kappa_n,
                zeta_n_char, gamma_n_pade_to_bergstrom_crit, gamma_n_crit,
                gamma_pade_to_bergstrom_crit, gamma_crit)
            u_n_hat_gamma_n_locmax_hat_val = u_n_hat_func(
                gamma_n_hat_val, gamma_n_locmax_hat_val, kappa_n,
                zeta_n_char, gamma_n_pade_to_bergstrom_crit, gamma_n_crit,
                gamma_pade_to_bergstrom_crit, gamma_crit)
            e_n_sci_hat[indx] = (
                u_n_hat_gamma_n_locmax_hat_val
                - u_n_hat_gamma_n_locmin_hat_val
            )
        else: e_n_sci_hat[indx] = 0.
    
    return e_n_sci_hat.item() if gamma_n_hat_isscalar else e_n_sci_hat

def p_n_sci_hat_func(
        gamma_n_hat: npt.ArrayLike,
        kappa_n: float,
        zeta_n_char: float,
        gamma_n_pade_to_bergstrom_crit: float,
        gamma_n_crit: float,
        gamma_pade_to_bergstrom_crit: float,
        gamma_crit: float) -> npt.ArrayLike:
    """Rate-independent probability of segment scission.
    
    This function computes the rate-independent probability of segment
    scission.

    Args:
        gamma_n_hat (npt.ArrayLike): Applied segment stretch.
        kappa_n (float): Nondimensional segment stiffness.
        zeta_n_char (float): Nondimensional characteristic segment potential energy scale.
        gamma_n_pade_to_bergstrom_crit (float): Pade-to-Bergstrom critical segment stretch.
        gamma_n_crit (float): Critical segment stretch.
        gamma_pade_to_bergstrom_crit (float): Pade-to-Bergstrom critical absolute/equilibrium chain stretch.
        gamma_crit (float): Critical absolute/equilibrium chain stretch.
    
    Returns:
        npt.ArrayLike: Rate-independent probability of segment scission.
    
    """
    e_n_sci_hat = e_n_sci_hat_func(
        gamma_n_hat, kappa_n, zeta_n_char, gamma_n_pade_to_bergstrom_crit,
        gamma_n_crit, gamma_pade_to_bergstrom_crit, gamma_crit)
    return p_boltzmann_func(e_n_sci_hat)

def p_n_sur_hat_func(
        gamma_n_hat: npt.ArrayLike,
        kappa_n: float,
        zeta_n_char: float,
        gamma_n_pade_to_bergstrom_crit: float,
        gamma_n_crit: float,
        gamma_pade_to_bergstrom_crit: float,
        gamma_crit: float) -> npt.ArrayLike:
    """Rate-independent probability of segment survival.
    
    This function computes the rate-independent probability of segment
    survival.

    Args:
        gamma_n_hat (npt.ArrayLike): Applied segment stretch.
        kappa_n (float): Nondimensional segment stiffness.
        zeta_n_char (float): Nondimensional characteristic segment potential energy scale.
        gamma_n_pade_to_bergstrom_crit (float): Pade-to-Bergstrom critical segment stretch.
        gamma_n_crit (float): Critical segment stretch.
        gamma_pade_to_bergstrom_crit (float): Pade-to-Bergstrom critical absolute/equilibrium chain stretch.
        gamma_crit (float): Critical absolute/equilibrium chain stretch.
    
    Returns:
        npt.ArrayLike: Rate-independent probability of segment survival.
    
    """
    p_n_sci_hat = p_n_sci_hat_func(
        gamma_n_hat, kappa_n, zeta_n_char, gamma_n_pade_to_bergstrom_crit,
        gamma_n_crit, gamma_pade_to_bergstrom_crit, gamma_crit)
    return 1. - p_n_sci_hat

def p_c_sur_hat_func(
        gamma_n_hat: npt.ArrayLike,
        n: npt.ArrayLike,
        kappa_n: float,
        zeta_n_char: float,
        gamma_n_pade_to_bergstrom_crit: float,
        gamma_n_crit: float,
        gamma_pade_to_bergstrom_crit: float,
        gamma_crit: float) -> npt.ArrayLike:
    """Rate-independent probability of chain survival.
    
    This function computes the rate-independent probability of chain
    survival.

    Args:
        gamma_n_hat (npt.ArrayLike): Applied segment stretch.
        n (npt.ArrayLike): Number of chain segments.
        kappa_n (float): Nondimensional segment stiffness.
        zeta_n_char (float): Nondimensional characteristic segment potential energy scale.
        gamma_n_pade_to_bergstrom_crit (float): Pade-to-Bergstrom critical segment stretch.
        gamma_n_crit (float): Critical segment stretch.
        gamma_pade_to_bergstrom_crit (float): Pade-to-Bergstrom critical absolute/equilibrium chain stretch.
        gamma_crit (float): Critical absolute/equilibrium chain stretch.
    
    Returns:
        npt.ArrayLike: Rate-independent probability of chain survival.
    
    """
    p_n_sur_hat = p_n_sur_hat_func(
        gamma_n_hat, kappa_n, zeta_n_char, gamma_n_pade_to_bergstrom_crit,
        gamma_n_crit, gamma_pade_to_bergstrom_crit, gamma_crit)
    return p_n_sur_hat**n

def p_c_sci_hat_func(
        gamma_n_hat: npt.ArrayLike,
        n: npt.ArrayLike,
        kappa_n: float,
        zeta_n_char: float,
        gamma_n_pade_to_bergstrom_crit: float,
        gamma_n_crit: float,
        gamma_pade_to_bergstrom_crit: float,
        gamma_crit: float) -> npt.ArrayLike:
    """Rate-independent probability of chain scission.
    
    This function computes the rate-independent probability of chain
    scission.

    Args:
        gamma_n_hat (npt.ArrayLike): Applied segment stretch.
        n (npt.ArrayLike): Number of chain segments.
        kappa_n (float): Nondimensional segment stiffness.
        zeta_n_char (float): Nondimensional characteristic segment potential energy scale.
        gamma_n_pade_to_bergstrom_crit (float): Pade-to-Bergstrom critical segment stretch.
        gamma_n_crit (float): Critical segment stretch.
        gamma_pade_to_bergstrom_crit (float): Pade-to-Bergstrom critical absolute/equilibrium chain stretch.
        gamma_crit (float): Critical absolute/equilibrium chain stretch.
    
    Returns:
        npt.ArrayLike: Rate-independent probability of chain scission.
    
    """
    p_c_sur_hat = p_c_sur_hat_func(
        gamma_n_hat, n, kappa_n, zeta_n_char,
        gamma_n_pade_to_bergstrom_crit, gamma_n_crit,
        gamma_pade_to_bergstrom_crit, gamma_crit)
    return 1. - p_c_sur_hat

def epsilon_n_diss_hat_rate_independent_scission_func(
        gamma_n_hat_val: float,
        gamma_n_hat_val_prior: float,
        gamma_n_hat_max_val: float,
        gamma_n_hat_max_val_prior: float,
        epsilon_n_diss_hat_val_prior: float,
        kappa_n: float,
        zeta_n_char: float,
        gamma_n_pade_to_bergstrom_crit: float,
        gamma_n_crit: float,
        gamma_pade_to_bergstrom_crit: float,
        gamma_crit: float) -> float:
    """Nondimensional rate-independent dissipated segment scission
    energy.
    
    This function computes the nondimensional rate-independent
    dissipated segment scission energy at the current state.

    Args:
        gamma_n_hat_val (float): Applied segment stretch at the current state.
        gamma_n_hat_val_prior (float): Applied segment stretch at the prior state.
        gamma_n_hat_max_val (float): Maximum of the applied segment stretch through all deformation history at the current state.
        gamma_n_hat_max_val_prior (float): Maximum of the applied segment stretch through all deformation history at the prior state.
        epsilon_n_diss_hat_val_prior (float): Nondimensional rate-independent dissipated segment scission energy at the prior state.
        kappa_n (float): Nondimensional segment stiffness.
        zeta_n_char (float): Nondimensional characteristic segment potential energy scale.
        gamma_n_pade_to_bergstrom_crit (float): Pade-to-Bergstrom critical segment stretch.
        gamma_n_crit (float): Critical segment stretch.
        gamma_pade_to_bergstrom_crit (float): Pade-to-Bergstrom critical absolute/equilibrium chain stretch.
        gamma_crit (float): Critical absolute/equilibrium chain stretch.
    
    Returns:
        npt.ArrayLike: Nondimensional rate-independent dissipated
        segment scission energy.
    
    """
    # Dissipated energy cannot be destroyed
    if (gamma_n_hat_val <= gamma_n_hat_val_prior or
        gamma_n_hat_val < gamma_n_hat_max_val):
        return epsilon_n_diss_hat_val_prior
    # Dissipated energy from fully broken segments remains fixed
    elif gamma_n_hat_max_val_prior > gamma_n_crit:
        return epsilon_n_diss_hat_val_prior
    else:
        # No dissipated energy at equilibrium
        if (gamma_n_hat_val-1.) < (gamma_n_hat_val-gamma_n_hat_val_prior):
            epsilon_n_diss_hat_prime_val = 0.
        else:
            # Dissipated energy is created with respect to the prior
            # value of maximum applied segment stretch
            if gamma_n_hat_val_prior < gamma_n_hat_max_val_prior:
                gamma_n_hat_val_prior = gamma_n_hat_max_val_prior
            # Dissipated energy plateaus at the critical segment
            # stretch
            if (gamma_n_hat_max_val_prior < gamma_n_crit and
                gamma_n_hat_val > gamma_n_crit):
                gamma_n_hat_val = gamma_n_crit
            
            # Dissipated energy is created with increasing applied
            # segment stretch
            p_n_sci_hat_val_prior = p_n_sci_hat_func(
                gamma_n_hat_val_prior, kappa_n, zeta_n_char,
                gamma_n_pade_to_bergstrom_crit, gamma_n_crit,
                gamma_pade_to_bergstrom_crit, gamma_crit)
            p_n_sci_hat_val = p_n_sci_hat_func(
                gamma_n_hat_val, kappa_n, zeta_n_char,
                gamma_n_pade_to_bergstrom_crit, gamma_n_crit,
                gamma_pade_to_bergstrom_crit, gamma_crit)
            p_n_sci_hat_prime_val = (
                (p_n_sci_hat_val-p_n_sci_hat_val_prior)
                / (gamma_n_hat_val-gamma_n_hat_val_prior)
            )
            epsilon_n_sci_hat_val = (
                epsilon_n_sci_hat_func(
                    gamma_n_hat_val, kappa_n, zeta_n_char,
                    gamma_n_pade_to_bergstrom_crit, gamma_n_crit,
                    gamma_pade_to_bergstrom_crit, gamma_crit)
            )
            epsilon_n_diss_hat_prime_val = (
                p_n_sci_hat_prime_val * epsilon_n_sci_hat_val 
            )
        
        # Calculate dissipated segment scission energy
        return (
            epsilon_n_diss_hat_val_prior + epsilon_n_diss_hat_prime_val
            * (gamma_n_hat_val-gamma_n_hat_val_prior)
        )

def epsilon_cn_diss_hat_rate_independent_scission_func(
        gamma_n_hat_val: float,
        gamma_n_hat_val_prior: float,
        gamma_n_hat_max_val: float,
        gamma_n_hat_max_val_prior: float,
        epsilon_cn_diss_hat_val_prior: float,
        n: float | int,
        kappa_n: float,
        zeta_n_char: float,
        gamma_n_pade_to_bergstrom_crit: float,
        gamma_n_crit: float,
        gamma_pade_to_bergstrom_crit: float,
        gamma_crit: float) -> float:
    """Nondimensional rate-independent dissipated chain scission energy
    per segment.
    
    This function computes the nondimensional rate-independent
    dissipated chain scission energy per segment at the current state.

    Args:
        gamma_n_hat_val (float): Applied segment stretch at the current state.
        gamma_n_hat_val_prior (float): Applied segment stretch at the prior state.
        gamma_n_hat_max_val (float): Maximum of the applied segment stretch through all deformation history at the current state.
        gamma_n_hat_max_val_prior (float): Maximum of the applied segment stretch through all deformation history at the prior state.
        epsilon_cn_diss_hat_val_prior (float): Nondimensional rate-independent dissipated chain scission energy per segment at the prior state.
        n (float | int): Number of chain segments.
        kappa_n (float): Nondimensional segment stiffness.
        zeta_n_char (float): Nondimensional characteristic segment potential energy scale.
        gamma_n_pade_to_bergstrom_crit (float): Pade-to-Bergstrom critical segment stretch.
        gamma_n_crit (float): Critical segment stretch.
        gamma_pade_to_bergstrom_crit (float): Pade-to-Bergstrom critical absolute/equilibrium chain stretch.
        gamma_crit (float): Critical absolute/equilibrium chain stretch.
    
    Returns:
        npt.ArrayLike: Nondimensional rate-independent dissipated chain
        scission energy per segment.
    
    """
    # Dissipated energy cannot be destroyed
    if (gamma_n_hat_val <= gamma_n_hat_val_prior or
        gamma_n_hat_val < gamma_n_hat_max_val):
        return epsilon_cn_diss_hat_val_prior
    # Dissipated energy from fully broken segments remains fixed
    elif gamma_n_hat_max_val_prior > gamma_n_crit:
        return epsilon_cn_diss_hat_val_prior
    else:
        # No dissipated energy at equilibrium
        if (gamma_n_hat_val-1.) < (gamma_n_hat_val-gamma_n_hat_val_prior):
            epsilon_cn_diss_hat_prime_val = 0.
        else:
            # Dissipated energy is created with respect to the prior
            # value of maximum applied segment stretch
            if gamma_n_hat_val_prior < gamma_n_hat_max_val_prior:
                gamma_n_hat_val_prior = gamma_n_hat_max_val_prior
            # Dissipated energy plateaus at the critical segment
            # stretch
            if (gamma_n_hat_max_val_prior < gamma_n_crit and
                gamma_n_hat_val > gamma_n_crit):
                gamma_n_hat_val = gamma_n_crit
            
            # Dissipated energy is created with increasing applied
            # segment stretch
            p_c_sci_hat_val_prior = p_c_sci_hat_func(
                gamma_n_hat_val_prior, n, kappa_n, zeta_n_char,
                gamma_n_pade_to_bergstrom_crit, gamma_n_crit,
                gamma_pade_to_bergstrom_crit, gamma_crit)
            p_c_sci_hat_val = p_c_sci_hat_func(
                gamma_n_hat_val, n, kappa_n, zeta_n_char,
                gamma_n_pade_to_bergstrom_crit, gamma_n_crit,
                gamma_pade_to_bergstrom_crit, gamma_crit)
            p_c_sci_hat_prime_val = (
                (p_c_sci_hat_val-p_c_sci_hat_val_prior)
                / (gamma_n_hat_val-gamma_n_hat_val_prior)
            )
            epsilon_cn_sci_hat_val = (
                epsilon_cn_sci_hat_func(
                    gamma_n_hat_val, kappa_n, zeta_n_char,
                    gamma_n_pade_to_bergstrom_crit, gamma_n_crit,
                    gamma_pade_to_bergstrom_crit, gamma_crit)
            )
            epsilon_cn_diss_hat_prime_val = (
                p_c_sci_hat_prime_val * epsilon_cn_sci_hat_val 
            )
        
        # Calculate dissipated chain scission energy per segment
        return (
            epsilon_cn_diss_hat_val_prior + epsilon_cn_diss_hat_prime_val
            * (gamma_n_hat_val-gamma_n_hat_val_prior)
        )

def epsilon_n_diss_hat_crit_rate_independent_scission_func(
        gamma_n_hat_inc: float,
        kappa_n: float,
        zeta_n_char: float,
        gamma_n_pade_to_bergstrom_crit: float,
        gamma_n_crit: float,
        gamma_pade_to_bergstrom_crit: float,
        gamma_crit: float) -> float:
    """Nondimensional rate-independent critical dissipated segment
    scission energy.
    
    This function computes the nondimensional rate-independent critical
    dissipated segment scission energy.

    Args:
        gamma_n_hat_inc (float): Applied segment stretch increment.
        kappa_n (float): Nondimensional segment stiffness.
        zeta_n_char (float): Nondimensional characteristic segment potential energy scale.
        gamma_n_pade_to_bergstrom_crit (float): Pade-to-Bergstrom critical segment stretch.
        gamma_n_crit (float): Critical segment stretch.
        gamma_pade_to_bergstrom_crit (float): Pade-to-Bergstrom critical absolute/equilibrium chain stretch.
        gamma_crit (float): Critical absolute/equilibrium chain stretch.
    
    Returns:
        float: Nondimensional rate-independent critical dissipated
        segment scission energy.
    
    """
    # Define the applied segment stretch values to calculate over
    gamma_n_hat_nm_steps = (
        int(np.around((gamma_n_crit-1.)/gamma_n_hat_inc)) + 1
    )
    gamma_n_hat = np.linspace(1., gamma_n_crit, gamma_n_hat_nm_steps)

    # Initialization
    gamma_n_hat_max_val = gamma_n_hat[0]
    gamma_n_hat_max_val_prior = gamma_n_hat[0]
    epsilon_n_diss_hat = np.empty_like(gamma_n_hat)
    epsilon_n_diss_hat[0] = 0.
    # Advance through applied segment stretch
    for indx in range(1, np.shape(gamma_n_hat)[0]):
        # Collect parameters
        gamma_n_hat_val_prior = gamma_n_hat[indx-1]
        gamma_n_hat_val = gamma_n_hat[indx]
        gamma_n_hat_max_val = np.max(
            np.asarray([gamma_n_hat_max_val, gamma_n_hat_val]))
        epsilon_n_diss_hat_val_prior = epsilon_n_diss_hat[indx-1]
        
        # Calculate dissipated segment scission energy
        epsilon_n_diss_hat[indx] = (
            epsilon_n_diss_hat_rate_independent_scission_func(
                gamma_n_hat_val, gamma_n_hat_val_prior, gamma_n_hat_max_val,
                gamma_n_hat_max_val_prior, epsilon_n_diss_hat_val_prior,
                kappa_n, zeta_n_char, gamma_n_pade_to_bergstrom_crit,
                gamma_n_crit, gamma_pade_to_bergstrom_crit,
                gamma_crit)
        )
        
        # Update the maximum of the applied segment stretch through all
        # deformation history at the prior state
        gamma_n_hat_max_val_prior = gamma_n_hat_max_val
    
    # Deduce critical dissipated segment scission energy
    return epsilon_n_diss_hat[-1]

def epsilon_cn_diss_hat_crit_rate_independent_scission_func(
        gamma_n_hat_inc: float,
        n: float | int,
        kappa_n: float,
        zeta_n_char: float,
        gamma_n_pade_to_bergstrom_crit: float,
        gamma_n_crit: float,
        gamma_pade_to_bergstrom_crit: float,
        gamma_crit: float) -> float:
    """Nondimensional rate-independent critical dissipated chain
    scission energy per segment.
    
    This function computes the nondimensional rate-independent critical
    dissipated chain scission energy per segment.

    Args:
        gamma_n_hat_inc (float): Applied segment stretch increment.
        n (float | int): Number of chain segments.
        kappa_n (float): Nondimensional segment stiffness.
        zeta_n_char (float): Nondimensional characteristic segment potential energy scale.
        gamma_n_pade_to_bergstrom_crit (float): Pade-to-Bergstrom critical segment stretch.
        gamma_n_crit (float): Critical segment stretch.
        gamma_pade_to_bergstrom_crit (float): Pade-to-Bergstrom critical absolute/equilibrium chain stretch.
        gamma_crit (float): Critical absolute/equilibrium chain stretch.
    
    Returns:
        float: Nondimensional rate-independent critical dissipated chain
        scission energy per segment.
    
    """
    # Define the applied segment stretch values to calculate over
    gamma_n_hat_nm_steps = (
        int(np.around((gamma_n_crit-1.)/gamma_n_hat_inc)) + 1
    )
    gamma_n_hat = np.linspace(1., gamma_n_crit, gamma_n_hat_nm_steps)

    # Initialization
    gamma_n_hat_max_val = gamma_n_hat[0]
    gamma_n_hat_max_val_prior = gamma_n_hat[0]
    epsilon_cn_diss_hat = np.empty_like(gamma_n_hat)
    epsilon_cn_diss_hat[0] = 0.
    # Advance through applied segment stretch
    for indx in range(1, np.shape(gamma_n_hat)[0]):
        # Collect parameters
        gamma_n_hat_val_prior = gamma_n_hat[indx-1]
        gamma_n_hat_val = gamma_n_hat[indx]
        gamma_n_hat_max_val = np.max(
            np.asarray([gamma_n_hat_max_val, gamma_n_hat_val]))
        epsilon_cn_diss_hat_val_prior = epsilon_cn_diss_hat[indx-1]
        
        # Calculate dissipated chain scission energy per segment
        epsilon_cn_diss_hat[indx] = (
            epsilon_cn_diss_hat_rate_independent_scission_func(
                gamma_n_hat_val, gamma_n_hat_val_prior, gamma_n_hat_max_val,
                gamma_n_hat_max_val_prior, epsilon_cn_diss_hat_val_prior, n,
                kappa_n, zeta_n_char, gamma_n_pade_to_bergstrom_crit,
                gamma_n_crit, gamma_pade_to_bergstrom_crit,
                gamma_crit)
        )
        
        # Update the maximum of the applied segment stretch through all
        # deformation history at the prior state
        gamma_n_hat_max_val_prior = gamma_n_hat_max_val
    
    # Deduce critical dissipated chain scission energy per segment
    return epsilon_cn_diss_hat[-1]

def A_n_func(
        points: npt.NDArray[np.floating],
        weights: npt.NDArray[np.floating],
        n: float | int,
        gamma_n_hat_inc: float,
        kappa_n: float,
        zeta_n_char: float,
        gamma_n_pade_to_bergstrom_crit: float,
        gamma_n_crit: float,
        gamma_pade_to_bergstrom_crit: float,
        gamma_crit: float) -> float:
    """Reference absolute/equilibrium chain stretch.
    
    This function computes the reference absolute/equilibrium chain
    stretch via numerical quadrature.

    Args:
        points (npt.NDArray[np.floating]): Sample points for Gauss-Legendre quadrature used for numerically integrating various moments of the initial intact chain configuration equilibrium probability density distribution.
        weights (npt.NDArray[np.floating]): Weights for each sample point for Gauss-Legendre quadrature used for numerically integrating various moments of the initial intact chain configuration equilibrium probability density distribution.
        n (float | int): Number of chain segments.
        gamma_n_hat_inc (float): Applied segment stretch increment (for the calculation of the nondimensional rate-independent critical dissipated segment scission energy).
        kappa_n (float): Nondimensional segment stiffness.
        zeta_n_char (float): Nondimensional characteristic segment potential energy scale.
        gamma_n_pade_to_bergstrom_crit (float): Pade-to-Bergstrom critical segment stretch.
        gamma_n_crit (float): Critical segment stretch.
        gamma_pade_to_bergstrom_crit (float): Pade-to-Bergstrom critical absolute/equilibrium chain stretch.
        gamma_crit (float): Critical absolute/equilibrium chain stretch.
    
    Returns:
        float: Reference absolute/equilibrium chain stretch.
    
    """
    def Jac_func(x_max: float, x_min: float) -> float:
        """Scalar Jacobian for a transformation between two
        one-dimensional coordinate spaces.
        
        This function computes the scalar Jacobian for a transformation
        between two one-dimensional coordinate spaces.

        Args:
            x_max (float): Maximum coordinate.
            x_min (float): Minimum coordinate.
        
        Returns:
            float: Scalar Jacobian.
        
        """
        return (x_max-x_min) / 2.
    
    def gamma_points_func(
            points: npt.ArrayLike,
            gamma_crit: float) -> npt.ArrayLike:
        """Absolute/Equilibrium chain stretch as a function of master
        space coordinate points.

        This function computes the absolute/equilibrium chain stretch as
        a function of master space coordinate points.

        Args:
            points (npt.ArrayLike): Master space coordinate points.
            gamma_crit (float): Critical absolute/equilibrium chain stretch.
        
        Returns:
            npt.ArrayLike: Absolute/Equilibrium chain stretch as a
            function of master space coordinate points.
        
        """
        Jac = Jac_func(gamma_crit, 0.)
        return Jac * (1.+points) + 0.
    
    # Sort points in ascending order (and correspondingly sort the
    # weights)
    sort_indcs = np.argsort(points)
    points = points[sort_indcs]
    weights = weights[sort_indcs]
    
    # Nondimensional rate-independent critical dissipated segment
    # scission energy
    epsilon_n_diss_hat_crit = (
        epsilon_n_diss_hat_crit_rate_independent_scission_func(
            gamma_n_hat_inc, kappa_n, zeta_n_char,
            gamma_n_pade_to_bergstrom_crit, gamma_n_crit,
            gamma_pade_to_bergstrom_crit, gamma_crit)
    )
    
    # Jacobian for the master space-to-equilibrium chain configuration
    # space transformation
    Jac = Jac_func(gamma_crit, 0.)
    
    # Absolute/Equilibrium chain stretches corresponding to the master
    # space points for the initial intact chain configuration
    gamma_0_points = gamma_points_func(points, gamma_crit)

    # Integrand involved in the intact equilibrium chain configuration
    # partition function integration
    Z_intact = p_boltzmann_func(
        w_c_func(
            gamma_0_points, n, kappa_n, zeta_n_char, gamma_n_crit,
            gamma_pade_to_bergstrom_crit, gamma_crit))
    
    # Integrands of the zeroth moment and second moment of the initial
    # intact chain configuration equilibrium probability density
    # distribution (without normalization)
    I_0_intrgrnd = Z_intact * gamma_0_points**2
    I_2_intrgrnd = Z_intact * gamma_0_points**4
    
    # Zeroth moment and second moment of the initial intact chain
    # configuration equilibrium probability density distribution
    # (without normalization)
    I_0 = np.sum(np.multiply(weights, I_0_intrgrnd)) * Jac
    I_2 = np.sum(np.multiply(weights, I_2_intrgrnd)) * Jac

    # Total configuration equilibrium partition function
    Z_eq_tot = (1.+n*p_boltzmann_func(epsilon_n_diss_hat_crit)) * I_0
    
    # Reference absolute/equilibrium chain stretch
    return np.sqrt(I_2/Z_eq_tot)

def Lambda_n_ref_func(
        points: npt.NDArray[np.floating],
        weights: npt.NDArray[np.floating],
        n: float | int,
        gamma_n_hat_inc: float,
        kappa_n: float,
        zeta_n_char: float,
        gamma_n_pade_to_bergstrom_crit: float,
        gamma_n_crit: float,
        gamma_pade_to_bergstrom_crit: float,
        gamma_crit: float) -> float:
    """Reference equilibrium segment stretch.
    
    This function computes the reference equilibrium segment stretch
    via numerical quadrature.

    Args:
        points (npt.NDArray[np.floating]): Sample points for Gauss-Legendre quadrature used for numerically integrating various moments of the initial intact chain configuration equilibrium probability density distribution.
        weights (npt.NDArray[np.floating]): Weights for each sample point for Gauss-Legendre quadrature used for numerically integrating various moments of the initial intact chain configuration equilibrium probability density distribution.
        n (float | int): Number of chain segments.
        gamma_n_hat_inc (float): Applied segment stretch increment (for the calculation of the nondimensional rate-independent critical dissipated segment scission energy).
        kappa_n (float): Nondimensional segment stiffness.
        zeta_n_char (float): Nondimensional characteristic segment potential energy scale.
        gamma_n_pade_to_bergstrom_crit (float): Pade-to-Bergstrom critical segment stretch.
        gamma_n_crit (float): Critical segment stretch.
        gamma_pade_to_bergstrom_crit (float): Pade-to-Bergstrom critical absolute/equilibrium chain stretch.
        gamma_crit (float): Critical absolute/equilibrium chain stretch.
    
    Returns:
        float: Reference equilibrium segment stretch.
    
    """
    # Reference absolute/equilibrium chain stretch
    A_n = A_n_func(
        points, weights, n, gamma_n_hat_inc, kappa_n, zeta_n_char,
        gamma_n_pade_to_bergstrom_crit, gamma_n_crit,
        gamma_pade_to_bergstrom_crit, gamma_crit)
    
    # Reference equilibrium segment stretch
    return (
        gamma_n_func(
            A_n, kappa_n, zeta_n_char, gamma_pade_to_bergstrom_crit,
            gamma_crit)
    )