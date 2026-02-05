import numpy as np
import numpy.typing as npt

def inv_langevin_func(x: npt.ArrayLike) -> npt.ArrayLike:
    """Jedynak R[9,2] inverse Langevin approximant.
    
    This function returns the Jedynak R[9,2] inverse Langevin
    approximant.

    Args:
        x (npt.ArrayLike): Argument of the Jedynak R[9,2] inverse Langevin approximant.
    
    Returns:
        npt.ArrayLike: Jedynak R[9,2] inverse Langevin approximant.
    
    """
    inv_langevin_nmrtr = (
        3. * x - 1.00651042 * x**3 - 0.96225019 * x**5 + 1.47352941 * x**7
        - 0.48953069 * x**9
    )
    inv_langevin_dnmntr = (1.-x) * (1.+1.01524033*x)
    return inv_langevin_nmrtr / inv_langevin_dnmntr

def inv_langevin_prime_func(x: npt.ArrayLike) -> npt.ArrayLike:
    """Derivative of the Jedynak R[9,2] inverse Langevin approximant.
    
    This function returns the derivative of the Jedynak R[9,2] inverse
    Langevin approximant.

    Args:
        x (npt.ArrayLike): Argument of the derivative of the Jedynak R[9,2] inverse Langevin approximant.
    
    Returns:
        npt.ArrayLike: Derivative of the Jedynak R[9,2] inverse Langevin
        approximant.
    
    """
    inv_langevin_nmrtr = (
        3. * x - 1.00651042 * x**3 - 0.96225019 * x**5 + 1.47352941 * x**7
        - 0.48953069 * x**9
    )
    inv_langevin_dnmntr = (1.-x) * (1.+1.01524033*x)
    inv_langevin_nmrtr_prime = (
        3. - 3. * 1.00651042 * x**2 - 5. * 0.96225019 * x**4
        + 7. * 1.47352941 * x**6 - 9. * 0.48953069 * x**8
    )
    inv_langevin_dnmntr_prime = (1.01524033-1.) - 2. * 1.01524033 * x
    inv_langevin_prime_nmrtr = (
        inv_langevin_nmrtr_prime * inv_langevin_dnmntr
        - inv_langevin_dnmntr_prime * inv_langevin_nmrtr
    )
    inv_langevin_prime_dnmntr = inv_langevin_dnmntr**2
    return inv_langevin_prime_nmrtr / inv_langevin_prime_dnmntr

def s_cn_inv_langevin_fjc_func(x: npt.ArrayLike) -> npt.ArrayLike:
    """Nondimensional chain-level entropic free energy contribution
    per segment as calculated by the Jedynak R[9,2] inverse Langevin
    approximate.
    
    This function returns the nondimensional chain-level entropic
    free energy contribution per segment as calculated by the
    Jedynak R[9,2] inverse Langevin approximate.

    Args:
        x (npt.ArrayLike): Argument to the nondimensional chain-level entropic free energy contribution per segment as calculated by the Jedynak R[9,2] inverse Langevin approximate.
    
    Returns:
        npt.ArrayLike: Nondimensional chain-level entropic free energy
        contribution per segment as calculated by the Jedynak R[9,2]
        inverse Langevin approximate.
    
    """
    return (
            0.0602726941412868 * x**8 + 0.00103401966455583 * x**7
            - 0.162726405850159 * x**6 - 0.00150537112388157 * x**5
            - 0.00350216312906114 * x**4 - 0.00254138511870934 * x**3
            + 0.488744117329956 * x**2 + 0.0071635921950366 * x
            - 0.999999503781195 * np.log(1.00000000002049-x)
            - 0.992044340231098 * np.log(x+0.98498877114821)
            - 0.0150047080499398
        )