import numpy as np

def s_cn_inv_langevin_fjc_func(x: np.ndarray | float) -> np.ndarray | float:
    """Nondimensional chain-level entropic free energy contribution
    per segment as calculated by the Jedynak R[9,2] inverse Langevin
    approximate.
        
    This function computes the nondimensional chain-level entropic
    free energy contribution per segment as calculated by the
    Jedynak R[9,2] inverse Langevin approximate as a function of the
    result of the equilibrium chain stretch minus the segment
    stretch plus one.
    
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