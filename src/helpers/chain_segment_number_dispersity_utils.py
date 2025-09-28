import numpy as np

def unit_dirac_func(
        x: np.ndarray | float | int,
        x_0: float | int) -> np.ndarray | float:
    """Unit Dirac delta function.

    This function calculates the unit Dirac delta function, where the
    result is 1 if x = x_0, otherwise the result is 0.

    Args:
        x (np.ndarray | float | int): Arbitrary number.
        x_0 (float | int): Center shift.

    Returns:
        np.ndarray | float: Unit Dirac delta function.
    
    """
    return np.where(np.abs(x-x_0)<1e-10, 1.0, 0.0)

def p_n_uniform_func(
        n: np.ndarray | float | int,
        n_0: int) -> np.ndarray | float:
    """Uniform chain segment number probability distribution.

    This function calculates the probability of finding a chain with a
    given segment number in a uniform polymer network.

    Args:
        n (np.ndarray | float | int): Number of segments in the chain.
        n_0 (int): Uniform number of segments in the chains.

    Returns:
        np.ndarray | float: Chain segment number uniform probability
        distribution.
    
    """
    return unit_dirac_func(n, n_0)

def p_n_bimodal_func(
        n: np.ndarray | float | int,
        p_min: float,
        n_min: int,
        n_max: int) -> np.ndarray | float:
    """Bimodal chain segment number probability distribution.
    
    This function calculates the probability of finding a chain with a
    given segment number in a (sharply-)bimodal polymer network.

    Args:
        n (np.ndarray | float | int): Number of segments in the chain.
        p_min (float): Probability that a chain is composed of n_min segments (and p-1 is the probability that a chain is composed of n_max segments); p in (0, 1).
        n_min (int): Minimum number of segments that chains in the bimodal distribution can adopt.
        n_max (int): Maximum number of segments that chains in the bimodal distribution can adopt.
    
    Note: n_min < n_max must hold true.

    Returns:
        np.ndarray | float: Chain segment number bimodal probability
        distribution.
    
    """
    return (
        p_min * unit_dirac_func(n, n_min)
        + (1-p_min) * unit_dirac_func(n, n_max)
    )

def p_n_trimodal_func(
        n: np.ndarray | float | int,
        p_min: float,
        p_mid: float,
        n_min: int,
        n_mid: int,
        n_max: int) -> np.ndarray | float:
    """Trimodal chain segment number probability distribution.
    
    This function calculates the probability of finding a chain with a
    given segment number in a (sharply-)trimodal polymer network.

    Args:
        n (np.ndarray | float | int): Number of segments in the chain.
        p_min (float): Probability that a chain is composed of n_min segments; p_min in (0, 1).
        p_mid (float): Probability that a chain is composed of n_mid segments; p_mid in (0, 1).
        n_min (int): Minimum number of segments that chains in the trimodal distribution can adopt.
        n_mid (int): Middle number of segments that chains in the trimodal distribution can adopt.
        n_max (int): Maximum number of segments that chains in the trimodal distribution can adopt.
    
    Note: n_min < n_mid < n_max and p_min + p_mid < 1 must each hold
    true.

    Returns:
        np.ndarray | float: Chain segment number trimodal probability
        distribution.
    
    """
    return (
        p_min * unit_dirac_func(n, n_min)
        + p_mid * unit_dirac_func(n, n_mid)
        + (1-p_min-p_mid) * unit_dirac_func(n, n_max)
    )

def p_n_flory_func(
        n: np.ndarray | float | int,
        n_mean: float) -> np.ndarray | float:
    """Chain segment number probability distribution representative of
    random step-growth linear chain polymerization, as per the theory
    from Flory.

    This function calculates the probability of finding a chain with a
    given segment number in a polymer network formed via random
    step-growth linear chain polymerization with a given mean segment
    number.

    Args:
        n (np.ndarray | float | int): Number of segments in the chain.
        n_mean (float): Average number of segments in the polymer network.

    Returns:
        np.ndarray | float: Chain segment number probability
        distribution representative of random step-growth linear chain
        polymerization.
    
    """
    return (1./n_mean)*(1.-(1./n_mean))**(n-1)

def p_n_maxwell_boltzmann_func(
        n: np.ndarray | float | int,
        n_mean: float) -> np.ndarray | float:
    """Maxwell-Boltzmann chain segment number probability distribution.

    This function calculates the probability of finding a chain with a
    given segment number in a polymer network with chain segment number
    dispersity characterized by the Maxwell-Boltzmann distribution.

    Args:
        n (np.ndarray | float | int): Number of segments in the chain.
        n_mean (float): Average number of segments in the polymer network.

    Returns:
        np.ndarray | float: Chain segment number Maxwell-Boltzmann
        probability distribution.
    
    """
    return 32 * n**2 / (np.pi**2*n_mean**3) * np.exp(-4/np.pi*(n/n_mean)**2)

def p_n_gamma_func(
        n: np.ndarray | float | int,
        alpha: float,
        beta: float) -> np.ndarray | float:
    """Gamma chain segment number probability distribution.

    This function calculates the probability of finding a chain with a
    given segment number in a polymer network with chain segment number
    dispersity characterized by the Gamma distribution.

    Args:
        n (np.ndarray | float | int): Number of segments in the chain.
        alpha (float): Shape parameter; alpha>0.
        beta (float): Shape parameter; beta>0.

    Returns:
        np.ndarray | float: Chain segment number Gamma probability
        distribution.
    
    """
    return n**alpha * np.exp(-beta*n)

def p_n_weibull_func(
        n: np.ndarray | float | int,
        m: float,
        n_0: float) -> np.ndarray | float:
    """Weibull chain segment number probability distribution.

    This function calculates the probability of finding a chain with a
    given segment number in a polymer network with chain segment number
    dispersity characterized by the Weibull distribution.

    Args:
        n (np.ndarray | float | int): Number of segments in the chain.
        m (float): Shape parameter; m>0.
        n_0 (float): Scale parameter; n_0>0.

    Returns:
        np.ndarray | float: Chain segment number Weibull probability
        distribution.
    
    """
    return m / n_0 * (n/n_0)**(m-1) * np.exp(-(n/n_0)**m)

def p_n_rayleigh_func(
        n: np.ndarray | float | int,
        n_0: float) -> np.ndarray | float:
    """Rayleigh chain segment number probability distribution.

    This function calculates the probability of finding a chain with a
    given segment number in a polymer network with chain segment number
    dispersity characterized by the Rayleigh distribution.

    Args:
        n (np.ndarray | float | int): Number of segments in the chain.
        n_0 (float): Scale parameter; n_0>0.

    Returns:
        np.ndarray | float: Chain segment number Rayleigh probability
        distribution.
    
    """
    return p_n_weibull_func(n, 2.0, n_0)

def p_n_log_normal_func(
        n: np.ndarray | float | int,
        n_mean: float,
        sigma: float) -> np.ndarray | float:
    """Log-normal chain segment number probability distribution.

    This function calculates the probability of finding a chain with a
    given segment number in a polymer network with chain segment number
    dispersity characterized by the log-normal distribution.

    Args:
        n (np.ndarray | float | int): Number of segments in the chain.
        n_mean (float): Average number of segments in the polymer network.
        sigma (float): Standard deviation of segments in the polymer network.

    Returns:
        np.ndarray | float: Chain segment number log-normal probability
        distribution.
    
    """
    return (
        np.exp(-(np.log(n)-np.log(n_mean))**2/(2*sigma**2))
        / (n*np.sqrt(2*np.pi)*sigma)
    )

def p_n_inv_gaussian_func(
        n: np.ndarray | float | int,
        n_mean: float,
        lmbda: float) -> np.ndarray | float:
    """Inverse-Gaussian chain segment number probability distribution.

    This function calculates the probability of finding a chain with a
    given segment number in a polymer network with chain segment number
    dispersity characterized by the inverse-Gaussian distribution.

    Args:
        n (np.ndarray | float | int): Number of segments in the chain.
        lmbda (float): Shape parameter; lmbda>0.
        n_mean (float): Average number of segments in the polymer network.

    Returns:
        np.ndarray | float: Chain segment number inverse-Gaussian
        probability distribution.
    
    """
    return (
        np.sqrt(lmbda/(2*np.pi*n**3))
        * np.exp(-lmbda*(n-n_mean)**2/(2*n*n_mean**2))
    )

def p_n_init_func(
        n: np.ndarray,
        p_n_dist: str,
        p_n_func,
        p_n_p_args: tuple[float],
        p_n_n_args: tuple[int]) -> np.ndarray:
    """Polymer chain segment number probability distribution
    initialization.

    This function initializes an array repesenting the chain segment
    number probability distribution.

    Args:
        n (np.ndarray): 1D np.ndarray of N int entries of polymer chain segment numbers.
        p_n_dist (str): Short-hand name for the selected polymer chain segment number probability distribution function.
        p_n_func (function): The polymer chain segment number probability distribution function.
        p_n_p_args (tuple[float]): Probability-related arguments packaged in a float tuple for the polymer chain segment number probability distribution function.
        p_n_n_args (tuple[int]): Chain segment number-related argments packaged in an int tuple for the polymer chain segment number probability distribution function.
    
    Returns:
        np.ndarray: 1D np.ndarray of N float entries of the polymer
        chain segment number probability distribution.
    
    """
    # Number of salient chain segment numbers
    N = np.shape(n)[0]
    
    # Correct for the cases of the uniform, bimodal, and trimodal
    # polymer chain contour length distributions
    if N == 1 and p_n_dist != "uniform":
        error_str = (
            "Only one salient chain segment number is provided, but "
            + "p_n is not specified as uniform."
        )
        raise ValueError(error_str)
    elif N != 1 and p_n_dist == "uniform":
        error_str = (
            "p_n is specified as uniform, but the number of salient "
            + "chain segment number does not equal 1."
        )
        raise ValueError(error_str)
    elif N != 2 and p_n_dist == "bimodal":
        error_str = (
            "p_n is specified as bimodal, but the number of salient "
            + "chain segment number does not equal 2."
        )
        raise ValueError(error_str)
    elif N != 3 and p_n_dist == "trimodal":
        error_str = (
            "p_n is specified as trimodal, but the number of salient "
            + "chain segment number does not equal 3."
        )
        raise ValueError(error_str)
    
    # Initialize and return polymer chain segment number probability
    # distribution array
    return p_n_func(n, *p_n_p_args, *p_n_n_args)

def master_p_n_func(p_n_dist: str):
    """Master polymer chain segment number probability distribution
    function.

    This function returns the selected polymer chain segment number
    probability distribution function.

    Args:
        p_n_dist (str): Short-hand name for the selected polymer chain segment number probability distribution function.
    
    Returns:
        function: The selected polymer chain segment number probability
        distribution function.
    
    """
    if p_n_dist == "uniform": return p_n_uniform_func
    elif p_n_dist == "bimodal": return p_n_bimodal_func
    elif p_n_dist == "trimodal": return p_n_trimodal_func
    elif p_n_dist == "flory": return p_n_flory_func
    elif p_n_dist == "maxwell_boltzmann": return p_n_maxwell_boltzmann_func
    elif p_n_dist == "gamma": return p_n_gamma_func
    elif p_n_dist == "weibull": return p_n_weibull_func
    elif p_n_dist == "rayleigh": return p_n_rayleigh_func
    elif p_n_dist == "log_normal": return p_n_log_normal_func
    elif p_n_dist == "inv_gaussian": return p_n_inv_gaussian_func
    else:
        error_str = (
            "The called-for polymer chain segment number probability "
            + "distribution function is not implemented!"
        )
        raise NotImplementedError(error_str)