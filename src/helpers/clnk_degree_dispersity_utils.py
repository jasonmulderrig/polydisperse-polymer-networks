import numpy as np
from scipy.special import binom

def k_init_func(f: int) -> np.ndarray:
    """Elastically-effective cross-link degree.

    This function gathers the degree of all elastically-effective
    cross-links that exist in a polymer network.

    Args:
        f (int): Maximum cross-linker degree/functionality.
    
    Returns:
        np.ndarray: Degree of all elastically-effective cross-links that
        exist in a polymer network.
    
    """
    return np.arange(3, f+1, dtype=int)

def Xi_miller_macosko_theory_end_linking_random_cross_linking_func(
        f: int,
        xi: float,
        chi: float) -> float:
    """Probability of an elastically-ineffective connection site in an
    end-linked/randomly cross-linked polymer network described by the
    Miller-Macosko theory.

    This function calculates the probability of an
    elastically-ineffective connection site in an end-linked/randomly
    cross-linked polymer network described by the Miller-Macosko theory.

    Args:
        f (int): Maximum cross-linker degree/functionality.
        xi (float): Chain-to-cross-link connection probability.
        chi (float): Stoichiometric imbalance between the number of cross-linker sites and the number of chain ends.

    Returns:
        float: Probability of an elastically-ineffective connection
        site.
    
    """
    prob_dnmntr = chi * xi**2
    if f == 3:
        if prob_dnmntr <= 1./2.:
            error_str = (
                "chi * xi**2 must be greater than 1/2 in order for the "
                + "Miller-Macosko theory for end-linking and random "
                + "cross-linking to be applicable for the case where "
                + "f = 3."
            )
            raise ValueError(error_str)
        else: return 1. / prob_dnmntr - 1.
    elif f == 4:
        if prob_dnmntr <= 1./3.:
            error_str = (
                "chi * xi**2 must be greater than 1/3 in order for the "
                + "Miller-Macosko theory for end-linking and random "
                + "cross-linking to be applicable for the case where "
                + "f = 4."
            )
            raise ValueError(error_str)
        else: return np.sqrt(1./prob_dnmntr-3./4.) - 1. / 2.
    else:
        error_str = (
            "The non-trivial root solution scheme for the "
            + "Miller-Macosko theory for end-linking and random "
            + "cross-linking where f >= 5 has not been implemented yet!"
        )
        raise NotImplementedError(error_str)

def n_mean_miller_macosko_cross_linking_vulcanization_func(xi: float) -> float:
    """Average number of chain segments in a cross-linked/vulcanized
    polymer network described by the Miller-Macosko theory.

    This function calculates the average number of chain segments in a
    cross-linked/vulcanized polymer network described by the
    Miller-Macosko theory.

    Args:
        xi (float): Chain-to-cross-link connection probability, i.e., the conversion.

    Returns:
        float: Average number of chain segments in a
        cross-linked/vulcanized polymer network described by the
        Miller-Macosko theory.
    
    """
    return 1. / (1.-xi)

def Xi_miller_macosko_theory_cross_linking_vulcanization_func(
        f: int,
        n_mean: float,
        varrho: float) -> float:
    """Probability of an elastically-ineffective connection site in a
    cross-linked/vulcanized polymer network described by the
    Miller-Macosko theory.

    This function calculates the probability of an
    elastically-ineffective connection site in a cross-linked/vulcanized
    polymer network described by the Miller-Macosko theory.

    Args:
        f (int): Maximum cross-linker degree/functionality.
        n_mean (float): Average number of chain segments/monomer number of the repeat units.
        varrho (float): Extent of cross-linking, i.e., the fraction of repeat units that are cross-linked.

    Returns:
        float: Probability of an elastically-ineffective connection
        site.
    
    """
    prob_dnmntr = varrho * (n_mean-1)
    if f == 4:
        if prob_dnmntr <= 1./2.:
            error_str = (
                "varrho * (n_mean-1) must be greater than 1/2 in order "
                + "for the Miller-Macosko theory for cross-linking and "
                + "vulcanization to be applicable for the case where "
                + "f = 4."
            )
            raise ValueError(error_str)
        else: return np.sqrt(1./prob_dnmntr+1./4.) - 1. / 2.
    else:
        error_str = (
            "The non-trivial root solution scheme for the "
            + "Miller-Macosko theory for cross-linking and "
            + "vulcanization where f >= 4 has not been implemented yet!"
        )
        raise NotImplementedError(error_str)

def p_k_f_init_func(
        Xi_func,
        f: int,
        Xi_args: tuple[float]) -> np.ndarray:
    """Probability distribution of elastically-effective cross-links
    with degree k.

    This function initializes an array repesenting the probability
    distribution of elastically-effective cross-links with degree k.

    Args:
        Xi_func (function): The function governing the probability of an elastically-ineffective connection site with respect to polymerization and cross-linking theory.
        f (int): Maximum cross-linker degree/functionality.
        Xi_args (tuple[float]): Arguments packaged in a float tuple for the function governing the probability of an elastically-ineffective connection site.
    
    Returns:
        np.ndarray: Probability distribution of elastically-effective
        cross-links with degree k.
    
    """
    Xi = Xi_func(f, *Xi_args)
    k = k_init_func(f)
    p_k_f = np.empty(f-2)
    for k_indx in range(f-2):
        p_k_f[k_indx] = (
            binom(f, k[k_indx]) * Xi**(f-k[k_indx]) * (1.-Xi)**k[k_indx]
        )
    return p_k_f

def master_Xi_func(
        polymer_net_plymrztn_theory: str,
        polymer_net_plymrztn_tchnq: str):
    """Master function governing the probability of an
    elastically-ineffective connection site with respect to
    polymerization and cross-linking theory.

    This function returns the selected function governing the
    probability of an elastically-ineffective connection site with
    respect to polymerization and cross-linking theory.

    Args:
        polymer_net_plymrztn_theory (str): Short-hand name for selected theory describing the polymer network polymerization and cross-linking.
        polymer_net_plymrztn_tchnq (str): Short-hand name for the selected polymer network polymerization and cross-linking technique.
    
    Returns:
        function: The selected function governing the probability of an
        elastically-ineffective connection site with respect to
        polymerization and cross-linking theory.
    
    """
    if polymer_net_plymrztn_theory == "mmt":
        if polymer_net_plymrztn_tchnq == "end_linking_random_cross_linking":
            return Xi_miller_macosko_theory_end_linking_random_cross_linking_func
        elif polymer_net_plymrztn_tchnq == "cross_linking_vulcanization":
            return Xi_miller_macosko_theory_cross_linking_vulcanization_func
        else:
            error_str = (
                "So far, the Miller-Macosko theory description of "
                + "polymer network polymerization has been implemented "
                + "for the techniques of end-linking/random "
                + "cross-linking and cross-linking/vulcanization."
            )
            raise NotImplementedError(error_str)
    else:
        error_str = (
            "So far, only the Miller-Macosko theory description of "
            + "polymer network polymerization has been implemented!"
        )
        raise NotImplementedError(error_str)