def em_arg_en_func(en: float, f: float, chi: float) -> float:
    """Number of chains.

    This function calculates the number of chains, given the number of
    cross-linkers, the maximum cross-linker degree/functionality, and
    the stoichiometric imbalance between the number of cross-linker
    sites and the number of chain ends.

    Args:
        en (float): Number of cross-linkers.
        f (float): Maximum cross-linker degree/functionality.
        chi (float): Stoichiometric imbalance between the number of cross-linker sites and the number of chain ends.

    Returns:
        float: Number of chains.
    
    """
    return en * f / (2*chi)

def em_arg_nu_func(en_nu: float, nu: float) -> float:
    """Number of chains.

    This function calculates the number of chains, given the number of
    chain segment particles and the (average) number of segment
    particles per chain.

    Args:
        en_nu (float): Number of chain segment particles.
        nu (float): (Average) Number of segment particles per chain.

    Returns:
        float: Number of chains.
    
    """
    return en_nu / nu

def nu_arg_em_func(en_nu: float, em: float) -> float:
    """(Average) Number of segment particles per chain

    This function calculates the (average) number of segment particles
    per chain, given the number of chain segment particles and the
    number of chains.

    Args:
        en_nu (float): Number of chain segment particles.
        em (float): Number of chains.

    Returns:
        float: (Average) Number of segments per chain.
    
    """
    return en_nu / em

def en_arg_em_func(chi: float, em: float, f: float) -> float:
    """Number of cross-linkers.

    This function calculates the number of cross-linkers, given the
    stoichiometric imbalance between the number of cross-linker sites
    and the number of chain ends, the number of chains, and the maximum
    cross-linker degree/functionality.

    Args:
        chi (float): Stoichiometric imbalance between the number of cross-linker sites and the number of chain ends.
        em (float): Number of chains.
        f (float): Maximum cross-linker degree/functionality.

    Returns:
        float: Number of cross-linkers.
    
    """
    return 2 * chi * em / f

def en_nu_arg_em_func(em: float, nu: float) -> float:
    """Number of chain segment particles.

    This function calculates the number of chain segment particles,
    given the number of chains and the (average) number of segment
    particles per chain.

    Args:
        em (float): Number of chains.
        nu (float): (Average) Number of segment particles per chain.

    Returns:
        float: Number of chain segment particles.
    
    """
    return em * nu

def en_arg_en_tot_func(en_tot: float, en_other: float) -> float:
    """Number of particles (chain segment particles or cross-linkers).

    This function calculates the number of particles (chain segment
    particles or cross-linkers), given the number of constituents and
    the number of the other type of particle (cross-linkers or chain
    segment particles, respectively).

    Args:
        en_tot (float): Number of constituents.
        en_other (float): Number of the other type of particles (cross-linkers or chain segment particles).

    Returns:
        float: Number of particles (chain segment particles or
        cross-linkers).
    
    """
    return en_tot - en_other

def en_arg_phi_func(phi: float, en_tot: float) -> float:
    """Number of particles (chain segment particles or cross-linkers).

    This function calculates the number of particles (chain segment
    particles or cross-linkers), given the particle number fraction and
    the number of constituents.

    Args:
        phi (float): Particle (chain segment or cross-linker) number fraction.
        en_tot (float): Number of constituents.

    Returns:
        float: Number of particles (chain segment particles or cross-linkers).
    
    """
    return phi * en_tot

def en_tot_arg_en_func(en_nu: float, en: float) -> float:
    """Number of constituents.

    This function calculates the number of constituents,
    given the number of chain segment particles and the number of
    cross-linkers.

    Args:
        en_nu (float): Number of chain segment particles.
        en (float): Number of cross-linkers.

    Returns:
        float: Number of constituents.
    
    """
    return en_nu + en

def en_tot_arg_phi_func(en: float, phi: float) -> float:
    """Number of constituents.
    
    This function calculates the number of constituents, given the
    number of particles (chain segment particles or cross-linkers) and
    its number fraction.

    Args:
        en (float): Number of particles (chain segments or cross-linkers)
        phi (float): Particle (chain segment or cross-linker) number fraction.

    Returns:
        float: Number of constituents.
    
    """
    return en / phi

def phi_arg_en_func(en: float, en_tot: float) -> float:
    """Particle (chain segment or cross-linker) number fraction.

    This function calculates the particle (chain segment or
    cross-linker) number fraction, given the number of particles (chain
    segment particless or cross-linkers) and the number of constituents.

    Args:
        en (float): Number of particles (chain segment particles or cross-linkers)
        en_tot (float): Number of constituents.

    Returns:
        float: Particle (chain segment or cross-linker) number fraction.
    
    """
    return en / en_tot

def phi_arg_phi_func(phi_other: float) -> float:
    """Particle (chain segment or cross-linker) number fraction.

    This function calculates the particle (chain segment or
    cross-linker) number fraction, given the other particle
    (cross-linker or chain segment, respectively) number fraction.

    Args:
        phi_other (float): Other particle (cross-linker or chain segment, respectively) number fraction.

    Returns:
        float: Particle (chain segment or cross-linker) number fraction.
    
    """
    return 1 - phi_other