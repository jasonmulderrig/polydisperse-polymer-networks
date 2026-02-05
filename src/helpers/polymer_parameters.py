import numpy as np
from scipy import constants
from pylimer_tools.io.bead_spring_parameter_provider import (
    ParameterType,
    get_parameters_for_polymer,
    _get_relevant_everaers_row
)

def get_bond_stretching_stiffness(bond_comp: str) -> float:
    """Bond stretching stiffness.

    This function returns the stretching stiffness of a specified bond,
    in units of kJ/(mol*nm^2). The citation reference for each bond
    stretching stiffness value is provided in comments. Note that the
    bond stretching stiffness values here correspond with a harmonic
    bond stretching function with a leading 0.5 factor, of the form
    U_b(l_b) = 0.5 * k_b * (l_b-l_b^eq).

    Args:
        bond_comp (str): Bond name.
    
    Returns:
        float: Bond stretching stiffness (in units of kJ/(mol*nm^2)).

    """
    if bond_comp == "C-C":
        # Sorensen et al., Macromolecules, 1988, converted from
        # originally provided units of J/(molecule*Angstroms^2)
        k_b = 2.5835e5 # kJ/(mol*nm^2)
    elif bond_comp == "Si-O":
        # Smith et al., J. Phys. Chem. B, 2004, converted from
        # originally provided units of kcal/(mol*Angstroms^2)
        k_b =  2.9298e5 # kJ/(mol*nm^2)
    else:
        error_str = (
            "The bond stretching stiffness value associated with the "
            + "specified bond has not yet been implemented."
        )
        raise ValueError(error_str)
    return k_b # kJ/(mol*nm^2)

def get_bond_dissociation_energy(bond_comp: str) -> float:
    """Bond dissociation energy.

    This function returns the dissociation energy of a specified bond,
    in units of kJ/mol. The citation reference for each bond
    dissociation energy value is provided in comments.

    Args:
        bond_comp (str): Bond name.
    
    Returns:
        float: Bond dissociation energy (in units of kJ/mol).

    """
    if bond_comp == "C-C":
        # CRC Handbook of Chemistry and Physics for CH_3-C_2H_5 citing
        # Luo, Y.R., Comprehensive Handbook of Chemical Bond Energies,
        # CRC Press, 2007
        E_b_char = 370.3 # kJ/mol
    elif bond_comp == "Si-O":
        # Schwaderer et al., Langmuir, 2008, citing Holleman and
        # Wilberg et al., Inorganic Chemistry, 2001
        E_b_char = 444.0 # kJ/mol
    else:
        error_str = (
            "The dissociation energy value associated with the "
            + "specified bond has not yet been implemented."
        )
        raise ValueError(error_str)
    return E_b_char # kJ/mol

def get_bond_energy_parameters(bond_comp: str) -> tuple[float, float]:
    """Bond energy parameters.

    This function returns both the stretching stiffness and dissociation
    energy of a specified bond, in units of kJ/(mol*nm^2) and kJ/mol,
    respectively.

    Args:
        bond_comp (str): Bond name.
    
    Returns:
        tuple[float, float]: Bond stretching stiffness and bond
        dissociation energy (in units of kJ/(mol*nm^2) and kJ/mol,
        respectively).

    """
    return (
        get_bond_stretching_stiffness(bond_comp),
        get_bond_dissociation_energy(bond_comp)
    ) # kJ/(mol*nm^2), kJ/mol

def get_equilibrium_bond_length(bond_comp: str) -> float:
    """Equilibrium bond length.

    This function returns the equilibrium length of a specified bond, in
    units of nm. The citation reference for each equilibrium bond length
    value is provided in comments.

    Args:
        bond_comp (str): Bond name.
    
    Returns:
        float: Equilibrium bond length (in units of nm).

    """
    if bond_comp == "C-C":
        # Allen et al., J. Chem. Soc. Perkin Trans (1987), converted
        # from originally provided units of Angstroms
        l_b_eq = 0.1530 # nm
    elif bond_comp == "Si-O":
        # Allen et al., J. Chem. Soc. Perkin Trans (1987), converted
        # from originally provided units of Angstroms
        l_b_eq = 0.1645 # nm
    else:
        error_str = (
            "The equilibrium bond length value associated with the "
            + "specified bond has not yet been implemented."
        )
        raise ValueError(error_str)
    return l_b_eq # nm

def get_equilibrium_bond_angle(bond_angle_comp: str) -> float:
    """Equilibrium bond angle.

    This function returns the equilibrium angle of two bonds that meet
    at a particular center atom, in units of degrees. The citation
    reference for each equilibrium bond angle value is provided in
    comments. Note that the angle being reported here is the angle
    between two bonds that meet at a particular center atom. 

    Args:
        bond_angle_comp (str): Bond angle triplet name.
    
    Returns:
        float: Equilibrium bond angle (in units of degrees).

    """
    if bond_angle_comp == "C-C-C":
        # Sorensen et al., Macromolecules, 1988, converted from
        # originally provided units of radians
        theta_b_eq = 111.00 # degrees
    elif bond_angle_comp == "Si-O-Si":
        # Smith et al., J. Phys. Chem. B, 2004
        theta_b_eq = 137.63 # degrees
    elif bond_angle_comp == "O-Si-O":
        # Smith et al., J. Phys. Chem. B, 2004
        theta_b_eq = 105.56 # degrees
    else:
        error_str = (
            "The equilibrium bond angle value associated with the "
            + "specified bond angle triplet has not yet been "
            + "implemented."
        )
        raise ValueError(error_str)
    return theta_b_eq # degrees

def get_monomer_energy_parameters(polymer_comp: str) -> tuple[float, float]:
    """Monomer energy parameters.

    This function returns both the stretching stiffness and
    characteristic potential energy scale of a specified monomer, in
    units of kJ/(mol*nm^2) and kJ/mol, respectively.

    Args:
        polymer_comp (str): Polymer name.
    
    Returns:
        tuple[float, float]: Monomer stretching stiffness and monomer
        characteristic potential energy scale (in units of kJ/(mol*nm^2)
        and kJ/mol, respectively).

    """
    if polymer_comp == "PDMS":
        # 1 PDMS monomer consists of 2 Si-O bonds
        n_b_over_n_m = 2
        bond_comp = "Si-O"
        # Bond stiffness and bond dissociation energy, i.e.,
        # characteristic bond potential energy scale
        k_b, E_b_char = get_bond_energy_parameters(bond_comp) # kJ/(mol*nm^2), kJ/mol
        # Use series spring model to calculate monomer stiffness and
        # characteristic monomer potential energy scale
        k_m = k_b / n_b_over_n_m # kJ/(mol*nm^2)
        E_m_char = E_b_char * n_b_over_n_m # kJ/mol
    else:
        error_str = (
            "The monomer energy parameters associated with the "
            + "specified polymer have not yet been implemented."
        )
        raise ValueError(error_str)
    return k_m, E_m_char # kJ/(mol*nm^2), kJ/mol

def get_kuhn_segment_energy_parameters(
        polymer_comp: str) -> tuple[float, float]:
    """Kuhn segment energy parameters.

    This function returns both the stretching stiffness and
    characteristic potential energy scale of a specified Kuhn segment,
    in units of kJ/(mol*nm^2) and kJ/mol, respectively.

    Args:
        polymer_comp (str): Polymer name.
    
    Returns:
        tuple[float, float]: Kuhn segment stretching stiffness and Kuhn
        segment characteristic potential energy scale (in units of
        kJ/(mol*nm^2) and kJ/mol, respectively).

    """
    if polymer_comp == "PDMS":
        # Number of monomers per Kuhn segment
        n_m_over_n = (
            _get_relevant_everaers_row(polymer_comp)["M_k_over_M_m"]
        )
        # Monomer stiffness and characteristic monomer potential energy
        # scale
        k_m, E_m_char = get_monomer_energy_parameters(polymer_comp) # kJ/(mol*nm^2), kJ/mol
        # Use series spring model to calculate Kuhn segment stiffness
        # and characteristic Kuhn segment potential energy scale
        k_n = k_m / n_m_over_n # kJ/(mol*nm^2)
        E_n_char = E_m_char * n_m_over_n # kJ/mol
    else:
        error_str = (
            "The Kuhn segment energy parameters associated with the "
            + "specified polymer have not yet been implemented."
        )
        raise ValueError(error_str)
    return k_n, E_n_char # kJ/(mol*nm^2), kJ/mol

def bond_angle_chain_end_to_end_axis(theta_b_eq: float, radians: bool) -> float:
    """Bond angle with respect to the chain end-to-end axis.

    This function returns the bond angle with respect to the chain
    end-to-end axis as calculated from a chemical bond angle.

    Args:
        theta_b_eq (float): Chemical bond angle, in units of degrees.
        radians (bool): Boolean indicating if the bond angle with respect to the chain end-to-end axis ought to have units of radians (if True) or units of degrees (if False).
    
    Returns:
        float: Bond angle with respect to the chain end-to-end axis.

    """
    theta_b_eq_chn_end_to_end_axis = 0.5 * (180-theta_b_eq)
    if radians: theta_b_eq_chn_end_to_end_axis *= (np.pi/180)
    return float(theta_b_eq_chn_end_to_end_axis)

def get_equilibrium_monomer_length(polymer_comp: str) -> float:
    """Equilibrium monomer length.

    This function returns the equilibrium length of a specified monomer,
    in units of nm, with respect to the chain end-to-end axis.

    Args:
        polymer_comp (str): Polymer name.
    
    Returns:
        float: Equilibrium monomer length (in units of nm).

    """
    if polymer_comp == "PDMS":
        # 1 PDMS monomer consists of 2 Si-O bonds, with an Si-O-Si bond
        # angle and an O-Si-O bond angle
        bond_comp = "Si-O"
        bond_angle_Si_O_Si = "Si-O-Si"
        bond_angle_O_Si_O = "O-Si-O"
        # Equilibrium bond length
        l_b_eq = get_equilibrium_bond_length(bond_comp) # nm
        # Equilibrium bond angles
        theta_b_eq_Si_O_Si = get_equilibrium_bond_angle(bond_angle_Si_O_Si) # degrees
        theta_b_eq_O_Si_O = get_equilibrium_bond_angle(bond_angle_O_Si_O) # degrees
        # Equilibrium bond angles with respect to the chain end-to-end
        # axis
        theta_b_eq_Si_O_Si_chn_end_to_end_axis = (
            bond_angle_chain_end_to_end_axis(theta_b_eq_Si_O_Si, True)
        ) # radians
        theta_b_eq_O_Si_O_chn_end_to_end_axis = (
            bond_angle_chain_end_to_end_axis(theta_b_eq_O_Si_O, True)
        ) # radians
        # Equilibrium monomer length (with respect to the chain
        # end-to-end axis)
        l_m_eq = (
            l_b_eq
            * (np.cos(theta_b_eq_Si_O_Si_chn_end_to_end_axis)+np.cos(theta_b_eq_O_Si_O_chn_end_to_end_axis))
        ) # nm
    else:
        error_str = (
            "The equilibrium monomer length associated with the "
            + "specified polymer has not yet been implemented."
        )
        raise ValueError(error_str)
    return float(l_m_eq) # nm

def get_equilibrium_kuhn_segment_length(polymer_comp: str) -> float:
    """Equilibrium Kuhn segment length.

    This function returns the equilibrium length of a specified Kuhn
    segment, in units of nm, with respect to the chain end-to-end axis.

    Args:
        polymer_comp (str): Polymer name.
    
    Returns:
        float: Equilibrium Kuhn segment length (in units of nm).

    """
    if polymer_comp == "PDMS":
        # Number of monomers per Kuhn segment
        n_m_over_n = (
            _get_relevant_everaers_row(polymer_comp)["M_k_over_M_m"]
        )
        # Equilibrium monomer length
        l_m_eq = get_equilibrium_monomer_length(polymer_comp) # nm
        # Use series spring model to calculate equilibrium Kuhn segment
        # length
        l_n_eq = l_m_eq * n_m_over_n # nm
    else:
        error_str = (
            "The equilibrium Kuhn segment length associated with the "
            + "specified polymer has not yet been implemented."
        )
        raise ValueError(error_str)
    return l_n_eq # nm

def get_nondim_kuhn_segment_energy_parameters(
        polymer_comp: str,
        T: float) -> tuple[float, float]:
    """Nondimensional Kuhn segment energy parameters.

    This function returns both the nondimensional stretching stiffness
    and nondimensional characteristic potential energy scale of a
    specified Kuhn segment.

    Args:
        polymer_comp (str): Polymer name.
        T (float): Absolute temperature, in units of Kelvin.
    
    Returns:
        tuple[float, float]: Nondimensional Kuhn segment stretching
        stiffness and nondimensional Kuhn segment characteristic
        potential energy scale.

    """
    # Gather fundamental physics constants
    k_B  = constants.value(u'Boltzmann constant') # J/K
    N_A  = constants.value(u'Avogadro constant') # 1/mol
    beta = 1. / (k_B*T) # 1/J
    if polymer_comp == "PDMS":
        # Kuhn segment stiffness and characteristic Kuhn segment
        # potential energy scale
        k_n, E_n_char = get_kuhn_segment_energy_parameters(polymer_comp) # kJ/(mol*nm^2), kJ/mol
        # Equilibrium Kuhn segment length
        l_n_eq = get_equilibrium_kuhn_segment_length(polymer_comp) # nm
        # Nondimensionalize the Kuhn segment stiffness and
        # characteristic Kuhn segment potential energy scale
        kappa_n = k_n * l_n_eq**2 / N_A * 1000 * beta # kJ/(mol*nm^2) -> kJ/mol -> kJ -> J -> unitless
        zeta_n_char = E_n_char / N_A * 1000 * beta # kJ/mol -> kJ -> J -> unitless
    else:
        error_str = (
            "The nondimensional Kuhn segment energy parameters "
            + "associated with the specified polymer have not yet been "
            + "implemented."
        )
        raise ValueError(error_str)
    return kappa_n, zeta_n_char # unitless, unitless

def get_bead_density(polymer_comp: str) -> float:
    """Polymer density.

    This function returns the density of a polymer, in units of
    particle number/nm^3.

    Args:
        polymer_comp (str): Polymer name.
    
    Returns:
        float: Polymer density (in units of particle number/nm^3).

    """
    # Get polymer parameters and return polymer density
    params = get_parameters_for_polymer(
        polymer_comp, parameter_type=ParameterType.GAUSSIAN)
    return params.get_bead_density() # en/nm^3

def get_bead_length(polymer_comp: str) -> float:
    """Gaussian bead length.

    This function returns the Gaussian bead length of a polymer, in
    units of nm.

    Args:
        polymer_comp (str): Polymer name.
    
    Returns:
        float: Gaussian bead length (in units of nm).

    """
    # Get polymer parameters and return Gaussian bead length
    params = get_parameters_for_polymer(
        polymer_comp, parameter_type=ParameterType.GAUSSIAN)
    return params.get("<b>").to(params.get("distance_units")).magnitude # nm

def get_kuhn_segment_length(polymer_comp: str) -> float:
    """Kuhn segment length.

    This function returns the Kuhn segment length of a polymer, in units
    of nm.

    Args:
        polymer_comp (str): Polymer name.
    
    Returns:
        float: Kuhn segment length (in units of nm).

    """
    # Get polymer parameters and return Kuhn segment length
    params = get_parameters_for_polymer(
        polymer_comp, parameter_type=ParameterType.KUHN)
    return params.get("<b>").to(params.get("distance_units")).magnitude # nm

def M_to_n(polymer_comp: str, M: int | float) -> int:
    """Polymer chain molar mass-to-Kuhn segment number conversion.

    This function converts the molar mass of a polymer chain to Kuhn
    segment number.

    Args:
        polymer_comp (str): Polymer name.
        M (int | float): Polymer chain molar mass, in units of g/mol.
    
    Returns:
        int: Kuhn segment number (calculated via the ceiling integer
        function).

    """
    # Get polymer parameters
    params = get_parameters_for_polymer(
        polymer_comp, parameter_type=ParameterType.KUHN)
    # Extract Kuhn segment molecular weight
    M_n = params.get("Mw").magnitude # g/mol
    # Convert the polymer chain molar mass M to the Kuhn segment number
    # via the ceiling integer function
    return int(np.ceil(M/M_n))