import numpy as np
import numpy.typing as npt
from src.helpers.chain_segment_number_dispersity import (
    master_p_n_func,
    p_n_init_func
)
from src.helpers.chain_segment_number import n_init_func
from src.helpers.polymer_network import (
    em_arg_en_func,
    en_tot_arg_en_func
)
from src.helpers.simulation_box import (
    A_or_V_arg_rho_func,
    L_arg_A_or_V_func,
    mic_func
)
from src.descriptors.descriptors import core_pb_edge_id
from src.helpers.graph_analysis import (
    lexsorted_edges,
    sparse_A_arr_largest_connected_component,
    edges_and_edges_attr_to_sparse_A_arr_and_sparse_A_attr_arr,
    sparse_A_arr_and_sparse_A_attr_arr_to_edges_and_edges_attr
)
from src.helpers.polymer_parameters import (
    get_bead_density,
    get_bead_length
)
from pylimer_tools_cpp import (
    MCUniverseGenerator,
    MoleculeType,
    Universe
)
from pylimer_tools.calc.miller_macosko_theory import predict_p_from_w_sol

def crosslinked_polydisperse_end_linked_polymer_network_universe(
        polymer_comp: str,
        n_init: str,
        n: tuple[int],
        p_n_dist: str,
        p_n_p_args: tuple[float],
        p_n_n_args: tuple[int],
        en: int,
        f: int,
        chi: float,
        cnvrsn: float,
        cnvrsn_param: str,
        relax_clnks: bool) -> Universe:
    """Cross-linked polydisperse end-linked polymer network universe.

    This function generates a cross-linked polydisperse end-linked
    polymer network universe.

    Polymer chain segments are denoted as type 1, and crosslinkers are
    denoted as type 2.

    Args:
        polymer_comp (str): Polymer name.
        n_init (str): Short-hand description for the salient chain segment number initialization protocol; either "explicit" or "linspace".
        n (tuple[int]): Salient chain segment numbers, or information needed to properly initialize the salient chain segment numbers.
        p_n_dist (str): Short-hand name for the selected polymer chain segment number probability distribution function.
        p_n_p_args (tuple[float]): Probability-related arguments packaged in a float tuple for the polymer chain segment number probability distribution function.
        p_n_n_args (tuple[int]): Chain segment number-related argments packaged in an int tuple for the polymer chain segment number probability distribution function.
        en (int): Number of cross-linkers.
        f (int): Maximum cross-linker degree/functionality.
        chi (float): Stoichiometric imbalance between the number of cross-linker sites and the number of chain ends.
        cnvrsn (float): Conversion parameter; either the extent of polymerization, the gel/network fraction, or the soluble fraction.
        cnvrsn_param (str): String indicating the identity of the conversion parameter; either the extent of polymerization ("xi"), the gel/network fraction ("gel_frac"), or the soluble fraction ("sol_frac").
        relax_clnks (bool): Boolean indicating if the cross-link positions in the network ought to be in their relaxed state (if True) or in their as-polymerized unrelaxed state (if False).
    
    Returns:
        Universe: Cross-linked polydisperse end-linked polymer network
        universe.

    """
    # Create a default random number generator
    rng = np.random.default_rng()

    # Establish chemical cross-linkers as type 2
    clnkr_type = 2

    # Extract polymer density and bead length
    rho_en_tot = get_bead_density(polymer_comp) # en/nm^3
    b = get_bead_length(polymer_comp) # nm
    
    # Initialize the salient polymer chain segment numbers
    n, _ = n_init_func(n_init, n)

    # Acquire the specified polymer chain segment number probability
    # distribution function
    p_n_func = master_p_n_func(p_n_dist)
    
    # Initialize and normalize the polymer chain segment number
    # probability distribution array
    p_n = p_n_init_func(n, p_n_dist, p_n_func, p_n_p_args, p_n_n_args)
    p_n /= np.sum(p_n, dtype=float)
    
    # Number of polymer chains
    em = int(em_arg_en_func(1.*en, 1.*f, chi))

    # Polymer chain segment number in each chain
    n_chns = rng.choice(n, size=em, p=p_n)

    # Polymer chain segment particles in each chain, assuming that the
    # provided polymer chain segment number refers to that for precursor
    # chains prior to cross-linking
    nu_chns = n_chns + 1
    assert np.all(np.greater_equal(nu_chns, 1))

    # Number of polymer chain segment particles
    en_nu = np.sum(nu_chns)

    # Number of atoms/particles in the polymer network universe
    en_tot = int(en_tot_arg_en_func(en_nu, en))

    # Initialize three-dimensional polymer network universe box
    V = A_or_V_arg_rho_func(en_tot, rho_en_tot) # en/(en/nm^3) = nm^3
    L = L_arg_A_or_V_func(3, V) # nm, nm, nm
    assert np.allclose(L, L[0]*np.ones(3))
    generator = MCUniverseGenerator(*L) # nm, nm, nm
    generator.set_bead_distance(b) # nm

    # Add polymer chains and cross-linkers
    generator.add_strands(nr_of_strands=em, strand_lengths=nu_chns)
    generator.add_crosslinkers(
        en, crosslinker_functionality=f, crosslinker_type=clnkr_type
    )
    
    # Link polymer chains and cross-linkers as per the specified
    # conversion parameter
    if cnvrsn_param == "xi":
        generator.link_strands_to_conversion(crosslinker_conversion=cnvrsn)
    elif cnvrsn_param == "gel_frac":
        generator.link_strands_to_soluble_fraction(soluble_fraction=1.-cnvrsn)
    elif cnvrsn_param == "sol_frac":
        generator.link_strands_to_soluble_fraction(soluble_fraction=cnvrsn)
    else:
        error_str = (
            "The specified conversion parameter must be either the "
            + "extent of polymerization (i.e., cross-linker "
            + "conversion), ''xi'', the gel/network fraction, "
            + "''gel_frac'', or the soluble fraction, ''sol_frac''."
        )
        raise ValueError(error_str)

    # If called for, relax crosslink positions for better equilibration
    if relax_clnks: generator.relax_crosslinks()

    # Return polymer network universe
    return generator.get_universe()

def crosslinked_polydisperse_end_linked_phantom_polymer_network(
        polymer_comp: str,
        universe: Universe) -> tuple[float, float, npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.integer], npt.NDArray[np.integer], npt.NDArray[np.integer], npt.NDArray[np.floating]]:
    """Cross-linked polydisperse end-linked phantom polymer network.

    This function generates a cross-linked polydisperse end-linked
    phantom polymer network, and returns a set of parameters that
    captures the salient characteristics of this phantom polymer network
    representation.

    Polymer chain segments are denoted as type 1, and crosslinkers are
    denoted as type 2. The phantom polymer chains that remain fully in
    the simulation box are called core chains and are denoted as type 1
    (which necessarily includes all primary loop chains), and phantom
    polymer chains that cross the periodic boundary of the simulation
    box are called periodic boundary crossing chains and are denoted as
    type 2.

    Args:
        polymer_comp (str): Polymer name.
        universe (Universe): Cross-linked polydisperse end-linked polymer network universe.
    
    Returns:
        tuple[float, float, npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.integer], npt.NDArray[np.integer], npt.NDArray[np.integer], npt.NDArray[np.floating]]:
        Polymer density (in units of particle number/nm^3), bead/segment
        distance (in units of nm), 1D array with dim float entries of
        the simulation box side lengths (in units of nm), 2D array with
        (en_core_nodes, 3) float entries corresponding to the coordinate
        of each node in the phantom polymer network, 1D array with
        en_core_nodes int entries of the type of each node in the
        phantom polymer network, 2D array with (em_chns_with_clnks, 2)
        int entries corresponding to the phantom polymer chains/edges
        array, 1D array with em_chns_with_clnks int entries of the
        phantom polymer chain type, 1D array with em_chns_with_clnks
        float entries of the number of chain segments in each phantom
        polymer chain.

    """
    # Establish chemical cross-linkers as type 2
    clnkr_type = 2

    # Extract polymer density and bead length
    rho_en_tot = get_bead_density(polymer_comp) # en/nm^3
    b = get_bead_length(polymer_comp) # nm

    # Get the box bounding the polymer network universe
    L = universe.get_box().get_l() # nm, nm, nm
    assert np.allclose(L, L[0]*np.ones(3))

    # Get atoms/particles from the polymer network universe. Note that
    # the atom ids here are 1-indexed, not 0-indexed!
    atoms = universe.get_atoms()
    
    # Get polymer chains with cross-linkers from the polymer network
    # universe
    chns_with_clnks = universe.get_chains_with_crosslinker(clnkr_type)
    em_chns_with_clnks = len(chns_with_clnks)
    em = 0
    for indx in range(em_chns_with_clnks):
        if len(chns_with_clnks[indx]) >= 2: em += 1
    
    # Initialize polymer chain arrays
    conn_chns = np.empty((em, 2), dtype=int)
    conn_chns_type = np.empty(em, dtype=int)
    n_chns = np.empty(em)

    # Assess each polymer chain (with cross-linkers)
    chn_indx = 0
    for indx in range(em_chns_with_clnks):
        chn = chns_with_clnks[indx]
        if len(chn) >= 2:
            # Phantom network chain/edge array
            chn_ends = chn.get_strand_ends(
                crosslinker_type=clnkr_type, close_loop=True)
            conn_chns[chn_indx, 0] = chn_ends[0].get_id() # 1-indexed
            conn_chns[chn_indx, 1] = chn_ends[1].get_id() # 1-indexed

            # Chain/Edge type: core versus periodic boundary crossing
            if chn.get_strand_type() == MoleculeType.PRIMARY_LOOP:
                # Primary loop chains are always core chains/edges
                conn_chns_type[chn_indx] = 1
            else:
                # Assess if the non-primary loop chain crosses the
                # periodic boundary, or not
                chn_atoms = chn.get_atoms_lined_up(
                    crosslinker_type=clnkr_type, close_loop=True)
                chn_atom_coord = chn_atoms[0].get_coordinates()
                for chn_atom_indx in range(len(chn_atoms)-1):
                    chn_atom_0_coord = mic_func(
                        chn_atoms[chn_atom_indx].get_coordinates(), L)
                    chn_atom_1_coord = mic_func(
                        chn_atoms[chn_atom_indx+1].get_coordinates(), L)
                    chn_atom_1_pb_coord, _ = core_pb_edge_id(
                        chn_atom_0_coord, chn_atom_1_coord, L)
                    bond_vec = chn_atom_1_pb_coord - chn_atom_0_coord
                    chn_atom_coord += bond_vec

                # Check if the chain is in the core box, or not
                if np.all(np.logical_and(np.greater_equal(chn_atom_coord, np.zeros(3)), np.less(chn_atom_coord, L))):
                    conn_chns_type[chn_indx] = 1
                else: conn_chns_type[chn_indx] = 2

            # Number of chain segments
            n_chns[chn_indx] = chn.get_nr_of_bonds() * 1.

            chn_indx += 1
    
    # Correct the phantom network chain/edge array from being 1-indexed
    # to being 0-indexed
    conn_chns -= 1

    # Get node information
    core_nodes = np.unique(conn_chns)
    en_core_nodes = np.shape(core_nodes)[0]

    # Initialize node arrays
    coords = np.empty((en_core_nodes, 3))
    core_nodes_type = np.empty(en_core_nodes, dtype=int)
    for core_node_indx in range(en_core_nodes):
        atom = atoms[core_nodes[core_node_indx]]
        coords[core_node_indx] = mic_func(atom.get_coordinates(), L)
        core_nodes_type[core_node_indx] = atom.get_type()
    
    # Construct an array that returns the index for each core node
    # number
    core_nodes_indcs = -1 * np.ones(np.max(core_nodes)+1, dtype=int)
    core_nodes_indcs[core_nodes] = np.arange(en_core_nodes, dtype=int)

    # Update all original core node values with updated core node values
    # for each chain
    for chn_indx in range(em_chns_with_clnks):
        conn_chns[chn_indx, 0] = int(core_nodes_indcs[conn_chns[chn_indx, 0]])
        conn_chns[chn_indx, 1] = int(core_nodes_indcs[conn_chns[chn_indx, 1]])
    
    # Lexicographically sort the chains, chain types, and chain segment
    # numbers
    conn_chns, lexsort_indcs = lexsorted_edges(conn_chns)
    conn_chns_type = conn_chns_type[lexsort_indcs]
    n_chns = n_chns[lexsort_indcs]

    return (
        rho_en_tot, b, L, coords, core_nodes_type,
        conn_chns, conn_chns_type, n_chns
    )

def crosslinked_polydisperse_end_linked_phantom_polymer_network_gel(
        en: int,
        k_max: int,
        core_nodes_type: npt.NDArray[np.integer],
        conn_chns: npt.NDArray[np.integer],
        conn_chns_type: npt.NDArray[np.integer],
        n_chns: npt.NDArray[np.floating]) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.integer], npt.NDArray[np.floating], int, int, float]:
    """Cross-linked polydisperse end-linked phantom polymer network gel.

    This function extracts the gel component of a cross-linked
    polydisperse end-linked phantom polymer network.

    Polymer chain segments are denoted as type 1, and crosslinkers are
    denoted as type 2. The phantom polymer chains that remain fully in
    the simulation box are called core chains and are denoted as type 1
    (which necessarily includes all primary loop chains), and phantom
    polymer chains that cross the periodic boundary of the simulation
    box are called periodic boundary crossing chains and are denoted as
    type 2.

    Args:
        en (int): Number of nodes in the cross-linked polydisperse end-linked phantom polymer network.
        k_max (int): Maximum node degree/functionality.
        core_nodes_type: (npt.NDArray[np.integer]): 1D array with en int entries of the type of each node in the phantom polymer network.
        conn_chns: (npt.NDArray[np.integer]): 2D array with (em, 2) int entries corresponding to the original as-provided phantom polymer network chains/edges array.
        conn_chns_type: (npt.NDArray[np.integer]): 1D array with em int entries of the original as-provided phantom polymer network chain type.
        n_chns: (npt.NDArray[np.floating]): 1D array with em float entries of the number of chain segments in each original as-provided phantom polymer network chain.
    
    Returns:
        tuple[npt.NDArray[np.integer], npt.NDArray[np.integer], npt.NDArray[np.floating], int, int, float]:
        2D array with (em_gel, 2) int entries corresponding to the
        chains/edges array of the phantom polymer network gel, 1D array
        with em_gel int entries of the chain type of each chain in the
        phantom polymer network gel, 1D array with em_gel float entries
        of the number of chain segments in each chain of the phantom
        polymer network gel, the number of cross-linkers in the phantom
        polymer network gel, the number of polymer chains in the phantom
        polymer network gel, and the number of polymer chain segments in
        the phantom polymer network gel.

    """
    # Establish chemical cross-linkers as type 2
    clnkr_type = 2

    # Extract the cross-linked polydisperse end-linked phantom polymer
    # network gel via extracting the largest connected component of the
    # phantom network
    conn_chns_gel, conn_chns_type_gel, n_chns_gel = (
        sparse_A_arr_largest_connected_component(
            en, k_max, conn_chns, conn_chns_type, n_chns)
    )
    
    # Gather the number of cross-linkers, number of polymer chains, and
    # number of polymer chain segments in the cross-linked polydisperse
    # end-linked phantom polymer network gel
    en_gel = int(
        np.count_nonzero(core_nodes_type[np.unique(conn_chns_gel)]==clnkr_type))
    em_gel = np.shape(conn_chns_gel)[0]
    en_n_gel = np.sum(n_chns_gel)

    return (
        conn_chns_gel, conn_chns_type_gel, n_chns_gel, en_gel, em_gel, en_n_gel
    )

def yasuda_morita_procedure(
        A: npt.NDArray[np.integer]) -> tuple[npt.NDArray[np.integer], int, int, int]:
    """Yasuda-Morita procedure.
    
    This function applies the Yasuda-Morita procedure to yield the
    elastically-effective network satisfying the Scanlan-Case criteria.

    Args:
        A (npt.NDArray[np.integer]): 2D array with (en, en) int entries of the adjacency matrix (with no multiedges).
    
    Returns:
        tuple[npt.NDArray[np.integer], int, int, int]: 2D array with
        (en, en) int entries of the adjacency matrix of the
        elastically-effective network (with no multiedges), the number
        of bridge center nodes in the as-provided polymer network, the
        number of dangling nodes in the as-provided polymer network, and
        the number of primary loops in the as-provided polymer network.
    
    """
    # Initialize network feature counters
    num_bridge_center_nodes = 0
    num_dangling_nodes = 0
    num_primary_loops = 0

    # Gather nodes
    en = np.shape(A)[0]
    nodes = np.arange(en, dtype=int)

    # Yasuda-Morita procedure
    while True:
        # Initialize trackers
        bridge_center_node_elim = False
        dangling_node_elim = False
        loop_elim = False
        
        # Bridge center node elimination
        for center_node in range(en):
            center_node = int(center_node)
            # Edges excluding self-loops
            A_row = np.delete(A[center_node, :], center_node, axis=0)
            A_row_nodes = np.delete(nodes, center_node, axis=0)
            # Check if node is a bridge center node
            if np.sum(A_row) == 2:
                bridge_center_node_elim = True
                # Check if the bridge bridges the same two nodes, and
                # thus is actually a bridging loop
                if np.size(np.where(A_row == 2)[0]) > 0:
                    root_node = int(A_row_nodes[np.where(A_row == 2)[0][0]])
                    # Eliminate the bridge center node from the network
                    A[root_node, center_node] = 0
                    A[center_node, root_node] = 0
                    # Add loop to root node
                    A[root_node, root_node] += 2
                # Otherwise, the bridge bridges two distinct nodes
                else:
                    # Identify all nodes involved in the bridge
                    bridge_nodes = A_row_nodes[np.where(A_row == 1)[0]]
                    left_node = int(bridge_nodes[0])
                    right_node = int(bridge_nodes[1])
                    # Eliminate the bridge center node from the network
                    A[left_node, center_node] = 0
                    A[center_node, right_node] = 0
                    A[right_node, center_node] = 0
                    A[center_node, left_node] = 0
                    # Ensure bridge remains intact in the network
                    A[left_node, right_node] += 1
                    A[right_node, left_node] += 1
                # Count bridge center node network feature
                num_bridge_center_nodes += 1
                break
        if bridge_center_node_elim: continue
        else:
            
            # Dangling node elimination
            for dangling_node in range(en):
                dangling_node = int(dangling_node)
                # Edges excluding self-loops
                A_row = np.delete(A[dangling_node, :], dangling_node, axis=0)
                A_row_nodes = np.delete(nodes, dangling_node, axis=0)
                # Check if node is a dangling node
                if np.sum(A_row) == 1:
                    dangling_node_elim = True
                    # Identify the root node for the dangling node
                    root_node = int(A_row_nodes[np.where(A_row == 1)[0][0]])
                    # Eliminate the dangling node from the network
                    A[root_node, dangling_node] = 0
                    A[dangling_node, root_node] = 0
                    # Count dangling node network feature
                    num_dangling_nodes += 1
                    break
            if dangling_node_elim: continue
            else:
                
                # Loop elimination
                for node in range(en):
                    node = int(node)
                    if A[node, node] >= 2:
                        loop_elim = True
                        # Eliminate the loop from the network
                        A[node, node] -= 2
                        # Count primary loop network feature
                        num_primary_loops += 1
                        break
                if loop_elim: continue
                else: break # Yasuda-Morita procedure has finished

    return A, num_bridge_center_nodes, num_dangling_nodes, num_primary_loops

def elastically_effective_crosslinked_polydisperse_end_linked_phantom_polymer_network_gel(
        en: int,
        k_max: int,
        core_nodes_type: npt.NDArray[np.integer],
        conn_chns_gel: npt.NDArray[np.integer],
        conn_chns_type_gel: npt.NDArray[np.integer],
        n_chns_gel: npt.NDArray[np.floating]) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.integer], npt.NDArray[np.floating], npt.NDArray[np.floating], int, int, float, float, int, int, int, int]:
    """Elastically-effective cross-linked polydisperse end-linked
    phantom polymer network gel extraction procedure.

    This function extracts the elastically-effective component of a
    cross-linked polydisperse end-linked phantom polymer network gel.

    Polymer chain segments are denoted as type 1, and crosslinkers are
    denoted as type 2. The phantom polymer chains that remain fully in
    the simulation box are called core chains and are denoted as type 1
    (which necessarily includes all primary loop chains), and phantom
    polymer chains that cross the periodic boundary of the simulation
    box are called periodic boundary crossing chains and are denoted as
    type 2.

    Note that this extraction procedure operates upon two
    assumptions, as follows: (1) all chains that connect the same pair
    of nodes are each assumed to be of the same type, and (2) during
    bridging node consolidation, if at least one of the chains
    (connected to the bridging node) being consolidated is a type 2
    chain (it crosses the periodic boundary of the simulation box), then
    the resulting consolidated chain will be a type 2 chain.

    Args:
        en (int): Number of nodes in the cross-linked polydisperse end-linked phantom polymer network.
        k_max (int): Maximum node degree/functionality.
        core_nodes_type: (npt.NDArray[np.integer]): 1D array with en int entries of the type of each node in the phantom polymer network.
        conn_chns_gel: (npt.NDArray[np.integer]): 2D array with (em, 2) int entries corresponding to the as-provided phantom polymer network gel chains/edges array.
        conn_chns_type_gel: (npt.NDArray[np.integer]): 1D array with em int entries of the as-provided phantom polymer network gel chain type.
        n_chns_gel: (npt.NDArray[np.floating]): 1D array with em float entries of the number of chain segments in each as-provided phantom polymer network gel chain.
    
    Returns:
        tuple[npt.NDArray[np.integer], npt.NDArray[np.integer], npt.NDArray[np.floating], npt.NDArray[np.floating], int, int, float, float, int, int, int, int]:
        2D array with (em_gel_ee, 2) int entries corresponding to the
        chains/edges array of the elastically-effective phantom polymer
        network gel, 1D array with em_gel_ee int entries of the chain
        type of each chain in the elastically-effective phantom polymer
        network gel, 1D array with em_gel_ee float entries of the
        effective/equivalent number of chain segments in each chain of
        the elastically-effective phantom polymer network gel, 1D array
        with em_gel_ee float entries of the naive number of chain
        segments in each chain of the elastically-effective phantom
        polymer network gel, the number of cross-linkers in the
        elastically-effective phantom polymer network gel, the number of
        polymer chains in the elastically-effective phantom polymer
        network gel, the number of effective/equivalent polymer chain
        segments in the elastically-effective phantom polymer network
        gel, the total number of polymer chain segments in the
        elastically-effective phantom polymer network gel, the number of
        multichains in the as-provided phantom polymer network gel, the
        number of bridge center nodes in the as-provided phantom polymer
        network gel, the number of dangling nodes in the as-provided
        phantom polymer network gel, and the number of primary loops in
        the as-provided phantom polymer network gel.

    """
    # Establish chemical cross-linkers as type 2
    clnkr_type = 2

    # Initialize network feature counters
    num_multichains = 0
    num_bridge_center_nodes = 0
    num_dangling_nodes = 0
    num_primary_loops = 0
    
    # Gather symmetric sparse adjacency array and the sparse adjacency
    # chain segment number array representing the cross-linked
    # polydisperse end-linked phantom polymer network gel
    sparse_A_arr, sparse_A_n_chns_arr = (
        edges_and_edges_attr_to_sparse_A_arr_and_sparse_A_attr_arr(
            en, k_max, conn_chns_gel, binary_edges_type=conn_chns_type_gel,
            edges_attr=n_chns_gel, symmtry=True)
    )

    # Make a copy of the sparse adjacency chain segment number array.
    # The original array stores the effective/equivalent chain segment
    # numbers in the elastically-effective cross-linked network. The
    # copied array stores the naive chain segment numbers in the
    # elastically-effective cross-linked network, which will be
    # necessary for eventually calculating the total number of chain
    # segments in the elastically-effective cross-linked network.
    naive_sparse_A_n_chns_arr = sparse_A_n_chns_arr.copy()

    # Elastically-effective cross-linked polydisperse end-linked phantom
    # polymer network gel extraction procedure
    while True:
        # Initialize trackers
        multichain_elim = False
        bridge_center_node_elim = False
        dangling_node_elim = False
        primary_loop_elim = False

        # Multichain elimination
        for node in range(en):
            # Node is connected to at least one other node
            if np.count_nonzero(sparse_A_arr[node]) > 0:
                # Gather unique neighbor nodes and the number of
                # connections that exist with each unique neighbor node
                nghbr_nodes, chn_order = np.unique(
                    sparse_A_arr[node, np.nonzero(sparse_A_arr[node])[0]],
                    return_counts=True)
                # Avoid primary loop chains (which will be addressed
                # later)
                nghbr_nodes, chn_order = (
                    nghbr_nodes[nghbr_nodes!=node+1],
                    chn_order[nghbr_nodes!=node+1]
                )
                nghbr_nodes, chn_order = (
                    nghbr_nodes[nghbr_nodes!=-1*(node+1)],
                    chn_order[nghbr_nodes!=-1*(node+1)]
                )
                # Gather unique neighbor nodes with multiple connections
                nghbr_nodes, chn_order = (
                    nghbr_nodes[chn_order>1], chn_order[chn_order>1]
                )
                # At least one multichain exists
                if np.size(nghbr_nodes) > 0:
                    multichain_elim = True
                    for nghbr_node in np.nditer(nghbr_nodes):
                        nghbr_node = int(nghbr_node)
                        # Gather neighbor node indices
                        nghbr_node_k_indcs = (
                            np.where(sparse_A_arr[node] == nghbr_node)[0]
                        )
                        # Use the parallel spring model to calculate the
                        # effective/equivalent chain segment number in
                        # the effective/equivalent chain, noting that
                        # the chain segment number is a measure of
                        # spring compliance
                        n_chn = np.reciprocal(
                            np.sum(
                                np.reciprocal(
                                    sparse_A_n_chns_arr[node, nghbr_node_k_indcs]*1.)))
                        # Calculate naive chain segment number in the
                        # effective/equivalent chain
                        naive_n_chn = np.sum(
                            sparse_A_n_chns_arr[node, nghbr_node_k_indcs])
                        # Convert/Eliminate the multichain to a single
                        # effective/equivalent chain
                        sparse_A_n_chns_arr[node, nghbr_node_k_indcs[0]] = n_chn
                        naive_sparse_A_n_chns_arr[node, nghbr_node_k_indcs[0]] = (
                            naive_n_chn
                        )
                        sparse_A_arr[node, nghbr_node_k_indcs[1:]] = 0
                        sparse_A_n_chns_arr[node, nghbr_node_k_indcs[1:]] = 0.
                        naive_sparse_A_n_chns_arr[node, nghbr_node_k_indcs[1:]] = (
                            0.
                        )
                        # Address the symmetric duplicate multichain
                        if nghbr_node > 0:
                            symmtrc_node = nghbr_node - 1
                            symmtrc_nghbr_node = node + 1
                        else:
                            symmtrc_node = -1 * nghbr_node - 1
                            symmtrc_nghbr_node = -1 * (node+1)
                        # Gather symmetric neighbor node indices
                        symmtrc_nghbr_node_k_indcs = (
                            np.where(sparse_A_arr[symmtrc_node] == symmtrc_nghbr_node)[0]
                        )
                        # Convert/Eliminate the symmetric duplicate
                        # multichain to a single effective/equivalent
                        # chain
                        sparse_A_n_chns_arr[symmtrc_node, symmtrc_nghbr_node_k_indcs[0]] = (
                            n_chn
                        )
                        naive_sparse_A_n_chns_arr[symmtrc_node, symmtrc_nghbr_node_k_indcs[0]] = (
                            naive_n_chn
                        )
                        sparse_A_arr[symmtrc_node, symmtrc_nghbr_node_k_indcs[1:]] = (
                            0
                        )
                        sparse_A_n_chns_arr[symmtrc_node, symmtrc_nghbr_node_k_indcs[1:]] = (
                            0.
                        )
                        naive_sparse_A_n_chns_arr[symmtrc_node, symmtrc_nghbr_node_k_indcs[1:]] = (
                            0.
                        )
                        # Sort the symmetric duplicate node row of the
                        # sparse adjacency array and the sparse
                        # adjacency chain segment number arrays
                        sort_indcs = np.argsort(sparse_A_arr[symmtrc_node])
                        sparse_A_arr[symmtrc_node] = (
                            sparse_A_arr[symmtrc_node, sort_indcs]
                        )
                        sparse_A_n_chns_arr[symmtrc_node] = (
                            sparse_A_n_chns_arr[symmtrc_node, sort_indcs]
                        )
                        naive_sparse_A_n_chns_arr[symmtrc_node] = (
                            naive_sparse_A_n_chns_arr[symmtrc_node, sort_indcs]
                        )
                        # Count multichain network feature
                        num_multichains += 1
                    # Sort the node row of the sparse adjacency array
                    # and the sparse adjacency chain segment number
                    # arrays
                    sort_indcs = np.argsort(sparse_A_arr[node])
                    sparse_A_arr[node] = sparse_A_arr[node, sort_indcs]
                    sparse_A_n_chns_arr[node] = (
                        sparse_A_n_chns_arr[node, sort_indcs]
                    )
                    naive_sparse_A_n_chns_arr[node] = (
                        naive_sparse_A_n_chns_arr[node, sort_indcs]
                    )
                    break
        if multichain_elim: continue
        else:
            
            # Bridge center node elimination
            for center_node in range(en):
                # Node is connected to at least one other node
                if np.count_nonzero(sparse_A_arr[center_node]) > 0:
                    # Gather neighbor nodes
                    nghbr_nodes = (
                        sparse_A_arr[center_node, np.nonzero(sparse_A_arr[center_node])[0]]
                    )
                    # Avoid primary loop chains (which will be addressed
                    # later)
                    nghbr_nodes = nghbr_nodes[nghbr_nodes!=center_node+1]
                    nghbr_nodes = nghbr_nodes[nghbr_nodes!=-1*(center_node+1)]
                    # A node that is connected to 2 distinctly different
                    # nodes is a bridge center node. Since all
                    # multichains have been eliminated before, and
                    # primary loop chains have been avoided, then if a
                    # node here has 2 connection sites filled, those two
                    # connections are necessarily with two distinctly
                    # different nodes.
                    if np.size(nghbr_nodes) == 2:
                        # Identify the two neighbor nodes as the
                        # left node and right node, connected to the
                        # bridge center node
                        left_node = int(nghbr_nodes[0])
                        right_node = int(nghbr_nodes[1])
                        # Bridge center node exists
                        if left_node != right_node: # Just to be safe...
                            bridge_center_node_elim = True
                            # Gather neighbor node indices
                            left_node_k_indx = (
                                np.where(sparse_A_arr[center_node] == left_node)[0][0]
                            )
                            right_node_k_indx = (
                                np.where(sparse_A_arr[center_node] == right_node)[0][0]
                            )
                            nghbr_node_k_indcs = np.asarray(
                                [left_node_k_indx, right_node_k_indx],
                                dtype=int)
                            # Use the series spring model to calculate
                            # the effective/equivalent chain segment
                            # number in the effective/equivalent chain,
                            # noting that the chain segment number is a
                            # measure of spring compliance
                            n_chn = np.sum(
                                sparse_A_n_chns_arr[center_node, nghbr_node_k_indcs])
                            # Calculate naive chain segment number in
                            # the effective/equivalent chain
                            naive_n_chn = np.sum(
                                sparse_A_n_chns_arr[center_node, nghbr_node_k_indcs])
                            # Eliminate the bridge center node
                            sparse_A_arr[center_node, nghbr_node_k_indcs] = 0
                            sparse_A_n_chns_arr[center_node, nghbr_node_k_indcs] = (
                                0.
                            )
                            naive_sparse_A_n_chns_arr[center_node, nghbr_node_k_indcs] = (
                                0.
                            )
                            # Sort the bridge center node row of the
                            # sparse adjacency array and the sparse
                            # adjacency chain segment number arrays
                            sort_indcs = np.argsort(sparse_A_arr[center_node])
                            sparse_A_arr[center_node] = (
                                sparse_A_arr[center_node, sort_indcs]
                            )
                            sparse_A_n_chns_arr[center_node] = (
                                sparse_A_n_chns_arr[center_node, sort_indcs]
                            )
                            naive_sparse_A_n_chns_arr[center_node] = (
                                naive_sparse_A_n_chns_arr[center_node, sort_indcs]
                            )
                            # Consolidate the two bridge center series
                            # chains into one single
                            # effective/equivalent chain. Consolidate
                            # first from the left node to the right
                            # node, and sort the consolidated bridge
                            # center node rows of the sparse adjacency
                            # array and the sparse adjacency chain
                            # segment number arrays
                            if left_node > 0 and right_node > 0:
                                sparse_A_left_node = left_node - 1
                                sparse_A_center_node = center_node + 1
                                sparse_A_right_node = right_node
                            elif left_node > 0 and right_node < 0:
                                sparse_A_left_node = left_node - 1
                                sparse_A_center_node = center_node + 1
                                sparse_A_right_node = right_node
                            elif left_node < 0 and right_node > 0:
                                sparse_A_left_node = -1 * left_node - 1
                                sparse_A_center_node = -1 * (center_node+1)
                                sparse_A_right_node = -1 * right_node
                            else:
                                sparse_A_left_node = -1 * left_node - 1
                                sparse_A_center_node = -1 * (center_node+1)
                                sparse_A_right_node = right_node
                            center_node_k_indx = (
                                np.where(sparse_A_arr[sparse_A_left_node] == sparse_A_center_node)[0][0]
                            )
                            sparse_A_arr[sparse_A_left_node, center_node_k_indx] = (
                                sparse_A_right_node
                            )
                            sparse_A_n_chns_arr[sparse_A_left_node, center_node_k_indx] = (
                                n_chn
                            )
                            naive_sparse_A_n_chns_arr[sparse_A_left_node, center_node_k_indx] = (
                                naive_n_chn
                            )
                            sort_indcs = np.argsort(
                                sparse_A_arr[sparse_A_left_node])
                            sparse_A_arr[sparse_A_left_node] = (
                                sparse_A_arr[sparse_A_left_node, sort_indcs]
                            )
                            sparse_A_n_chns_arr[sparse_A_left_node] = (
                                sparse_A_n_chns_arr[sparse_A_left_node, sort_indcs]
                            )
                            naive_sparse_A_n_chns_arr[sparse_A_left_node] = (
                                naive_sparse_A_n_chns_arr[sparse_A_left_node, sort_indcs]
                            )
                            # In a symmetrically duplicate fashion,
                            # consolidate next from the right node to
                            # the left node, and sort the consolidated
                            # bridge center node rows of the sparse
                            # adjacency array and the sparse adjacency
                            # chain segment number arrays
                            if left_node > 0 and right_node > 0:
                                sparse_A_left_node = left_node
                                sparse_A_center_node = center_node + 1
                                sparse_A_right_node = right_node - 1
                            elif left_node > 0 and right_node < 0:
                                sparse_A_left_node = -1 * left_node
                                sparse_A_center_node = -1 * (center_node+1)
                                sparse_A_right_node = -1 * right_node - 1
                            elif left_node < 0 and right_node > 0:
                                sparse_A_left_node = left_node
                                sparse_A_center_node = center_node + 1
                                sparse_A_right_node = right_node - 1
                            else:
                                sparse_A_left_node = left_node
                                sparse_A_center_node = -1 * (center_node+1)
                                sparse_A_right_node = -1 * right_node - 1
                            center_node_k_indx = (
                                np.where(sparse_A_arr[sparse_A_right_node] == sparse_A_center_node)[0][0]
                            )
                            sparse_A_arr[sparse_A_right_node, center_node_k_indx] = (
                                sparse_A_left_node
                            )
                            sparse_A_n_chns_arr[sparse_A_right_node, center_node_k_indx] = (
                                n_chn
                            )
                            naive_sparse_A_n_chns_arr[sparse_A_right_node, center_node_k_indx] = (
                                naive_n_chn
                            )
                            sort_indcs = np.argsort(
                                sparse_A_arr[sparse_A_right_node])
                            sparse_A_arr[sparse_A_right_node] = (
                                sparse_A_arr[sparse_A_right_node, sort_indcs]
                            )
                            sparse_A_n_chns_arr[sparse_A_right_node] = (
                                sparse_A_n_chns_arr[sparse_A_right_node, sort_indcs]
                            )
                            naive_sparse_A_n_chns_arr[sparse_A_right_node] = (
                                naive_sparse_A_n_chns_arr[sparse_A_right_node, sort_indcs]
                            )
                            # Count bridge center node network feature
                            num_bridge_center_nodes += 1
                            break
            if bridge_center_node_elim: continue
            else:

                # Dangling node elimination
                for dangling_node in range(en):
                    # Node is connected to at least one other node
                    if np.count_nonzero(sparse_A_arr[dangling_node]) > 0:
                        # Gather neighbor nodes
                        nghbr_nodes = (
                            sparse_A_arr[dangling_node, np.nonzero(sparse_A_arr[dangling_node])[0]]
                        )
                        # Avoid primary loop chains (which will be
                        # addressed later)
                        nghbr_nodes = nghbr_nodes[nghbr_nodes!=dangling_node+1]
                        nghbr_nodes = nghbr_nodes[nghbr_nodes!=-1*(dangling_node+1)]
                        # Dangling node is connected to only 1 other
                        # node; dangling node exists
                        if np.size(nghbr_nodes) == 1:
                            dangling_node_elim = True
                            # Identify the root node connected to the
                            # dangling node
                            root_node = nghbr_nodes[0]
                            # Eliminate the dangling chain from the
                            # dangling node to the root node, and sort
                            # the resulting dangling node row of the
                            # sparse adjacency array and the sparse
                            # adjacency chain segment number arrays
                            root_node_k_indx = (
                                np.where(sparse_A_arr[dangling_node] == root_node)[0][0]
                            )
                            sparse_A_arr[dangling_node, root_node_k_indx] = 0
                            sparse_A_n_chns_arr[dangling_node, root_node_k_indx] = (
                                0.
                            )
                            naive_sparse_A_n_chns_arr[dangling_node, root_node_k_indx] = (
                                0.
                            )
                            sort_indcs = np.argsort(sparse_A_arr[dangling_node])
                            sparse_A_arr[dangling_node] = (
                                sparse_A_arr[dangling_node, sort_indcs]
                            )
                            sparse_A_n_chns_arr[dangling_node] = (
                                sparse_A_n_chns_arr[dangling_node, sort_indcs]
                            )
                            naive_sparse_A_n_chns_arr[dangling_node] = (
                                naive_sparse_A_n_chns_arr[dangling_node, sort_indcs]
                            )
                            # In a symmetrically duplicate fashion,
                            # eliminate the dangling chain from the root
                            # node to the dangling node, and sort the
                            # resulting root node row of the sparse
                            # adjacency array and the sparse adjacency
                            # chain segment number arrays
                            if root_node > 0:
                                symmtrc_root_node = root_node - 1
                                symmtrc_dangling_node = dangling_node + 1
                            else:
                                symmtrc_root_node = -1 * root_node - 1
                                symmtrc_dangling_node = -1 * (dangling_node+1)
                            symmtrc_dangling_node_k_indx = (
                                np.where(sparse_A_arr[symmtrc_root_node] == symmtrc_dangling_node)[0][0]
                            )
                            sparse_A_arr[symmtrc_root_node, symmtrc_dangling_node_k_indx] = (
                                0
                            )
                            sparse_A_n_chns_arr[symmtrc_root_node, symmtrc_dangling_node_k_indx] = (
                                0.
                            )
                            naive_sparse_A_n_chns_arr[symmtrc_root_node, symmtrc_dangling_node_k_indx] = (
                                0.
                            )
                            sort_indcs = np.argsort(
                                sparse_A_arr[symmtrc_root_node])
                            sparse_A_arr[symmtrc_root_node] = (
                                sparse_A_arr[symmtrc_root_node, sort_indcs]
                            )
                            sparse_A_n_chns_arr[symmtrc_root_node] = (
                                sparse_A_n_chns_arr[symmtrc_root_node, sort_indcs]
                            )
                            naive_sparse_A_n_chns_arr[symmtrc_root_node] = (
                                naive_sparse_A_n_chns_arr[symmtrc_root_node, sort_indcs]
                            )
                            # Count dangling node network feature
                            num_dangling_nodes += 1
                            break
                if dangling_node_elim: continue
                else:

                    # Primary loop elimination
                    for node in range(en):
                        # Node is connected to at least one other node
                        if np.count_nonzero(sparse_A_arr[node]) > 0:
                            # Gather indices of connection sites
                            # occupied by primary loops
                            primary_loop_k_indcs = (
                                np.where(np.logical_or(sparse_A_arr[node]==node+1, sparse_A_arr[node]==-1*(node+1)))[0]
                            )
                            num_primary_loop_k_indcs = np.size(
                                primary_loop_k_indcs)
                            # Primary loop(s) exists
                            if num_primary_loop_k_indcs > 0:
                                # Assert that there is at least one
                                # primary loop at hand, where each
                                # primary loop occupies two connection
                                # sites
                                assert (
                                    num_primary_loop_k_indcs >= 2
                                    and num_primary_loop_k_indcs % 2 == 0
                                )
                                primary_loop_elim = True
                                # Eliminate primary loop(s), and sort
                                # the node row of the sparse adjacency
                                # array and the sparse adjacency chain
                                # segment number arrays
                                sparse_A_arr[node, primary_loop_k_indcs] = 0
                                sparse_A_n_chns_arr[node, primary_loop_k_indcs] = (
                                    0.
                                )
                                naive_sparse_A_n_chns_arr[node, primary_loop_k_indcs] = (
                                    0.
                                )
                                sort_indcs = np.argsort(sparse_A_arr[node])
                                sparse_A_arr[node] = (
                                    sparse_A_arr[node, sort_indcs]
                                )
                                sparse_A_n_chns_arr[node] = (
                                    sparse_A_n_chns_arr[node, sort_indcs]
                                )
                                naive_sparse_A_n_chns_arr[node] = (
                                    naive_sparse_A_n_chns_arr[node, sort_indcs]
                                )
                                # Count primary loop network feature
                                num_primary_loops += 1
                                break
                    if primary_loop_elim: continue
                    # Elastically-effective cross-linked polydisperse
                    # end-linked phantom polymer network extraction
                    # procedure has finished
                    else: break

    # Gather chains array, chain types array, and effective/equivalent
    # chain segment numbers array of the elastically-effective
    # cross-linked polydisperse end-linked phantom polymer network gel
    conn_chns_gel_ee, conn_chns_type_gel_ee, n_chns_gel_ee = (
        sparse_A_arr_and_sparse_A_attr_arr_to_edges_and_edges_attr(
            sparse_A_arr, return_binary_edges_type=True,
            sparse_A_attr_arr=sparse_A_n_chns_arr)
    )
    
    # Gather naive chain segment numbers array of the
    # elastically-effective cross-linked polydisperse end-linked phantom
    # polymer network gel
    _, naive_n_chns_gel_ee = (
        sparse_A_arr_and_sparse_A_attr_arr_to_edges_and_edges_attr(
            sparse_A_arr, return_binary_edges_type=False,
            sparse_A_attr_arr=naive_sparse_A_n_chns_arr)
    )
    
    # Gather the number of elastically-effective cross-linkers, number
    # of elastically-effective chains, number of effective/equivalent
    # elastically-effective chain segments, and total number of
    # elastically-effective chain segments in the elastically-effective
    # cross-linked polydisperse end-linked phantom polymer network gel
    en_gel_ee = int(
        np.count_nonzero(core_nodes_type[np.unique(conn_chns_gel_ee)]==clnkr_type))
    em_gel_ee = np.shape(conn_chns_gel_ee)[0]
    en_n_gel_ee = np.sum(n_chns_gel_ee)
    en_n_tot_gel_ee = np.sum(naive_n_chns_gel_ee)
    
    return (
        conn_chns_gel_ee, conn_chns_type_gel_ee, n_chns_gel_ee,
        naive_n_chns_gel_ee, en_gel_ee, em_gel_ee, en_n_gel_ee, en_n_tot_gel_ee,
        num_multichains, num_bridge_center_nodes, num_dangling_nodes,
        num_primary_loops
    )

def predict_xi_from_cnvrsn_for_crosslinked_polydisperse_end_linked_polymer_network(
        chi: float,
        cnvrsn: float,
        cnvrsn_param: str,
        universe: Universe) -> float:
    """Predict the expected extent of the cross-linking reaction with
    respect to a provided reaction conversion parameter and a provided
    cross-linked polydisperse end-linked polymer network universe using
    Miller-Macosko theory.

    This function predicts the expected extent of the cross-linking
    reaction with respect to a provided reaction conversion parameter
    and a provided cross-linked polydisperse end-linked polymer network
    universe using Miller-Macosko theory.

    Args:
        chi (float): Stoichiometric imbalance between the number of cross-linker sites and the number of chain ends.
        cnvrsn (float): Conversion parameter; either the extent of the cross-linking reaction, the gel/network fraction, or the soluble fraction.
        cnvrsn_param (str): String indicating the identity of the conversion parameter; either the extent of the cross-linking reaction ("xi"), the gel/network fraction ("gel_frac"), or the soluble fraction ("sol_frac").
        universe (Universe): Cross-linked polydisperse end-linked polymer network universe. Importantly, it is assumed that the cross-linked polydisperse end-linked polymer network universe was synthesized with the exact same stoichiometric imbalance and conversion parameter as provided to this function.
    
    Returns:
        float: Expected extent of the cross-linking reaction.

    """
    # Establish chemical cross-linkers as type 2
    clnkr_type = 2

    # Return the expected extent of the cross-linking reaction if given
    # straight away
    if cnvrsn_param == "xi": return cnvrsn
    
    # Otherwise, predict the expected extent of the cross-linking
    # reaction
    else:
        # Calculate the soluble fraction
        if cnvrsn_param == "gel_frac": sol_frac = 1. - cnvrsn
        elif cnvrsn_param == "sol_frac": sol_frac = cnvrsn
        else:
            error_str = (
                "The specified conversion parameter must be either the "
                + "extent of polymerization (i.e., cross-linker "
                + "conversion), ''xi'', the gel/network fraction, "
                + "''gel_frac'', or the soluble fraction, ''sol_frac''."
            )
            raise ValueError(error_str)
        
        # Return the predicted the expected extent of the cross-linking
        # reaction
        return (
            predict_p_from_w_sol(
                sol_frac, network=universe, crosslinker_type=clnkr_type,
                functionality_per_type=universe.determine_functionality_per_type(),
                weight_fractions=universe.compute_weight_fractions(), r=chi,
                b2=1.)
        )