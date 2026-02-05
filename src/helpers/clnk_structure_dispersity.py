import numpy as np
import numpy.typing as npt
import networkx as nx
from scipy.spatial.transform import Rotation
from src.helpers.combinatorics import (
    permutations_with_replacement,
    indist_balls_in_dist_empty_bins_combinations,
    multinomial_coeff
)
from src.helpers.clnk_structure import (
    com_x_clnk_func,
    y_clnk_init_func,
    amended_3_chn_clnk_X_hat_clnk_func,
    regular_tetrahedral_4_chn_clnk_X_hat_clnk_func
)

def clnk_graph_func(
        y_clnk_init: npt.NDArray[np.floating],
        X_clnk: npt.NDArray[np.floating]) -> nx.Graph:
    """Graph of nodal positions (initial cross-link junction position
    and initial cross-link chain end positions) from a given initial
    cross-link structure RVE.

    This function initializes a NetworkX Graph of the nodal positions
    (initial cross-link junction position and initial cross-link chain
    end positions) from a given initial cross-link structure RVE.

    Args:
        y_clnk_init (npt.NDArray[np.floating]): Initial cross-link junction position for the cross-link structure RVE.
        X_clnk (npt.NDArray[np.floating]): Initial chain end position for each chain in the cross-link structure RVE.
    
    Returns:
        nx.Graph: Graph of nodal positions (initial cross-link junction
        position and initial cross-link chain end positions) from a
        given initial cross-link structure RVE.
    
    """
    clnk_graph = nx.Graph()
    clnk_graph.add_node(0, pos=y_clnk_init)
    for chn_indx in range(np.shape(X_clnk)[0]):
        clnk_graph.add_node(chn_indx+1, pos=X_clnk[chn_indx])
    return clnk_graph

def clnk_graph_add_edges_and_n_edge_attribute_func(
        clnk_graph: nx.Graph,
        n_clnk: npt.NDArray[np.floating | np.integer]) -> nx.Graph:
    """Graph of an initial cross-link structure RVE where the segment
    number of each chain is stored as an edge attribute.

    This function adds chains as edges to a NetworkX Graph representing
    the nodal positions of the initial cross-link structure RVE, and
    stores the segment number of each chain as an edge attribute.

    Args:
        clnk_graph (nx.Graph): Graph of nodal positions (initial cross-link junction position and initial cross-link chain end positions) from a given initial cross-link structure RVE.
        n_clnk (npt.NDArray[np.floating | np.integer]): Number of chain segments for each chain in the cross-link structure RVE.
    
    Returns:
        nx.Graph: Graph of an initial cross-link structure RVE where the
        segment number of each chain is stored as an edge attribute.
    
    """
    k_num = np.shape(n_clnk)[0]
    assert clnk_graph.number_of_nodes()-1 == k_num
    for chn_indx in range(k_num):
        clnk_graph.add_edge(0, chn_indx+1, n=float(n_clnk[chn_indx]))
    return clnk_graph

def is_geometrically_isomorphic(
        clnk_graph_0: nx.Graph,
        clnk_graph_1: nx.Graph,
        tol: float = 1.e-6) -> bool:
    """Geometric isomorphism test between two initial cross-link
    structure RVEs.

    This function tests if two initial cross-link structure RVEs are
    geometrically isomorphic.

    Args:
        clnk_graph_0 (nx.Graph): NetworkX Graph representation of a initial cross-link structure RVE.
        clnk_graph_1 (nx.Graph): NetworkX Graph representation of another initial cross-link structure RVE.
        tol (float): Kabsch algorithm root-sum-squared distance metric tolerance value. Default is 1.e-6.
    
    Returns:
        bool: Result of the geometric isomorphism test between two
        initial cross-link structure RVEs.
    
    """
    # Extract the general graph isomorphism mapping between the two
    # provided initial cross-link structure graphs, where the edge
    # attribute of chain segment number is required for matching
    clnk_graph_matcher = nx.algorithms.isomorphism.GraphMatcher(
        clnk_graph_0, clnk_graph_1,
        edge_match=nx.algorithms.isomorphism.numerical_edge_match("n", None))
    
    # Assess each general graph isomorphism mapping between the two
    # provided initial cross-link structure graphs to test if each
    # mapping is also invariant to rotations of the initial cross-link
    # structures
    for clnk_graph_mapping in clnk_graph_matcher.isomorphisms_iter():
        # Extract two generally isomorphic initial cross-link structure
        # graphs as specified by the general graph isomorphism mapping
        y_clnk_init_X_clnk_0 = (
            np.asarray(
                [clnk_graph_0.nodes[node]["pos"] for node in clnk_graph_mapping.keys()])
        )
        y_clnk_init_X_clnk_1 = (
            np.asarray(
                [clnk_graph_1.nodes[clnk_graph_mapping[node]]["pos"] for node in clnk_graph_mapping.keys()])
        )

        # Remove any possible translation applied to the nodal
        # coordiantes of the two generally isomorphic initial cross-link
        # structures by centering the center-of-mass of the initial
        # cross-link structures at the origin
        y_clnk_init_X_clnk_0 -= com_x_clnk_func(y_clnk_init_X_clnk_0)
        y_clnk_init_X_clnk_1 -= com_x_clnk_func(y_clnk_init_X_clnk_1)

        # Use the Kabsch algorithm to attempt to find the rotation that
        # optimally aligns the two generally isomorphic initial
        # cross-link structures, and from this operation, yield the
        # associated root-sum-squared distance metric
        _, rssd = Rotation.align_vectors(
            y_clnk_init_X_clnk_0, y_clnk_init_X_clnk_1)
        
        # If any two generally isomorphic initial cross-link structures
        # can be rotated such that optimal alignment is achieved (as
        # measured by comparing the root-sum-squared distance with some
        # small tolerance value), then the two provided initial
        # cross-link structure graphs are geometrically isomorphic
        if rssd < tol: return True
    
    # If no two generally isomorphic initial cross-link structures can
    # be rotated such that optimal alignment is achieved, then the two
    # provided initial cross-link structure graphs are not geometrically
    # isomorphic
    return False

def geometrically_isomorphic_clnks_symmetric_under_chain_permutation_assembly(
        n: npt.NDArray[np.floating | np.integer],
        p_n: npt.NDArray[np.floating],
        X_hat_clnk: npt.NDArray[np.floating]) -> tuple[npt.NDArray[np.floating | np.integer], npt.NDArray[np.floating]]:
    """Assemble representative cross-link structures and their
    associated probability distributions in the case where all
    permutations of chains in the cross-link structure belong to the
    same geometrically isomorphic group (and thus, there is symmetry
    equivalence under chain permutation).

    This function generates representative cross-link structures and
    calculates the associated probability distribution
    for each such cross-link structure in the case where all
    permutations of chains in the cross-link structure belong to the
    same geometrically isomorphic group (and thus, there is symmetry
    equivalence under chain permutation), given a set of salient chain
    segment numbers, the probability distribution of said chain segment
    numbers, and the initial unit chain end position for each chain in
    the cross-link structure RVE.

    Args:
        n (npt.NDArray[np.floating | np.integer]): Salient chain segment numbers.
        p_n (npt.NDArray[np.floating]): Polymer chain segment number probability distribution.
        X_hat_clnk (npt.NDArray[np.floating]): Initial unit chain end position for each chain in the cross-link structure RVE.
    
    Returns:
        tuple[npt.NDArray[np.floating | np.integer], npt.NDArray[np.floating]]:
        Representative cross-link structures and their associated
        probability distributions in the case where all permutations of
        chains in the cross-link structure belong to the same
        geometrically isomorphic group (and thus, there is symmetry
        equivalence under chain permutation).
    
    """
    # Initialize parameters
    N = np.shape(n)[0]
    k = np.shape(X_hat_clnk)[0]

    # Gather initial cross-link unit chain end positions for the
    # cross-link structures where all permutations of chains in the
    # cross-link structure belong to the same geometrically isomorphic
    # group
    amended_3_chn_clnk_X_hat_clnk = amended_3_chn_clnk_X_hat_clnk_func()
    regular_tetrahedral_4_chn_clnk_X_hat_clnk = (
        regular_tetrahedral_4_chn_clnk_X_hat_clnk_func()
    )
    
    # Boilerplate checks
    if np.shape(n) != np.shape(p_n):
        error_str = (
            "The shape of the salient chain segment number array must "
            + "match that for the corresponding chain segment number "
            + "probability distribution."
        )
        raise ValueError(error_str)
    if not np.isclose(np.sum(p_n), 1.):
        error_str = (
            "The chain segment number probability distribution must be "
            + "a normalized distribution (normalized to sum to 1)."
        )
        raise ValueError(error_str)
    geo_isomrphc_clnk_symmtrc_under_chn_prmttn = False
    if k == 3:
        geo_isomrphc_clnk_symmtrc_under_chn_prmttn = np.allclose(
            X_hat_clnk, amended_3_chn_clnk_X_hat_clnk)
    elif k == 4:
        geo_isomrphc_clnk_symmtrc_under_chn_prmttn = np.allclose(
            X_hat_clnk, regular_tetrahedral_4_chn_clnk_X_hat_clnk)
    if not geo_isomrphc_clnk_symmtrc_under_chn_prmttn:
        error_str = (
            "This function is only applicable to the amended 3-chain "
            + "cross-link structure or the regular tetrahedral 4-chain "
            + "cross-link structure because these are the only two "
            + "cross-link structures where all permutations of chains "
            + "(in these cross-link structures) belong to the same "
            + "geometrically isomorphic group."
        )
        raise ValueError(error_str)
    
    # Deduce and return the representative cross-link structures and
    # their associated probabilities of occurance
    m_clnks = m_clnks_symmetric_under_chain_permutation_func(k, N)
    n_clnks = n_clnks_symmetric_under_chain_permutation_func(n, m_clnks)
    p_n_k_clnks = p_n_k_clnks_symmetric_under_chain_permutation_func(
        C_clnks_symmetric_under_chain_permutation_func(m_clnks), p_n, m_clnks)
    return n_clnks, p_n_k_clnks

def geometrically_isomorphic_clnks_assembly(
        n: npt.NDArray[np.floating | np.integer],
        p_n: npt.NDArray[np.floating],
        k: npt.NDArray[np.integer],
        X_hat_clnks: list[npt.NDArray[np.floating]],
        consolidate_geo_isomrphc_group: bool = False) -> tuple[list[list[npt.NDArray[np.floating | np.integer]]], list[list[npt.NDArray[np.floating]]]] | tuple[list[npt.NDArray[np.floating | np.integer]], list[npt.NDArray[np.floating]]]:
    """Assemble geometrically isomorphic cross-link structures and their
    associated probability distributions.

    This function generates all possible permutations of cross-link
    structures and calculates the associated probability distribution
    for each such cross-link structure, given a set of salient chain
    segment numbers, the probability distribution of said chain segment
    numbers, and the degree of the cross-link structures. Then, the
    cross-link structures are segregated into separate geometrically
    isomorphic groups. If desired, each of these geometrically
    isomorphic groups of cross-link structures can be consolidated by
    returning the leading cross-link structure from the group (as a
    group representative).

    Args:
        n (npt.NDArray[np.floating | np.integer]): Salient chain segment numbers.
        p_n (npt.NDArray[np.floating]): Polymer chain segment number probability distribution.
        k (npt.NDArray[np.integer]): Degree of all cross-links.
        X_hat_clnks (list[npt.NDArray[np.floating]]): Initial unit chain end position for each chain in each of the initial cross-link structure RVEs.
        consolidate_geo_isomrphc_group (bool): Boolean indicating if the geometrically isomorphic groups of cross-link structures ought to be consolidated (if True) or left as is (if False). Default is False.
    
    Returns:
        tuple[list[list[npt.NDArray[np.floating | np.integer]]], list[list[npt.NDArray[np.floating]]]] | tuple[list[npt.NDArray[np.floating | np.integer]], list[npt.NDArray[np.floating]]]:
        Geometrically isomorphic groups of cross-link structures and
        their associated probability distributions.
    
    """
    # Initialize parameters
    N = np.shape(n)[0]
    k_num = np.shape(k)[0]

    # Boilerplate checks
    if np.shape(n) != np.shape(p_n):
        error_str = (
            "The shape of the salient chain segment number array must "
            + "match that for the corresponding chain segment number "
            + "probability distribution."
        )
        raise ValueError(error_str)
    if not np.isclose(np.sum(p_n), 1.):
        error_str = (
            "The chain segment number probability distribution must be "
            + "a normalized distribution (normalized to sum to 1)."
        )
        raise ValueError(error_str)
    if k_num != len(X_hat_clnks):
        error_str = (
            "The length of the list of specified initial cross-link "
            + "unit chain end positions, i.e., initial cross-link "
            + "structures, must be equal to the number of cross-link "
            + "degrees being considered."
        )
        raise ValueError(error_str)
    for k_indx in range(k_num):
        if np.shape(X_hat_clnks[k_indx]) != (k[k_indx], 3):
            error_str = (
                "The number of chains in each initial cross-link "
                + "structure must be equal to the corresponding entry "
                + "in the cross-link degree array."
            )
            raise ValueError(error_str)

    # Initialize parameters
    n_clnks = []
    p_n_k_clnks = []
    k_3_n_clnks_symmtry_prmttn_geo_isomrphc_group = False
    k_4_n_clnks_symmtry_prmttn_geo_isomrphc_group = False

    # Account for the case of monodisperse chain segment number
    # probability distribution
    p_n_poly = True
    if N == 1:
        p_n_poly = False
        for k_indx in range(k_num):
            # Monodisperse cross-link structure
            n_clnks.append([n[0]*np.ones_like(n, shape=(1, k[k_indx]))])
            p_n_k_clnks.append([np.asarray([1.])])
    else:
        p_n_unique = np.unique(p_n)
        if np.shape(p_n_unique)[0] == 2:
            if np.allclose(p_n_unique, np.asarray([0., 1.])):
                p_n_poly = False
                n_mono_indx = np.where(p_n==1.)[0][0]
                for k_indx in range(k_num):
                    # Monodisperse cross-link structure
                    n_clnks.append(
                        [n[n_mono_indx]*np.ones_like(n, shape=(1, k[k_indx]))])
                    p_n_k_clnks.append([np.asarray([1.])])
    # Account for the case of the polydisperse chain segment number
    # probability distribution
    if p_n_poly:
        # Gather the initial cross-link junction position
        y_clnk_init = y_clnk_init_func()

        # Gather initial cross-link unit chain end positions for the
        # cross-link structures where all permutations of chains in the
        # cross-link structure belong to the same geometrically
        # isomorphic group
        amended_3_chn_clnk_X_hat_clnk = amended_3_chn_clnk_X_hat_clnk_func()
        regular_tetrahedral_4_chn_clnk_X_hat_clnk = (
            regular_tetrahedral_4_chn_clnk_X_hat_clnk_func()
        )
        
        # Evaluate all possible cross-link structures for each
        # cross-link degree
        for k_indx in range(k_num):
            # Gather the unit cross-link chain end positions
            X_hat_clnk = X_hat_clnks[k_indx]

            # If applicable, construct the cross-link structures and
            # their associated probabilities of occurance for the case
            # where all permutations of chains in the cross-link
            # structure belong to the same geometrically isomorphic
            # group
            if consolidate_geo_isomrphc_group and k[k_indx] == 3:
                if np.allclose(X_hat_clnk, amended_3_chn_clnk_X_hat_clnk):
                    k_3_n_clnks_symmtry_prmttn_geo_isomrphc_group = True
                    n_clnks_k_vals, p_n_k_clnks_k_vals = (
                        geometrically_isomorphic_clnks_symmetric_under_chain_permutation_assembly(
                            n, p_n, X_hat_clnk)
                    )
            elif consolidate_geo_isomrphc_group and k[k_indx] == 4:
                if np.allclose(X_hat_clnk, regular_tetrahedral_4_chn_clnk_X_hat_clnk):
                    k_4_n_clnks_symmtry_prmttn_geo_isomrphc_group = True
                    n_clnks_k_vals, p_n_k_clnks_k_vals = (
                        geometrically_isomorphic_clnks_symmetric_under_chain_permutation_assembly(
                            n, p_n, X_hat_clnk)
                    )
            else:
                # Gather all possible cross-link structures by
                # permutating the salient chain segment numbers, with
                # replacement, over all the chains in the cross-link
                # structure
                k_permutations = permutations_with_replacement(N, k[k_indx])
                num_k_permutations = np.shape(k_permutations)[0]
                
                # Construct all possible cross-link structures and their
                # associated probabilities of occurance within this
                # population of cross-links
                n_clnks_k_permutations = n[k_permutations]
                p_n_k_clnks_k_permutations = np.prod(p_n[k_permutations], axis=1)
                
                # Initialize lists to segregate all possible cross-link
                # structures into separate geometrically isomorphic
                # groups, and populate these lists with the first
                # possible cross-link structure and its associated
                # probability to instantiate the first geometrically
                # isomorphic group
                n_clnks_k_vals = []
                p_n_k_clnks_k_vals = []
                n_clnks_k_vals.append(np.atleast_2d(n_clnks_k_permutations[0]))
                p_n_k_clnks_k_vals.append(
                    np.atleast_1d(p_n_k_clnks_k_permutations[0]))
                
                # Segregate all possible cross-link structures into
                # separate geometrically isomorphic groups
                if num_k_permutations > 1:
                    # Initialize a graph to represent the baseline
                    # initial unit cross-link structure
                    clnk_graph = clnk_graph_func(y_clnk_init, X_hat_clnk)
                    
                    # Segregate the remaining possible cross-link
                    # structures into separate geometrically isomorphic
                    # groups 
                    for prmttn in range(1, num_k_permutations):
                        # Extract a cross-link structure and its
                        # associated probability of occurance from the
                        # population of all possible cross-link
                        # structures
                        n_clnk_prmttn = n_clnks_k_permutations[prmttn]
                        p_n_k_clnk_prmttn = p_n_k_clnks_k_permutations[prmttn]

                        # Initialize a boolean that tracks if the
                        # extracted cross-link structure belongs to a
                        # previously instantiated group of geometrically
                        # isomorphic cross-link structures, or not
                        is_geo_isomrphc = False

                        # Determine if the cross-link structure belongs
                        # to a previously instantiated group of
                        # geometrically isomorphic cross-link structures
                        for geo_isomrphc_group_indx in range(len(n_clnks_k_vals)):
                            # Extract the leading cross-link structure
                            # from a previously instantiated group of
                            # geometrically isomorphic cross-link
                            # structures
                            n_clnk_geo_isomrphc_group = (
                                n_clnks_k_vals[geo_isomrphc_group_indx][0]
                            )

                            # Initialize graphs (on top of the baseline
                            # initial unit cross-link structure) that
                            # represent the initial unit cross-link
                            # structure and the segment number of each
                            # chain in the cross-link for the extracted
                            # cross-link structure and the leading
                            # geometrically isomorphic cross-link
                            # structure
                            clnk_prmttn_graph = (
                                clnk_graph_add_edges_and_n_edge_attribute_func(
                                    clnk_graph.copy(), n_clnk_prmttn)
                            )
                            clnk_geo_isomrphc_group_graph = (
                                clnk_graph_add_edges_and_n_edge_attribute_func(
                                    clnk_graph.copy(), n_clnk_geo_isomrphc_group)
                            )
                            
                            # Test if the extracted cross-link structure
                            # belongs to this particular group of
                            # geometrically isomorphic cross-link
                            # structures
                            if is_geometrically_isomorphic(clnk_prmttn_graph, clnk_geo_isomrphc_group_graph):
                                # If the extracted cross-link structure
                                # belongs to this particular group of
                                # geometrically isomorphic cross-link
                                # structures, then add it to this group
                                n_clnks_k_vals[geo_isomrphc_group_indx] = np.vstack(
                                    (
                                        n_clnks_k_vals[geo_isomrphc_group_indx],
                                        np.atleast_2d(n_clnk_prmttn)))
                                p_n_k_clnks_k_vals[geo_isomrphc_group_indx] = (
                                    np.hstack(
                                        (
                                            p_n_k_clnks_k_vals[geo_isomrphc_group_indx],
                                            np.atleast_1d(p_n_k_clnk_prmttn)))
                                )
                                is_geo_isomrphc = True
                                break
                            
                        # Instantiate a new group of geometrically
                        # isomorphic cross-link structures if the
                        # extracted cross-link structure is found to not
                        # belong to any of the other groups
                        if not is_geo_isomrphc:
                            n_clnks_k_vals.append(np.atleast_2d(n_clnk_prmttn))
                            p_n_k_clnks_k_vals.append(
                                np.atleast_1d(p_n_k_clnk_prmttn))
            
            # Append the lists representing the geometrically isomorphic
            # groups of cross-link structures with a particular degree
            # to lists representing that for all cross-link degrees
            n_clnks.append(n_clnks_k_vals)
            p_n_k_clnks.append(p_n_k_clnks_k_vals)
    
    # If called for, consolidate the groups of geometrically isomorphic
    # cross-link structures together by retaining the leading cross-link
    # structure from each group and summing the cross-link structure
    # probabilities within each group together
    if consolidate_geo_isomrphc_group:
        # Consolidate the groups of geometrically isomorphic cross-link
        # structures together for each cross-link degree
        for k_indx in range(k_num):
            if ((k[k_indx] == 3 and k_3_n_clnks_symmtry_prmttn_geo_isomrphc_group) or
                (k[k_indx] == 4 and k_4_n_clnks_symmtry_prmttn_geo_isomrphc_group)):
                continue
            else:
                # Initialize consolidated cross-link structure arrays
                num_geo_isomrphc_groups = len(n_clnks[k_indx])
                n_clnks_k_vals = np.empty_like(
                    n, shape=(num_geo_isomrphc_groups, k[k_indx]))
                p_n_k_clnks_k_vals = np.empty_like(
                    p_n, shape=(num_geo_isomrphc_groups,))
                
                # Consolidate each group of geometrically isomorphic
                # cross-link structures together by retaining the
                # leading cross-link structure from each group and
                # summing the cross-link structure probabilities within
                # each group together
                for geo_isomrphc_group_indx in range(num_geo_isomrphc_groups):
                    n_clnks_k_vals[geo_isomrphc_group_indx] = (
                        n_clnks[k_indx][geo_isomrphc_group_indx][0]
                    )
                    p_n_k_clnks_k_vals[geo_isomrphc_group_indx] = np.sum(
                        p_n_k_clnks[k_indx][geo_isomrphc_group_indx])
                
                # Update the lists representing the geometrically
                # isomorphic groups of cross-link structures with this
                # consolidated format
                n_clnks[k_indx] = n_clnks_k_vals
                p_n_k_clnks[k_indx] = p_n_k_clnks_k_vals
    
    return n_clnks, p_n_k_clnks

def p_clnks_init_func(
        p_k_clnks: npt.NDArray[np.floating],
        p_n_k_clnks: list[npt.NDArray[np.floating]] | list[list[npt.NDArray[np.floating]]]) -> list[npt.NDArray[np.floating]] | list[list[npt.NDArray[np.floating]]]:
    """Probability distribution of distinct elastically-effective
    cross-link structures with degree k.

    This function calculates the probability distribution of distinct
    elastically-effective cross-link structures with degree k.

    Args:
        p_k_clnks (npt.NDArray[np.floating]): Probability distribution of elastically-effective cross-linkers with degree k.
        p_n_k_clnks (list[npt.NDArray[np.floating]] | list[list[npt.NDArray[np.floating]]]): Probability distribution of distinct (elastically-effective) cross-link structures with degree k.
    
    Returns:
        list[npt.NDArray[np.floating]] | list[list[npt.NDArray[np.floating]]]:
        Probability distribution of distinct elastically-effective
        cross-link structures with degree k.
    
    """
    # Boilerplate initialization and check
    k_num = np.shape(p_k_clnks)[0]
    if k_num != len(p_n_k_clnks):
        error_str = (
            "The cross-link structures represented by p_k_clnks and "
            + "p_n_k_clnks are not compatible with one another."
        )
        raise ValueError(error_str)
    
    p_clnks = []
    if type(p_n_k_clnks[0]) == np.ndarray:
        for k_indx in range(k_num):
            p_clnks.append(p_k_clnks[k_indx, np.newaxis]*p_n_k_clnks[k_indx])
    elif type(p_n_k_clnks[0]) == list and type(p_n_k_clnks[0][0]) == np.ndarray:
        for k_indx in range(k_num):
            p_clnks_k_vals = []
            for geo_isomrphc_group_indx in range(len(p_n_k_clnks[k_indx])):
                p_clnks_k_vals.append(
                    p_k_clnks[k_indx, np.newaxis]*p_n_k_clnks[k_indx][geo_isomrphc_group_indx])
            p_clnks.append(p_clnks_k_vals)
    else:
        error_str = (
            "The probability distribution of distinct cross-link "
            + "structures with degree k is not provided in the correct "
            + "type of either list[npt.NDArray[np.floating]] or "
            + "list[list[npt.NDArray[np.floating]]]."
        )
        raise ValueError(error_str)
    return p_clnks

def m_clnks_symmetric_under_chain_permutation_func(
        k: int,
        N: int) -> npt.NDArray[np.integer]:
    """Chain segment number multiplicity for each distinct cross-link
    structure (with symmetry equivalence under chain permutation).

    This function computes the chain segment number multiplicity for
    each distinct cross-link structure (with symmetry equivalence under
    chain permutation).

    Args:
        k (int): Cross-link degree.
        N (int): Number of salient polymer chain segment numbers.
    
    Returns:
        npt.NDArray[np.integer]: Chain segment number multiplicity for
        each distinct cross-link structure (with symmetry equivalence
        under chain permutation).
    
    """
    return indist_balls_in_dist_empty_bins_combinations(k, N)

def n_clnks_symmetric_under_chain_permutation_func(
        n: npt.NDArray[np.floating | np.integer],
        m_clnks: npt.NDArray[np.integer]) -> npt.NDArray[np.floating | np.integer]:
    """Chain segment number for each chain in each distinct cross-link
    structure (with symmetry equivalence under chain permutation).

    This function tabulates the chain segment number for each chain in
    each distinct cross-link structure (with symmetry equivalence under
    chain permutation).

    Args:
        n (npt.NDArray[np.floating | np.integer]): Salient chain segment numbers (sorted from least to greatest).
        m_clnks (npt.NDArray[np.integer]): Chain segment number multiplicity for each distinct cross-link structure (with symmetry equivalence under chain permutation).
    
    Returns:
        npt.NDArray[np.floating | np.integer]: Chain segment number for
        each chain in each distinct cross-link structure (with symmetry
        equivalence under chain permutation) (sorted from least to
        greatest for each cross-link structure).
    
    """
    # Boilerplate checks
    if np.shape(n)[0] != np.shape(m_clnks)[1]:
        error_str = (
            "The number of segments represented in the segment number "
            + "multiplicity array does not equal the provided number "
            + "of segments."
        )
        raise ValueError(error_str)
    
    return np.vstack([np.repeat(n, m_clnk) for m_clnk in m_clnks])

def C_clnks_symmetric_under_chain_permutation_func(
        m_clnks: npt.NDArray[np.integer]) -> npt.NDArray[np.integer]:
    """Number of permutations that exist for each distinct cross-link
    structure due to symmetry equivalence under chain permutation.

    This function calculates the number of permutations that exist for
    each distinct cross-link structure due to symmetry equivalence under
    chain permutation.

    Args:
        m_clnks (npt.NDArray[np.integer]): Chain segment number multiplicity for each distinct cross-link structure (with symmetry equivalence under chain permutation).
    
    Returns:
        npt.NDArray[np.integer]: Number of permutations that exist for
        each distinct cross-link structure due to symmetry equivalence
        under chain permutation.
    
    """
    return np.hstack([multinomial_coeff(m_clnk) for m_clnk in m_clnks])

def p_n_k_clnks_symmetric_under_chain_permutation_func(
        C_clnks: npt.NDArray[np.integer],
        p_n: npt.NDArray[np.floating],
        m_clnks: npt.NDArray[np.integer]) -> npt.NDArray[np.floating]:
    """Probability distribution of distinct cross-link structures (with
    symmetry equivalence under chain permutation) with degree k.

    This function calculates the probability distribution of distinct
    cross-link structures (with symmetry equivalence under chain
    permutation) with degree k.

    Args:
        C_clnks: (npt.NDArray[np.integer]): Number of permutations that exist for each distinct cross-link structure due to symmetry equivalence under chain permutation.
        p_n (npt.NDArray[np.floating]): Polymer chain segment number probability distribution.
        m_clnks: (npt.NDArray[np.integer]): Chain segment number multiplicity for each distinct cross-link structure (with symmetry equivalence under chain permutation).
    
    Returns:
        npt.NDArray[np.floating]: Probability distribution of distinct
        cross-link structures (with symmetry equivalence under chain
        permutation) with degree k.
    
    """
    # Boilerplate checks
    if np.shape(C_clnks)[0] != np.shape(m_clnks)[0]:
        error_str = (
            "The number of cross-link structures represented by "
            + "C_clnks and m_clnks are not compatible with one "
            + "another."
        )
        raise ValueError(error_str)
    if np.shape(p_n)[0] != np.shape(m_clnks)[1]:
        error_str = (
            "The number of segments represented in the segment number "
            + "multiplicity array does not equal the provided number "
            + "of segments in the chain segment number probability "
            + "distribution."
        )
        raise ValueError(error_str)

    C_R = np.shape(C_clnks)[0]
    p_n_k_clnks = np.empty(C_R)
    for clnk_indx in range(C_R):
        p_n_k_clnks[clnk_indx] = (
            C_clnks[clnk_indx] * np.prod(np.power(p_n, m_clnks[clnk_indx]))
        )
    return p_n_k_clnks