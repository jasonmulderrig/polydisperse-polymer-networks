import numpy as np
from src.helpers.network_topology_initialization_utils import (
    tessellation_protocol,
    tessellation
)

def core_pb_edge_id(
        core_node_0_coords: np.ndarray,
        core_node_1_coords: np.ndarray,
        L: np.ndarray) -> tuple[np.ndarray, float]:
    """Periodic boundary edge and node identification.

    This function uses the minimum image criterion to determine/identify
    the node coordinates of a particular periodic boundary edge.

    Args:
        core_node_0_coords (np.ndarray): 1D np.ndarray with dim float entries of the coordinates of the core node in the periodic boundary edge.
        core_node_1_coords (np.ndarray): 1D np.ndarray with dim float entries of the coordinates of the core node that translates/tessellates to the periodic node in the periodic boundary edge.
        L (np.ndarray): 1D np.ndarray with dim float entries of the tessellation scaling (i.e., simulation box side lengths).

    Returns:
        tuple[np.ndarray, float]: 1D np.ndarray with dim float entries
        of the coordinates of the periodic node in the periodic boundary
        edge, and the length of the periodic boundary edge,
        respectively.
    
    """
    # Confirm that coordinate dimensions match
    if np.shape(core_node_0_coords)[0] != np.shape(core_node_1_coords)[0]:
        error_str = (
            "The dimensionality of the core node coordinates at hand "
            + "in the periodic boundary edge and node identification "
            + "do not match." 
        )
        raise ValueError(error_str)
    
    # Calculate network dimension
    dim = np.shape(core_node_0_coords)[0]

    # Tessellation protocol
    tsslltn, tsslltn_num = tessellation_protocol(dim)
    
    # Use tessellation protocol to tessellate core_node_1
    core_node_1_tsslltn_coords = tessellation(core_node_1_coords, tsslltn, L)
    
    # Use minimum image/distance criterion to select the correct
    # periodic boundary node and edge corresponding to core_node_1
    l_pb_nodes_1 = np.empty(tsslltn_num)
    for pb_node_1 in range(tsslltn_num):
        l_pb_nodes_1[pb_node_1] = np.linalg.norm(
            core_node_1_tsslltn_coords[pb_node_1]-core_node_0_coords)
    pb_node_1 = np.argmin(l_pb_nodes_1)
    
    return core_node_1_tsslltn_coords[pb_node_1], l_pb_nodes_1[pb_node_1]

def l_func(
        conn_edges: np.ndarray,
        conn_edges_type: np.ndarray,
        coords: np.ndarray,
        L: np.ndarray) -> np.ndarray:
    """Euclidean edge lengths.

    This function calculates the Euclidean length of each supplied edge.

    Args:
        conn_edges (np.ndarray): 2D np.ndarray with (em, 2) int entries of edges from the graph capturing the periodic connections between the core nodes.
        conn_edges_type (np.ndarray): 1D np.ndarray with em float entries of the type label for the edges from the graph capturing the periodic connections between the core nodes. Core edges are of type 1, and periodic boundary edges are of type 2.
        coords (np.ndarray): 2D np.ndarray with (en, dim) float entries of the coordinates of the core nodes.
        L (np.ndarray): 1D np.ndarray with dim float entries of the tessellation scaling (i.e., simulation box side lengths).
    
    Returns:
        np.ndarray: 1D np.ndarray with em float entries of Euclidean
        edge lengths.
    
    """
    # Initialize edge length np.ndarray
    em = np.shape(conn_edges)[0]
    l_edges = np.empty(em)

    # Calculate and store the length of each edge
    for edge in range(em):
        # Node numbers
        core_node_0 = int(conn_edges[edge, 0])
        core_node_1 = int(conn_edges[edge, 1])
        # Edge type
        edge_type = conn_edges_type[edge]
        
        # Self-loops have zero edge length
        if core_node_0 == core_node_1: l_edges[edge] = 0.0
        # Edge is a core edge
        elif edge_type == 1:
            # Core edge length
            l_edges[edge] = np.linalg.norm(
                coords[core_node_1]-coords[core_node_0])
        # Edge is a periodic boundary edge
        elif edge_type == 2:
            # Periodic boundary edge length
            _, l_edges[edge] = core_pb_edge_id(
                coords[core_node_0], coords[core_node_1], L)
    
    return l_edges

def l_inv_func(
        conn_edges: np.ndarray,
        conn_edges_type: np.ndarray,
        coords: np.ndarray,
        L: np.ndarray) -> np.ndarray:
    """Inverse Euclidean edge lengths.

    This function calculates the inverse Euclidean length of each
    supplied edge.

    Args:
        conn_edges (np.ndarray): 2D np.ndarray with (em, 2) int entries of edges from the graph capturing the periodic connections between the core nodes.
        conn_edges_type (np.ndarray): 1D np.ndarray with em float entries of the type label for the edges from the graph capturing the periodic connections between the core nodes. Core edges are of type 1, and periodic boundary edges are of type 2.
        coords (np.ndarray): 2D np.ndarray with (en, dim) float entries of the coordinates of the core nodes.
        L (np.ndarray): 1D np.ndarray with dim float entries of the tessellation scaling (i.e., simulation box side lengths).
    
    Returns:
        np.ndarray: 1D np.ndarray with em float entries of inverse
        Euclidean edge lengths.
    
    """
    # Calculate Euclidean edge lengths
    l = l_func(conn_edges, conn_edges_type, coords, L)

    # Calculate and return inverse Euclidean edge length (where the
    # inverse Euclidean edge length for self-loops is set to zero)
    return np.reciprocal(l, where=l!=0.0)

def l_cmpnts_func(
        conn_edges: np.ndarray,
        conn_edges_type: np.ndarray,
        coords: np.ndarray,
        L: np.ndarray) -> np.ndarray:
    """Euclidean edge length components.

    This function calculates the Euclidean length components of each
    supplied edge.

    Args:
        conn_edges (np.ndarray): 2D np.ndarray with (em, 2) int entries of edges from the graph capturing the periodic connections between the core nodes.
        conn_edges_type (np.ndarray): 1D np.ndarray with em float entries of the type label for the edges from the graph capturing the periodic connections between the core nodes. Core edges are of type 1, and periodic boundary edges are of type 2.
        coords (np.ndarray): 2D np.ndarray with (en, dim) float entries of the coordinates of the core nodes.
        L (np.ndarray): 1D np.ndarray with dim float entries of the tessellation scaling (i.e., simulation box side lengths).
    
    Returns:
        np.ndarray: 2D np.ndarray with (em, dim) float entries of
        Euclidean edge length components.
    
    """
    # Initialize edge length components np.ndarray
    em = np.shape(conn_edges)[0]
    dim = np.shape(coords)[1]
    l_cmpnt_edges = np.empty((em, dim))

    # Calculate and store the length components of each edge
    for edge in range(em):
        # Node numbers
        core_node_0 = int(conn_edges[edge, 0])
        core_node_1 = int(conn_edges[edge, 1])
        # Edge type
        edge_type = conn_edges_type[edge]

        # Self-loops have zero edge length components
        if core_node_0 == core_node_1: l_cmpnt_edges[edge] = np.zeros(dim)
        # Edge is a core edge
        elif edge_type == 1:
            # Core edge length components
            l_cmpnt_edges[edge] = coords[core_node_1] - coords[core_node_0]
        # Edge is a periodic boundary edge
        elif edge_type == 2:
            # Periodic boundary edge length components
            core_node_0_coords = coords[core_node_0]
            core_node_1_coords = coords[core_node_1]
            pb_node_1_coords, _ = core_pb_edge_id(
                core_node_0_coords, core_node_1_coords, L)
            l_cmpnt_edges[edge] = pb_node_1_coords - core_node_0_coords
    
    return l_cmpnt_edges

def l_naive_func(
        conn_edges: np.ndarray,
        coords: np.ndarray) -> np.ndarray:
    """Naive Euclidean edge lengths.

    This function calculates the naive Euclidean length of each supplied
    edge.

    Args:
        conn_edges (np.ndarray): 2D np.ndarray with (em, 2) int entries of edges from the graph capturing the periodic connections between the core nodes.
        coords (np.ndarray): 2D np.ndarray with (en, dim) float entries of the coordinates of the core nodes.
    
    Returns:
        np.ndarray: 1D np.ndarray with em float entries of naive
        Euclidean edge lengths.
    
    """
    # Initialize edge length np.ndarray
    em = np.shape(conn_edges)[0]
    l_naive_edges = np.empty(em)

    # Calculate and store the length of each edge
    for edge in range(em):
        # Node numbers
        core_node_0 = int(conn_edges[edge, 0])
        core_node_1 = int(conn_edges[edge, 1])
        
        # Self-loops have zero edge length
        if core_node_0 == core_node_1: l_naive_edges[edge] = 0.0
        else:
            l_naive_edges[edge] = np.linalg.norm(
                coords[core_node_1]-coords[core_node_0])
    
    return l_naive_edges

def gamma_func(
        conn_edges: np.ndarray,
        conn_edges_type: np.ndarray,
        l_cntr_conn_edges: np.ndarray,
        coords: np.ndarray,
        L: np.ndarray) -> np.ndarray:
    """Chain/Edge stretches.

    This function calculates the chain/edge stretch of each supplied
    edge.

    Args:
        conn_edges (np.ndarray): 2D np.ndarray with (em, 2) int entries of edges from the graph capturing the periodic connections between the core nodes.
        conn_edges_type (np.ndarray): 1D np.ndarray with em float entries of the type label for the edges from the graph capturing the periodic connections between the core nodes. Core edges are of type 1, and periodic boundary edges are of type 2.
        l_cntr_conn_edges (np.ndarray): 1D np.ndarray with em float entries of the contour length of the edges from the graph capturing the periodic connections between the core nodes.
        coords (np.ndarray): 2D np.ndarray with (en, dim) float entries of the coordinates of the core nodes.
        L (np.ndarray): 1D np.ndarray with dim float entries of the tessellation scaling (i.e., simulation box side lengths).
    
    Returns:
        np.ndarray: 1D np.ndarray with em float entries of chain/edge
        stretches.
    
    """
    # Calculate Euclidean edge lengths
    l = l_func(conn_edges, conn_edges_type, coords, L)

    # Calculate and return chain/edge stretch
    return l / l_cntr_conn_edges

def gamma_inv_func(
        conn_edges: np.ndarray,
        conn_edges_type: np.ndarray,
        l_cntr_conn_edges: np.ndarray,
        coords: np.ndarray,
        L: np.ndarray) -> np.ndarray:
    """Inverse chain/edge stretches.

    This function calculates the inverse chain/edge stretch of each
    supplied edge.

    Args:
        conn_edges (np.ndarray): 2D np.ndarray with (em, 2) int entries of edges from the graph capturing the periodic connections between the core nodes.
        conn_edges_type (np.ndarray): 1D np.ndarray with em float entries of the type label for the edges from the graph capturing the periodic connections between the core nodes. Core edges are of type 1, and periodic boundary edges are of type 2.
        l_cntr_conn_edges (np.ndarray): 1D np.ndarray with em float entries of the contour length of the edges from the graph capturing the periodic connections between the core nodes.
        coords (np.ndarray): 2D np.ndarray with (en, dim) float entries of the coordinates of the core nodes.
        L (np.ndarray): 1D np.ndarray with dim float entries of the tessellation scaling (i.e., simulation box side lengths).
    
    Returns:
        np.ndarray: 1D np.ndarray with em float entries of inverse
        chain/edge stretches.
    
    """
    # Calculate chain/edge stretches
    gamma = gamma_func(conn_edges, conn_edges_type, l_cntr_conn_edges, coords, L)

    # Calculate and return inverse chain/edge stretches (where the
    # inverse chain/edge stretches for self-loops is set to zero)
    return np.reciprocal(gamma, where=gamma!=0.0)

def k_func(conn_edges: np.ndarray) -> np.ndarray:
    """Node degree.

    This function calculates the node degree for a graph.

    Args:
        conn_edges (np.ndarray): 2D np.ndarray with (em, 2) int entries of edges from the graph capturing the periodic connections between the core nodes.
    
    Returns:
        np.ndarray: 1D np.ndarray with en int entries of the degree of
        each node.

    """
    k = np.zeros(np.max(conn_edges)+1, dtype=int)

    for edge in range(np.shape(conn_edges)[0]):
        # Node numbers
        core_node_0 = int(conn_edges[edge, 0])
        core_node_1 = int(conn_edges[edge, 1])

        k[core_node_0] += 1
        # Each self-loop only adds an entry of 1 to the node degree
        if core_node_0 != core_node_1: k[core_node_1] += 1
    
    return k