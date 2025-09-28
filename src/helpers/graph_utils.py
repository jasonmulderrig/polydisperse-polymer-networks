import numpy as np

def lexsorted_edges(edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lexicographically sorted edges.

    This function takes an np.ndarray of (A, B) nodes specifying edges
    and lexicographically sorts the edge entries.

    Args:
        edges (np.ndarray): 2D np.ndarray with (em, 2) int entries of edges.
    
    Returns:
        tuple[np.ndarray, np.ndarray]: 2D np.ndarray with (em, 2) int
        entries of lexicographically sorted edges and the 1D np.ndarray
        of em int entries of the indices of the lexicographic edge sort.
    
    """
    edges = np.sort(edges, axis=1)
    lexsort_indcs = np.lexsort((edges[:, 1], edges[:, 0]))
    return edges[lexsort_indcs], lexsort_indcs

def unique_lexsorted_edges(edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unique lexicographically sorted edges.

    This function takes an np.ndarray of (A, B) nodes specifying edges,
    lexicographically sorts the edge entries, and extracts the resulting
    unique edges.

    Args:
        edges (np.ndarray): 2D np.ndarray with (em, 2) int entries of edges.
    
    Returns:
        tuple[np.ndarray, np.ndarray]: 2D np.ndarray with
        (em_unique_count, 2) int entries of unique lexicographically
        sorted edges and the 1D np.ndarray with em_unique_count int
        entries of the counts of each unique edge.
    
    """
    edges, _ = lexsorted_edges(edges)
    return np.unique(edges, axis=0, return_counts=True)

def edges_to_A_func(edges: np.ndarray) -> np.ndarray:
    """Adjacency matrix corresponding to an array of edges.

    This function takes an np.ndarray of (A, B) nodes specifying edges
    and generates the corresponding adjacency matrix.

    Args:
        edges (np.ndarray): 2D np.ndarray with (em, 2) int entries of edges.
    
    Returns:
        np.ndarray: 2D np.ndarray with (en, en) int entries of the
        adjacency matrix.
    
    """
    # Initialize adjacency matrix
    en = np.max(edges) + 1
    A = np.zeros((en, en), dtype=int)

    # Populate adjacency matrix
    for edge in range(np.shape(edges)[0]):
        node_0 = int(edges[edge, 0])
        node_1 = int(edges[edge, 1])

        A[node_0, node_1] += 1
        # Each self-loop only adds an entry of 1 to the diagonal
        if node_0 != node_1: A[node_1, node_0] += 1
    
    return A

def A_to_edges_func(A: np.ndarray) -> np.ndarray:
    """Edges array corresponding to an adjacency matrix.

    This function takes an adjacency matrix, generates the corresponding
    np.ndarray of (A, B) nodes specifying edges, and returns the edges
    in a lexicographically sorted fashion.

    Args:
        A (np.ndarray): 2D np.ndarray with (en, en) int entries of the adjacency matrix.
    
    Returns:
        np.ndarray: 2D np.ndarray with (em, 2) int entries of
        lexicographically sorted edges.
    
    """
    # Ensure that A is both square and symmetric
    en = np.shape(A)[0]
    if np.shape(A)[1] != en: raise ValueError("A is not a square matrix!")
    if not np.allclose(A, np.transpose(A)):
        raise ValueError("A is not symmetric!")

    # Initialize diagonal and upper-triangular parts of adjacency matrix
    row_indcs, col_indcs = np.triu_indices(en, k=0)
    num_indcs = np.sum(np.arange(en+1))

    # Generate edge list
    edges_node_0 = []
    edges_node_1 = []
    for indx in range(num_indcs):
        node_0 = int(row_indcs[indx])
        node_1 = int(col_indcs[indx])
        edge_counts = A[node_0, node_1]
        if edge_counts == 0: continue
        else:
            for _ in range(edge_counts):
                edges_node_0.append(node_0)
                edges_node_1.append(node_1)
    
    edges = np.stack(
        (np.asarray(edges_node_0, dtype=int), np.asarray(edges_node_1, dtype=int)),
        axis=-1)
    
    # Gather and return lexicographically sorted edges
    edges, _ = lexsorted_edges(edges)
    return edges

def largest_connected_component(A: np.ndarray) -> np.ndarray:
    """Largest/maximum connected component.

    This function isolates and returns the largest/maximum connected
    component.

    Args:
        A (np.ndarray): 2D np.ndarray with (en, en) int entries of the adjacency matrix.
    
    Returns:
        np.ndarray: 2D np.ndarray with (en, en) int entries of the
        adjacency matrix of the largest/maximum connected component.

    """
    # Ensure that A is both square and symmetric
    en = np.shape(A)[0]
    if np.shape(A)[1] != en: raise ValueError("A is not a square matrix!")
    if not np.allclose(A, np.transpose(A)):
        raise ValueError("A is not symmetric!")
    
    # Binary adjacency matrix
    A_bin = (A > 0).astype(int)

    # Initialize breadth-first search relevant arrays and lists
    visited = np.zeros(en, dtype=bool)
    components = []

    for strt_node in range(en):
        if not visited[strt_node]:
            # Initialize a breadth-first search to extract each
            # component
            queue = [strt_node]
            component = []
            # Execute the breadth-first search
            while len(queue) > 0:
                node = queue.pop(0)
                if not visited[node]:
                    visited[node] = True
                    component.append(node)
                    nghbrs = np.where(A_bin[node] > 0)[0]
                    if np.shape(nghbrs)[0] > 0:
                        for nghbr in np.nditer(nghbrs):
                            nghbr = int(nghbr)
                            if not visited[nghbr]: queue.append(nghbr)
            
            # Save extracted component
            components.append(component)
    
    # Extract the nodes of the largest/maximum connected component
    lcc_nodes = np.sort(np.asarray(max(components, key=len)))

    # Extract the adjacency matrix of the largest connected component
    A_lcc = np.zeros_like(A)
    for node_0 in np.nditer(lcc_nodes):
        node_0 = int(node_0)
        for node_1 in np.nditer(lcc_nodes):
            node_1 = int(node_1)
            A_lcc[node_0, node_1] = A[node_0, node_1]

    return A_lcc

def yasuda_morita_procedure(
        A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Yasuda-Morita procedure.
    
    This function applies the Yasuda-Morita procedure to yield the
    elastically-effective network satisfying the Scanlan-Case criteria.

    Args:
        A (np.ndarray): 2D np.ndarray with (en, en) int entries of the adjacency matrix (with no multiedges).
    
    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 2D
        np.ndarray with (en, en) int entries of the adjacency matrix of
        the elastically-effective network (with no multiedges), 2D
        np.ndarray with (bridge_center_num, 3) entries of nodes involved
        with removed bridge centers, 2D np.ndarray with
        (dangling_edges_num, 2) entries of nodes involved with removed
        dangling edges, and 1D np.ndarray with self_loop_num entries of
        nodes involved with removed self-loops.
    
    """
    # Gather nodes
    en = np.shape(A)[0]
    nodes = np.arange(en, dtype=int)

    # Initialize lists
    bridge_center_node_list = []
    dangling_node_list = []
    loop_list = []

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
                    # Add to bridge center node list
                    bridge_center_node_list.append(
                        (root_node, center_node, root_node))
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
                    # Add to bridge center node list
                    bridge_center_node_list.append(
                        (left_node, center_node, right_node))
                break
        if bridge_center_node_elim == True:
            continue
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
                    # Add to dangling node list
                    dangling_node_list.append((root_node, dangling_node))
                    break
            if dangling_node_elim == True:
                continue
            else:
                
                # Loop elimination
                for node in range(en):
                    node = int(node)
                    if A[node, node] >= 2:
                        loop_elim = True
                        # Eliminate the loop from the network
                        A[node, node] -= 2
                        # Add to loop list
                        loop_list.append(node)
                        break
                if loop_elim == True:
                    continue
                else: break # Yasuda-Morita procedure has finished
    
    bridge_center_node_arr = np.asarray(bridge_center_node_list, dtype=int)
    dangling_node_arr = np.asarray(dangling_node_list, dtype=int)
    loop_arr = np.asarray(loop_list, dtype=int)

    return A, bridge_center_node_arr, dangling_node_arr, loop_arr

def surviving_bridge_restoration(
        A: np.ndarray,
        bridge_center_node_arr: np.ndarray) -> np.ndarray:
    """Surviving bridge restoration.

    This function adds back bridges that were once removed to yield the
    most fundamental elastically-effective network but yet still exist
    between surviving nodes in end-linked networks.

    Args:
        A (np.ndarray): 2D np.ndarray with (en, en) int entries of the adjacency matrix (with no multiedges).
        bridge_center_node_arr (np.ndarray): 2D np.ndarray with (bridge_center_num, 3) entries of nodes involved with bridge centers that were removed to yield the most fundamental elastically-effective network.
    
    Returns:
        np.ndarray: 2D np.ndarray with (en, en) int entries of the
        adjacency matrix of the elastically-effective network
        (with no multiedges) containing bridge centers in between
        surviving nodes.
    
    """
    bridge_center_num = np.shape(bridge_center_node_arr)[0]
    if bridge_center_num > 0:
        # Add back bridging centers between surviving nodes in reverse
        # elimination order
        for bridge_center in range(bridge_center_num-1, -1, -1):
            left_node = int(bridge_center_node_arr[bridge_center, 0])
            center_node = int(bridge_center_node_arr[bridge_center, 1])
            right_node = int(bridge_center_node_arr[bridge_center, 2])

            # Ignore bridging loops
            if left_node == right_node: pass
            else:
                # Add back bridging center node between surviving nodes
                if np.sum(A[left_node, :]) > 0 and np.sum(A[right_node, :]) > 0:
                    A[left_node, right_node] -= 1
                    A[right_node, left_node] -= 1
                    A[left_node, center_node] += 1
                    A[center_node, right_node] += 1
                    A[right_node, center_node] += 1
                    A[center_node, left_node] += 1
                # Ignore bridges that bridge eliminated nodes
                else: pass
    
    return A

def multiedge_restoration(
        A: np.ndarray,
        unique_edges: np.ndarray,
        unique_edges_counts: np.ndarray) -> np.ndarray:
    """Multiedge restoration.
    
    This function restores multiedges in an adjacency matrix.

    Args:
        A (np.ndarray): 2D np.ndarray with (en, en) int entries of the adjacency matrix (with no multiedges).
        unique_edges (np.ndarray): 2D np.ndarray with (em_unique_count, 2) int entries of unique edges.
        unique_edges_counts (np.ndarray): 1D np.ndarray with em_unique_count int entries of the counts/number of (multi)edges involved for each unique edge.
    
    Returns:
        np.ndarray: 2D np.ndarray with (en, en) int entries of the
        adjacency matrix with multiedges restored.
    
    """
    if np.any(unique_edges_counts>1):
        for edge in range(np.shape(unique_edges)[0]):
            edge_counts = int(unique_edges_counts[edge])
            if edge_counts == 1: continue
            else:
                # Multiedge nodes
                node_0 = int(unique_edges[edge, 0])
                node_1 = int(unique_edges[edge, 1])

                if A[node_0, node_1] == 0: continue
                else:
                    # Add back redundant edges
                    for _ in range(edge_counts-1):
                        A[node_0, node_1] += 1
                        if node_0 != node_1: A[node_1, node_0] += 1
    
    return A

def elastically_effective_graph(edges: np.ndarray) -> np.ndarray:
    """Elastically-effective graph.

    This function returns the portion of a given graph that corresponds
    to the elastically-effective network in the graph.

    Args:
        edges (np.ndarray): 2D np.ndarray with (em, 2) int entries of edges.
    
    Returns:
        np.ndarray: 2D np.ndarray with (ee_em, 2) int entries of edges
        corresponding to the elastically-effective network.
    
    """
    # Gather unique edges and unique edge counts
    unique_edges, unique_edges_counts = unique_lexsorted_edges(edges)

    # Acquire adjacency matrix with no multiedges
    A = edges_to_A_func(edges)
    A[A>1] = 1
    
    # Apply the Yasuda-Morita procedure to return the
    # elastically-effective network that satisfies the Scanlan-Case
    # criteria.
    A, _, _, _ = yasuda_morita_procedure(A)
    
    # Restore multiedges
    A = multiedge_restoration(A, unique_edges, unique_edges_counts)

    # As a hard fail-safe, isolate and return the largest/maximum
    # connected component
    return A_to_edges_func(largest_connected_component(A))

def elastically_effective_end_linked_graph(edges: np.ndarray) -> np.ndarray:
    """Elastically-effective end-linked graph.

    This function returns the portion of a given graph that corresponds
    to the elastically-effective end-linked network in the graph.

    Args:
        edges (np.ndarray): 2D np.ndarray with (em, 2) int entries of edges.
    
    Returns:
        np.ndarray: 2D np.ndarray with (eeel_em, 2) int entries of edges
        corresponding to the elastically-effective end-linked network.
    
    """
    # Gather unique edges and unique edge counts
    unique_edges, unique_edges_counts = unique_lexsorted_edges(edges)

    # Acquire adjacency matrix with no multiedges
    A = edges_to_A_func(edges)
    A[A>1] = 1
    
    # Apply the Yasuda-Morita procedure to return the
    # elastically-effective network that satisfies the Scanlan-Case
    # criteria.
    A, bridge_center_node_arr, _, _ = yasuda_morita_procedure(A)

    # Add back bridging centers between surviving nodes in reverse
    # elimination order
    A = surviving_bridge_restoration(A, bridge_center_node_arr)

    # Restore multiedges
    A = multiedge_restoration(A, unique_edges, unique_edges_counts)

    # As a hard fail-safe, isolate and return the largest/maximum
    # connected component
    return A_to_edges_func(largest_connected_component(A))