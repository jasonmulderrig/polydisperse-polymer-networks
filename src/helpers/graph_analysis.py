import numpy as np
import numpy.typing as npt

def lexsorted_edges(
        edges: npt.NDArray[np.integer]) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.integer]]:
    """Lexicographically sorted edges.

    This function takes an array of (A, B) nodes specifying edges and
    lexicographically sorts the edge entries.

    Args:
        edges (npt.NDArray[np.integer]): 2D array with (em, 2) int entries of edges.
    
    Returns:
        tuple[npt.NDArray[np.integer], npt.NDArray[np.integer]]: 2D
        array with (em, 2) int entries of lexicographically sorted edges
        and the 1D array of em int entries of the indices of the
        lexicographic edge sort.
    
    """
    edges = np.sort(edges, axis=1)
    lexsort_indcs = np.lexsort((edges[:, 1], edges[:, 0]))
    return edges[lexsort_indcs], lexsort_indcs

def edges_to_A_func(edges: npt.NDArray[np.integer]) -> npt.NDArray[np.integer]:
    """Adjacency matrix corresponding to an array of edges.

    This function takes an array of (A, B) nodes specifying edges and
    generates the corresponding adjacency matrix.

    Args:
        edges (npt.NDArray[np.integer]): 2D array with (em, 2) int entries of edges.
    
    Returns:
        npt.NDArray[np.integer]: 2D array with (en, en) int entries of
        the adjacency matrix.
    
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

def A_to_edges_func(A: npt.NDArray[np.integer]) -> npt.NDArray[np.integer]:
    """Edges array corresponding to an adjacency matrix.

    This function takes an adjacency matrix, generates the corresponding
    array of (A, B) nodes specifying edges, and returns the edges in a
    lexicographically sorted fashion.

    Args:
        A (npt.NDArray[np.integer]): 2D array with (en, en) int entries of the adjacency matrix.
    
    Returns:
        npt.NDArray[np.integer]: 2D array with (em, 2) int entries of
        lexicographically sorted edges.
    
    """
    # Ensure that A is both square and symmetric
    en = np.shape(A)[0]
    if np.shape(A)[1] != en: raise ValueError("A is not a square matrix!")
    if not np.array_equal(A, np.transpose(A)):
        raise ValueError("A is not symmetric!")

    # Initialize diagonal and upper-triangular parts of adjacency matrix
    row_indcs, col_indcs = np.triu_indices(en, k=0)
    num_indcs = np.sum(np.arange(en+1))
    
    # Calculate number of edges in the diagonal and upper-triangular
    # parts of adjacency matrix, and initialize edge list
    em = np.sum(A[row_indcs, col_indcs])
    edges = np.empty((em, 2), dtype=int)

    # Populate edge list
    edge = 0
    for indx in range(num_indcs):
        node_0 = int(row_indcs[indx])
        node_1 = int(col_indcs[indx])
        edge_counts = A[node_0, node_1]
        if edge_counts == 0: continue
        else:
            for _ in range(edge_counts):
                edges[edge, 0] = node_0
                edges[edge, 1] = node_1
                edge += 1
    
    # Gather and return lexicographically sorted edges
    edges, _ = lexsorted_edges(edges)
    return edges

def edges_and_edges_attr_to_sparse_A_arr_and_sparse_A_attr_arr(
        en: int,
        k_max: int,
        edges: npt.NDArray[np.integer],
        binary_edges_type: npt.NDArray[np.integer] | None = None,
        edges_attr: npt.NDArray[np.floating] | None = None,
        symmtry: bool = True) -> npt.NDArray[np.integer] | tuple[npt.NDArray[np.integer], npt.NDArray[np.floating]] | None:
    """Sparse adjacency array and corresponding sparse adjacency edge
    attributes array converted from edges, binary edge type, and edge
    attributes arrays.

    First and foremost, this function generates a sparse adjacency array
    converted from an edges array. The sparse adjacency array is an
    en-by-k_max array. In this array, nodes are assumed to be numbered
    from 1 to en, and each node could possibly have a maximal degree of
    k_max. An entry "val" in the ith row corresponds to the edge
    (i+1, val). Do note that in the edges array, the nodes are assumed
    to be numbered from 0 to en-1. Node numbers are correctly converted
    between each representation as the sparse adjacency array is
    constructed from the edges array. Also note that the rows in the
    sparse adjacency array are zero-indexed, and referred to as such.
    All zero entries in the sparse adjacency array corresponds the
    absence of an edge.

    The sparse adjacency array is able to naturally represent a binary
    edge type within its structure. To account for this, a binary edge
    type array (corresponding to the edges array) can be optionally
    provided. Here, each edge is assigned either a type 1 or type 2. In
    the sparse adjacency array, an entry "val" in the ith row
    corresponds to a type 1 edge (i+1, val), and an entry "-1*val" in
    the ith row corresponds to a type 2 edge (i+1, val).

    Moreover, this function is able to construct an analogous sparse
    adjacency array containing corresponding edge attributes. To execute
    this functionality, an edge attributes array (corresponding to the
    edges array) can be optionally provided. Each sparse adjacency edge
    attributes array entry thus stores the attribute of the edge in the
    corresponding sparse adjacency array entry.

    By default, this function generates a symmetric sparse adjacency
    array. However, if called for, a sparse adjacency array that
    corresponds to the upper-triangular part of its analogous adjacency
    matrix representation is able to be created. This upper-triangular
    sparse adjacency array is initialized by addressing each edge
    provided in the edges array, without any additional considerations.
    If a sparse adjacency edge attributes array is also called for, then
    it will be created in a corresponding symmetric or upper-triangular
    fashion as well.

    Args:
        en (int): Number of nodes.
        k_max (int): Maximal degree of any given node.
        edges (npt.NDArray[np.integer]): 2D array with (em, 2) int entries of edges.
        binary_edges_type (npt.NDArray[np.integer] | None): 1D array with em int entries of binary edge type labels. Labels can be either 1 or 2.
        edges_attr (npt.NDArray[np.floating] | None): 1D array with em float entries of edge attributes.
        symmtry (bool): Boolean flag to create either a symmetric (if True) or an upper-triangular (if False) sparse adjacency array. If called for, a corresponding symmetric or upper-triangular sparse adjacency edge attributes array will also be created.
    
    Returns:
        npt.NDArray[np.integer] | tuple[npt.NDArray[np.integer], npt.NDArray[np.floating]] | None:
        Sparse adjacency array; Sparse adjacency array and corresponding
        sparse adjacency edge attributes array.
    
    """
    # Make copies of the supplied arrays
    edges_copy = np.atleast_2d(edges.copy())
    if binary_edges_type is not None:
        binary_edges_type_copy = np.atleast_1d(binary_edges_type.copy())
    if edges_attr is not None:
        edges_attr_copy = np.atleast_1d(edges_attr.copy())
    
    # Confirm that the nodes in the provided network are within the
    # specified number of nodes
    edges_en = np.max(edges_copy)
    if edges_en > en-1:
        error_str = (
            "The largest node number in the provided edge array is "
            + "greater than the specified number of nodes."
        )
        raise ValueError(error_str)
    
    # Confirm that the provided network does not violate the maximal
    # node degree
    em = np.shape(edges_copy)[0]
    k = np.zeros(en, dtype=int)
    for edge in range(em):
        k[int(edges_copy[edge, 0])] += 1
        k[int(edges_copy[edge, 1])] += 1
    if np.max(k) > k_max:
        error_str = (
            "The maximal degree in the provided network is greater "
            + "than the specified maximal degree."
        )
        raise ValueError(error_str)
    
    # If provided, confirm that the binary edge type and edge attributes
    # arrays have the same number of edge entries as the edges array
    if binary_edges_type is not None:
        if np.shape(edges_copy)[0] != np.shape(binary_edges_type_copy)[0]:
            error_str = (
                "The number of edges represented in the binary edge "
                + "type array is not equal to the number of edges in "
                + "the edges array."
            )
            raise ValueError(error_str)
    if edges_attr is not None:
        if np.shape(edges_copy)[0] != np.shape(edges_attr_copy)[0]:
            error_str = (
                "The number of edges represented in the edge "
                + "attributes array is not equal to the number of "
                + "edges in the edges array."
            )
            raise ValueError(error_str)
    
    # If provided, confirm that the entries in the binary edge type
    # array are only 1 or 2. After that, convert entries of 2 to -1
    if binary_edges_type is not None:
        if np.any(np.logical_not(np.logical_or(binary_edges_type_copy==1, binary_edges_type_copy==2))):
            error_str = (
                "There exists at least one entry in the binary edge "
                + "type array that is not equal to either 1 or 2."
            )
            raise ValueError(error_str)
        else: binary_edges_type_copy[binary_edges_type_copy==2] = -1
    
    # Lexicographically sort edges. If provided, lexicographically sort
    # the binary edge types and edge attributes.
    edges_copy, lexsort_indcs = lexsorted_edges(edges_copy)
    if binary_edges_type is not None:
        binary_edges_type_copy = binary_edges_type_copy[lexsort_indcs]
    if edges_attr is not None:
        edges_attr_copy = edges_attr_copy[lexsort_indcs]
    
    # Initialize the sparse adjacency array. If provided, also
    # initialize the sparse adjacency edge attributes array.
    sparse_A_arr = np.zeros((en, k_max), dtype=int)
    if edges_attr is not None: sparse_A_attr_arr = np.zeros((en, k_max))

    # Directly enter each edge from the edges array in the sparse
    # adjacency array in order to create the default upper-triangular 
    # sparse adjacency array
    for edge in range(em):
        # Node numbers
        core_node_0 = int(edges_copy[edge, 0])
        core_node_1 = int(edges_copy[edge, 1])

        # Adjust neighbor node number from edges array for entry in the
        # sparse adjacency array
        sparse_A_nghbr_node = core_node_1 + 1

        # Determine appropriate column index 
        k_indx = np.where(sparse_A_arr[core_node_0]==0)[0][0]

        # Input sparse adjacency array entry. If provided, follow
        # convention to account for binary edge types
        if binary_edges_type is not None:
            sparse_A_arr[core_node_0, k_indx] = int(
                binary_edges_type_copy[edge] * sparse_A_nghbr_node)
        else: sparse_A_arr[core_node_0, k_indx] = sparse_A_nghbr_node

        # If provided, input the edge attribute to the sparse adjacency
        # edge attributes array
        if edges_attr is not None:
            sparse_A_attr_arr[core_node_0, k_indx] = edges_attr_copy[edge]
        
        # Each primary loop is entered in the sparse adjacency array
        # twice, since it occupies two connection sites
        if core_node_0 == core_node_1:
            # Determine appropriate column index 
            k_indx = np.where(sparse_A_arr[core_node_0]==0)[0][0]

            # Input sparse adjacency array entry. If provided, follow
            # convention to account for binary edge types
            if binary_edges_type is not None:
                sparse_A_arr[core_node_0, k_indx] = int(
                    binary_edges_type_copy[edge] * sparse_A_nghbr_node)
            else: sparse_A_arr[core_node_0, k_indx] = sparse_A_nghbr_node

            # If provided, input the edge attribute to the sparse
            # adjacency edge attributes array
            if edges_attr is not None:
                sparse_A_attr_arr[core_node_0, k_indx] = edges_attr_copy[edge]
    
    # If called for, symmetrize the sparse adjacency array
    if symmtry:
        for edge in range(em):
            # Node numbers
            core_node_0 = int(edges_copy[edge, 0])
            core_node_1 = int(edges_copy[edge, 1])

            # Primary loops are already accounted for
            if core_node_0 == core_node_1: continue
            else:
                # Adjust neighbor node number from edges array for
                # entry in the sparse adjacency array
                sparse_A_nghbr_node = core_node_0 + 1

                # Determine appropriate column index 
                k_indx = np.where(sparse_A_arr[core_node_1]==0)[0][0]

                # Input sparse adjacency array entry. If provided,
                # follow convention to account for binary edge types
                if binary_edges_type is not None:
                    sparse_A_arr[core_node_1, k_indx] = int(
                        binary_edges_type_copy[edge] * sparse_A_nghbr_node)
                else: sparse_A_arr[core_node_1, k_indx] = sparse_A_nghbr_node

                # If provided, input the edge attribute to the sparse
                # adjacency edge attributes array
                if edges_attr is not None:
                    sparse_A_attr_arr[core_node_1, k_indx] = edges_attr_copy[edge]
    
    # Sort the values of each row of the sparse adjacency array. If
    # provided, correspondingly sort the values of the sparse adjacency
    # edge attributes array 
    sort_indcs = np.argsort(sparse_A_arr, axis=1)
    sparse_A_arr = np.take_along_axis(sparse_A_arr, sort_indcs, axis=1)
    if edges_attr is not None:
        sparse_A_attr_arr = np.take_along_axis(
            sparse_A_attr_arr, sort_indcs, axis=1)
        return sparse_A_arr, sparse_A_attr_arr
    else: return sparse_A_arr

def sparse_A_arr_and_sparse_A_attr_arr_to_edges_and_edges_attr(
        sparse_A_arr: npt.NDArray[np.integer],
        return_binary_edges_type: bool = False,
        sparse_A_attr_arr: npt.NDArray[np.floating] | None = None) -> npt.NDArray[np.integer] | tuple[npt.NDArray[np.integer], npt.NDArray[np.integer]] | tuple[npt.NDArray[np.integer], npt.NDArray[np.integer], npt.NDArray[np.floating]] | None:
    """Edges, binary edge types, and edge attributes arrays converted
    from sparse adjacency array and corresponding sparse adjacency edge
    attributes array.
    
    First and foremost, this function generates an edges array converted
    from a sparse adjacency array. If called for, this function is also
    able to construct a binary edges type array corresponding to each
    edge in the edges array. Moreover, this function is able to
    construct a corresponding edge attributes array. To execute this
    functionality, a sparse adjacency edge attributes array
    (corresponding to the sparse adjacency array array) can be
    optionally provided.

    Args:
        sparse_A_arr (npt.NDArray[np.integer]): Sparse adjacency array.
        return_binary_edges_type (bool): Boolean flag to return the binary edge types array (if True) or not (if False).
        sparse_A_attr_arr (npt.NDArray[np.floating] | None): Sparse adjacency edge attributes array.
    
    Returns:
        npt.NDArray[np.integer] | tuple[npt.NDArray[np.integer], npt.NDArray[np.integer]] | tuple[npt.NDArray[np.integer], npt.NDArray[np.integer], npt.NDArray[np.floating]] | None:
        Edges array; Edges array and binary edge types array; Edges
        array and edge attributes array; Edges array, binary edge types
        array, and edge attributes array.
    
    """
    # Make copies of the supplied arrays
    sparse_A_arr_copy = np.atleast_2d(sparse_A_arr.copy())
    if sparse_A_attr_arr is not None:
        sparse_A_attr_arr_copy = np.atleast_2d(sparse_A_attr_arr.copy())

    # Confirm that the nodes in the provided network are within the
    # specified number of nodes, as dictated by the number of rows in
    # the provided sparse adjacency array
    sparse_A_arr_en = np.max(np.abs(sparse_A_arr_copy))
    en = np.shape(sparse_A_arr_copy)[0]
    if sparse_A_arr_en > en:
        error_str = (
            "The largest node number in the provided sparse adjacency "
            + "array is greater than the specified number of nodes, as "
            + "dictated by the number of rows in the provided sparse "
            + "adjacency array."
        )
        raise ValueError(error_str)
    
    # If provided, confirm that the sparse adjacency edge attributes
    # array has the same shape as the sparse adjacency array
    if sparse_A_attr_arr is not None:
        if np.shape(sparse_A_arr_copy) != np.shape(sparse_A_attr_arr_copy):
            error_str = (
                "The shape of the sparse adjacency edge attributes is "
                + "not the same shape as the sparse adjacency array."
            )
            raise ValueError(error_str)

    # Determine total number of edges in the sparse adjacency array
    em = 0
    for core_node_0 in range(en):
        # Node is connected to at least one other node
        if np.count_nonzero(sparse_A_arr_copy[core_node_0]) > 0:
            # Evaluate primary loops. Gather indices of connection sites
            # occupied by primary loops.
            primary_loop_k_indcs = (
                np.where(np.logical_or(sparse_A_arr_copy[core_node_0]==core_node_0+1, sparse_A_arr_copy[core_node_0]==-1*(core_node_0+1)))[0]
            )
            num_primary_loop_k_indcs = np.size(primary_loop_k_indcs)
            
            # Primary loop(s) exists
            if num_primary_loop_k_indcs > 0:
                # Assert that there is at least one primary loop at
                # hand, where each primary loop occupies two connection
                # sites
                assert (
                    num_primary_loop_k_indcs >= 2
                    and num_primary_loop_k_indcs % 2 == 0
                )

                # Remove primary loops
                sparse_A_arr_copy[core_node_0, primary_loop_k_indcs] = 0
                em += int(num_primary_loop_k_indcs/2)
            
            # Node is connected to at least one other neighbor node
            if np.count_nonzero(sparse_A_arr_copy[core_node_0]) > 0:
                # Gather and evaluate each neighbor node
                for k_indx in np.nditer(np.nonzero(sparse_A_arr_copy[core_node_0])[0]):
                    k_indx = int(k_indx)
                    sparse_A_nghbr_node = int(
                        sparse_A_arr_copy[core_node_0, k_indx])

                    # Adjust neighbor node number from sparse adjacency
                    # array for entry in the edges array
                    if sparse_A_nghbr_node > 0:
                        core_node_1 = sparse_A_nghbr_node - 1
                    else: core_node_1 = -1 * sparse_A_nghbr_node - 1

                    # Remove edge
                    sparse_A_arr_copy[core_node_0, k_indx] = 0

                    # Remove symmetric duplicate edge entry, if it
                    # exists
                    if sparse_A_nghbr_node > 0:
                        symmtrc_sparse_A_nghbr_node = core_node_0 + 1
                    else: symmtrc_sparse_A_nghbr_node = -1 * (core_node_0+1)
                    symmtrc_k_indx_arr = (
                        np.where(sparse_A_arr_copy[core_node_1]==symmtrc_sparse_A_nghbr_node)[0]
                    )
                    if np.size(symmtrc_k_indx_arr) > 0:
                        symmtrc_k_indx = symmtrc_k_indx_arr[0]
                        sparse_A_arr_copy[core_node_1, symmtrc_k_indx] = 0
                    em += 1
    
    # Make a new copy of the sparse adjacency array
    sparse_A_arr_copy = sparse_A_arr.copy()
    
    # Initialize edges array. If called for, initialize the binary edge
    # type array. If provided, initialize the edge attributes array. 
    edges = np.empty((em, 2), dtype=int)
    if return_binary_edges_type: binary_edges_type = np.empty(em, dtype=int)
    if sparse_A_attr_arr is not None: edges_attr = np.empty(em)

    # Directly enter each edge from the sparse adjacency array in the 
    # edges array
    edge = 0
    for core_node_0 in range(en):
        # Node is connected to at least one other node
        if np.count_nonzero(sparse_A_arr_copy[core_node_0]) > 0:
            # Evaluate primary loops. Gather indices of connection sites
            # occupied by primary loops.
            primary_loop_k_indcs = (
                np.where(np.logical_or(sparse_A_arr_copy[core_node_0]==core_node_0+1, sparse_A_arr_copy[core_node_0]==-1*(core_node_0+1)))[0]
            )
            num_primary_loop_k_indcs = np.size(primary_loop_k_indcs)
            
            # Primary loop(s) exists
            if num_primary_loop_k_indcs > 0:
                # Assert that there is at least one primary loop at
                # hand, where each primary loop occupies two connection
                # sites
                assert (
                    num_primary_loop_k_indcs >= 2
                    and num_primary_loop_k_indcs % 2 == 0
                )

                # Evaluate each primary loop by evaluating every other
                # index of connection sites occupied by primary loops
                # (since each primary loop is entered in the sparse
                # adjacency array twice, and each such pairing will
                # remain grouped together in this manner)
                for indx in range(0, num_primary_loop_k_indcs, 2):
                    # Store the primary loop edge in the edges array
                    edges[edge, 0] = core_node_0
                    edges[edge, 1] = core_node_0

                    # If called for, store the binary edge type
                    if return_binary_edges_type:
                        if sparse_A_arr_copy[core_node_0, primary_loop_k_indcs[indx]] > 0:
                            binary_edges_type[edge] = 1
                        else: binary_edges_type[edge] = 2

                    # If provided, store the edge attribute
                    if sparse_A_attr_arr is not None:
                        edges_attr[edge] = (
                            sparse_A_attr_arr_copy[core_node_0, primary_loop_k_indcs[indx]]
                        )
                    edge += 1
                
                # Remove primary loops
                sparse_A_arr_copy[core_node_0, primary_loop_k_indcs] = 0

                # If provided, remove the edge attribute
                if sparse_A_attr_arr is not None:
                    sparse_A_attr_arr_copy[core_node_0, primary_loop_k_indcs] = 0.
            
            # Node is connected to at least one other neighbor node
            if np.count_nonzero(sparse_A_arr_copy[core_node_0]) > 0:
                # Gather and evaluate each neighbor node
                for k_indx in np.nditer(np.nonzero(sparse_A_arr_copy[core_node_0])[0]):
                    k_indx = int(k_indx)
                    sparse_A_nghbr_node = int(
                        sparse_A_arr_copy[core_node_0, k_indx])

                    # Adjust neighbor node number from sparse adjacency
                    # array for entry in the edges array
                    if sparse_A_nghbr_node > 0:
                        core_node_1 = sparse_A_nghbr_node - 1
                    else: core_node_1 = -1 * sparse_A_nghbr_node - 1

                    # Store the edge in the edges array
                    edges[edge, 0] = core_node_0
                    edges[edge, 1] = core_node_1

                    # If called for, store the binary edge type
                    if return_binary_edges_type:
                        if sparse_A_nghbr_node > 0: binary_edges_type[edge] = 1
                        else: binary_edges_type[edge] = 2
                    
                    # If provided, store the edge attribute
                    if sparse_A_attr_arr is not None:
                        edges_attr[edge] = (
                            sparse_A_attr_arr_copy[core_node_0, k_indx]
                        )
                    
                    # Remove edge
                    sparse_A_arr_copy[core_node_0, k_indx] = 0
                    
                    # If provided, remove the edge attribute
                    if sparse_A_attr_arr is not None:
                        sparse_A_attr_arr_copy[core_node_0, k_indx] = 0.

                    # Remove symmetric duplicate edge entry, if it
                    # exists
                    if sparse_A_nghbr_node > 0:
                        symmtrc_sparse_A_nghbr_node = core_node_0 + 1
                    else: symmtrc_sparse_A_nghbr_node = -1 * (core_node_0+1)
                    symmtrc_k_indx_arr = (
                        np.where(sparse_A_arr_copy[core_node_1]==symmtrc_sparse_A_nghbr_node)[0]
                    )
                    if np.size(symmtrc_k_indx_arr) > 0:
                        symmtrc_k_indx = symmtrc_k_indx_arr[0]
                        sparse_A_arr_copy[core_node_1, symmtrc_k_indx] = 0
                        
                        # If provided, remove the edge attribute
                        if sparse_A_attr_arr is not None:
                            sparse_A_attr_arr_copy[core_node_1, symmtrc_k_indx] = (
                                0.
                            )
                    edge += 1
    
    # Lexicographically sort the edges. If provided, lexicographically
    # sort the edge attributes. If called for, lexicographically sort
    # the binary edge types.
    edges, lexsort_indcs = lexsorted_edges(edges)
    if return_binary_edges_type:
        binary_edges_type = binary_edges_type[lexsort_indcs]
    if sparse_A_attr_arr is not None:
        edges_attr = edges_attr[lexsort_indcs]
    
    # Return edges. If called for, return the binary edge types. If
    # provided, then return edge attributes. 
    if return_binary_edges_type and sparse_A_attr_arr is not None:
        return edges, binary_edges_type, edges_attr
    elif return_binary_edges_type and sparse_A_attr_arr is None:
        return edges, binary_edges_type
    elif not return_binary_edges_type and sparse_A_attr_arr is not None:
        return edges, edges_attr
    else: return edges

def A_largest_connected_component(
        A: npt.NDArray[np.integer]) -> npt.NDArray[np.integer]:
    """Largest/maximum connected component extraction via the adjacency
    matrix.

    This function isolates and returns the largest/maximum connected
    component via analyzing the adjacency matrix.

    Args:
        A (npt.NDArray[np.integer]): 2D array with (en, en) int entries of the adjacency matrix.
    
    Returns:
        npt.NDArray[np.integer]: 2D array with (en, en) int entries of
        the adjacency matrix of the largest/maximum connected component.

    """
    # Ensure that A is both square and symmetric
    en = np.shape(A)[0]
    if np.shape(A)[1] != en: raise ValueError("A is not a square matrix!")
    if not np.array_equal(A, np.transpose(A)):
        raise ValueError("A is not symmetric!")
    
    # Binary adjacency matrix
    A_bin = (A > 0).astype(int)

    # Initialize breadth-first search relevant arrays and lists
    visited = np.zeros(en, dtype=bool)
    components = []

    for start_node in range(en):
        if not visited[start_node]:
            # Initialize a breadth-first search to extract each
            # component
            queue = [start_node]
            component = []
            # Execute the breadth-first search
            while len(queue) > 0:
                node = queue.pop(0)
                if not visited[node]:
                    visited[node] = True
                    component.append(node)
                    nghbrs = np.where(A_bin[node]>0)[0]
                    if np.size(nghbrs) > 0:
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

def sparse_A_arr_largest_connected_component(
        en: int,
        k_max: int,
        edges: npt.NDArray[np.integer],
        binary_edges_type: npt.NDArray[np.integer] | None = None,
        edges_attr: npt.NDArray[np.floating] | None = None) -> npt.NDArray[np.integer] | tuple[npt.NDArray[np.integer], npt.NDArray[np.integer]] | tuple[npt.NDArray[np.integer], npt.NDArray[np.integer], npt.NDArray[np.floating]] | None:
    """Largest/maximum connected component extraction via the sparse
    adjacency array.

    This function isolates and returns the largest/maximum connected
    component via analyzing the sparse adjacency array.

    Args:
        en (int): Number of nodes.
        k_max (int): Maximal degree of any given node.
        edges (npt.NDArray[np.integer]): 2D array with (em, 2) int entries of edges.
        binary_edges_type (npt.NDArray[np.integer] | None): 1D array with em int entries of binary edge type labels. Labels can be either 1 or 2.
        edges_attr (npt.NDArray[np.floating] | None): 1D array with em float entries of edge attributes.
    
    Returns:
        npt.NDArray[np.integer] | tuple[npt.NDArray[np.integer], npt.NDArray[np.integer]] | tuple[npt.NDArray[np.integer], npt.NDArray[np.integer], npt.NDArray[np.floating]] | None:
        Edges array; Edges array and binary edge types array; Edges
        array and edge attributes array; Edges array, binary edge types
        array, and edge attributes array.

    """
    # Gather symmetric sparse adjacency array, and if called for, the
    # sparse adjacency edge attributes array
    if edges_attr is not None:
        sparse_A_arr, sparse_A_attr_arr = (
            edges_and_edges_attr_to_sparse_A_arr_and_sparse_A_attr_arr(
                en, k_max, edges, binary_edges_type=binary_edges_type,
                edges_attr=edges_attr, symmtry=True)
        )
    else:
        sparse_A_arr = (
            edges_and_edges_attr_to_sparse_A_arr_and_sparse_A_attr_arr(
                en, k_max, edges, binary_edges_type=binary_edges_type,
                edges_attr=edges_attr, symmtry=True)
        )
    
    # Return the absolute value of all sparse adjacency array entries,
    # and correspondingly re-sort the array entries
    abs_sparse_A_arr = np.abs(sparse_A_arr)
    abs_sparse_A_arr = np.take_along_axis(
        abs_sparse_A_arr, np.argsort(abs_sparse_A_arr, axis=1), axis=1)

    # Initialize breadth-first search relevant arrays and lists
    visited = np.zeros(en, dtype=bool)
    components = []

    for start_node in range(en):
        if not visited[start_node]:
            # Initialize a breadth-first search to extract each
            # component
            queue = [start_node]
            component = []
            # Execute the breadth-first search
            while len(queue) > 0:
                node = queue.pop(0)
                if not visited[node]:
                    visited[node] = True
                    component.append(node)
                    if np.count_nonzero(abs_sparse_A_arr[node]) > 0:
                        for k_indx in np.nditer(np.nonzero(abs_sparse_A_arr[node])[0]):
                            k_indx = int(k_indx)
                            nghbr = int(abs_sparse_A_arr[node, k_indx]-1)
                            if not visited[nghbr]: queue.append(nghbr)
            
            # Save extracted component
            components.append(component)
    
    # Extract the nodes of the largest/maximum connected component
    lcc_nodes = np.sort(np.asarray(max(components, key=len)))

    # Extract the sparse adjacency array of the largest connected
    # component. If called for, extract the sparse adjacency edge
    # attributes array of the largest connected component.
    sparse_A_arr_lcc = np.zeros_like(sparse_A_arr)
    sparse_A_arr_lcc[lcc_nodes] = sparse_A_arr[lcc_nodes]
    if edges_attr is not None:
        sparse_A_attr_arr_lcc = np.zeros_like(sparse_A_attr_arr)
        sparse_A_attr_arr_lcc[lcc_nodes] = sparse_A_attr_arr[lcc_nodes]

    # Gather edges array. If called for, gather the binary edge types
    # array. In addition, if called for, gather the edge attributes
    # array.
    if binary_edges_type is not None and edges_attr is not None:
        return (
            sparse_A_arr_and_sparse_A_attr_arr_to_edges_and_edges_attr(
                sparse_A_arr_lcc, return_binary_edges_type=True,
                sparse_A_attr_arr=sparse_A_attr_arr_lcc)
        )
    elif binary_edges_type is not None and edges_attr is None:
        return (
            sparse_A_arr_and_sparse_A_attr_arr_to_edges_and_edges_attr(
                sparse_A_arr_lcc, return_binary_edges_type=True,
                sparse_A_attr_arr=None)
        )
    elif binary_edges_type is None and edges_attr is not None:
        return (
            sparse_A_arr_and_sparse_A_attr_arr_to_edges_and_edges_attr(
                sparse_A_arr_lcc, return_binary_edges_type=False,
                sparse_A_attr_arr=sparse_A_attr_arr_lcc)
        )
    else:
        return (
            sparse_A_arr_and_sparse_A_attr_arr_to_edges_and_edges_attr(
                sparse_A_arr_lcc, return_binary_edges_type=False,
                sparse_A_attr_arr=None)
        )