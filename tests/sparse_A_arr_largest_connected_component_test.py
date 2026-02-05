# Add current path to system path for direct execution
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# Import logging
import logging
logging.disable(logging.WARNING)

# Import modules
import numpy as np
from src.helpers.graph_utils import sparse_A_arr_largest_connected_component

if __name__ == "__main__":
    en = 8
    k_max = 3
    edges_orig = np.array([
        [0, 0],
        [0, 1],
        [1, 6],
        [2, 3],
        [3, 4],
        [5, 7],
        [5, 7],
        [6, 7]], dtype=int) # in lexicographically-sorted order
    binary_edges_type_orig = np.array([1, 2, 1, 1, 2, 1, 1, 2], dtype=int)
    edges_attr_orig = np.array([2., 4., 7., 8., 9., 6., 5., 3.])

    # Test 1
    edges_lcc = sparse_A_arr_largest_connected_component(
        en, k_max, edges_orig, binary_edges_type=None, edges_attr=None)
    edges_lcc_true = np.array([
        [0, 0],
        [0, 1],
        [1, 6],
        [5, 7],
        [5, 7],
        [6, 7]], dtype=int)
    assert np.array_equal(edges_lcc, edges_lcc_true)

    # Test 2
    edges_lcc, binary_edges_type_lcc = sparse_A_arr_largest_connected_component(
        en, k_max, edges_orig, binary_edges_type=binary_edges_type_orig,
        edges_attr=None)
    edges_lcc_true = np.array([
        [0, 0],
        [0, 1],
        [1, 6],
        [5, 7],
        [5, 7],
        [6, 7]], dtype=int)
    binary_edges_type_lcc_true = np.array([1, 2, 1, 1, 1, 2], dtype=int)
    assert np.array_equal(edges_lcc, edges_lcc_true)
    assert np.array_equal(binary_edges_type_lcc, binary_edges_type_lcc_true)

    # Test 3
    edges_lcc, edges_attr_lcc = sparse_A_arr_largest_connected_component(
        en, k_max, edges_orig, binary_edges_type=None,
        edges_attr=edges_attr_orig)
    edges_lcc_true = np.array([
        [0, 0],
        [0, 1],
        [1, 6],
        [5, 7],
        [5, 7],
        [6, 7]], dtype=int)
    edges_attr_lcc_true = np.array([2., 4., 7., 6., 5., 3.])
    assert np.array_equal(edges_lcc, edges_lcc_true)
    assert np.array_equal(edges_attr_lcc, edges_attr_lcc_true)

    # Test 4
    edges_lcc, binary_edges_type_lcc, edges_attr_lcc = sparse_A_arr_largest_connected_component(
        en, k_max, edges_orig, binary_edges_type=binary_edges_type_orig,
        edges_attr=edges_attr_orig)
    edges_lcc_true = np.array([
        [0, 0],
        [0, 1],
        [1, 6],
        [5, 7],
        [5, 7],
        [6, 7]], dtype=int)
    edges_attr_lcc_true = np.array([2., 4., 7., 6., 5., 3.])
    binary_edges_type_lcc_true = np.array([1, 2, 1, 1, 1, 2], dtype=int)
    assert np.array_equal(edges_lcc, edges_lcc_true)
    assert np.array_equal(edges_attr_lcc, edges_attr_lcc_true)
    assert np.array_equal(binary_edges_type_lcc, binary_edges_type_lcc_true)