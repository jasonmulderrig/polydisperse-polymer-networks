# Add current path to system path for direct execution
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# Import logging
import logging
logging.disable(logging.WARNING)

# Import modules
import numpy as np
from src.helpers.graph_utils import (
    edges_and_edges_attr_to_sparse_A_arr_and_sparse_A_attr_arr,
    sparse_A_arr_and_sparse_A_attr_arr_to_edges_and_edges_attr
)

if __name__ == "__main__":
    en = 5
    k_max = 3
    edges_orig = np.array([
        [0, 0],
        [0, 1],
        [1, 3],
        [2, 4],
        [2, 4],
        [3, 4]], dtype=int) # in lexicographically-sorted order
    binary_edges_type_orig = np.array([1, 2, 1, 1, 1, 2], dtype=int)
    edges_attr_orig = np.array([2., 4., 7., 6., 5., 3.])

    # Test 1
    sparse_A_arr = edges_and_edges_attr_to_sparse_A_arr_and_sparse_A_attr_arr(
        en, k_max, edges_orig, binary_edges_type=None, edges_attr=None,
        symmtry=True)
    sparse_A_arr_true = np.array([
        [1, 1, 2],
        [0, 1, 4],
        [0, 5, 5],
        [0, 2, 5],
        [3, 3, 4]], dtype=int)
    assert np.array_equal(sparse_A_arr, sparse_A_arr_true)

    # Test 2
    sparse_A_arr = edges_and_edges_attr_to_sparse_A_arr_and_sparse_A_attr_arr(
        en, k_max, edges_orig, binary_edges_type=None, edges_attr=None,
        symmtry=False)
    sparse_A_arr_true = np.array([
        [1, 1, 2],
        [0, 0, 4],
        [0, 5, 5],
        [0, 0, 5],
        [0, 0, 0]], dtype=int)
    assert np.array_equal(sparse_A_arr, sparse_A_arr_true)

    # Test 3
    sparse_A_arr = edges_and_edges_attr_to_sparse_A_arr_and_sparse_A_attr_arr(
        en, k_max, edges_orig, binary_edges_type=binary_edges_type_orig,
        edges_attr=None, symmtry=True)
    sparse_A_arr_true = np.array([
        [-2, 1, 1],
        [-1, 0, 4],
        [0, 5, 5],
        [-5, 0, 2],
        [-4, 3, 3]], dtype=int)
    assert np.array_equal(sparse_A_arr, sparse_A_arr_true)

    # Test 4
    sparse_A_arr = edges_and_edges_attr_to_sparse_A_arr_and_sparse_A_attr_arr(
        en, k_max, edges_orig, binary_edges_type=binary_edges_type_orig,
        edges_attr=None, symmtry=False)
    sparse_A_arr_true = np.array([
        [-2, 1, 1],
        [0, 0, 4],
        [0, 5, 5],
        [-5, 0, 0],
        [0, 0, 0]], dtype=int)
    assert np.array_equal(sparse_A_arr, sparse_A_arr_true)

    # Test 5
    sparse_A_arr, sparse_A_attr_arr = edges_and_edges_attr_to_sparse_A_arr_and_sparse_A_attr_arr(
        en, k_max, edges_orig, binary_edges_type=None,
        edges_attr=edges_attr_orig, symmtry=True)
    sparse_A_arr_true = np.array([
        [1, 1, 2],
        [0, 1, 4],
        [0, 5, 5],
        [0, 2, 5],
        [3, 3, 4]], dtype=int)
    sparse_A_attr_arr_true = np.array([
        [2., 2., 4.],
        [0., 4., 7.],
        [0., 6., 5.],
        [0., 7., 3.],
        [6., 5., 3.]
    ])
    assert np.array_equal(sparse_A_arr, sparse_A_arr_true)
    assert np.array_equal(sparse_A_attr_arr, sparse_A_attr_arr_true)

    # Test 6
    sparse_A_arr, sparse_A_attr_arr = edges_and_edges_attr_to_sparse_A_arr_and_sparse_A_attr_arr(
        en, k_max, edges_orig, binary_edges_type=None,
        edges_attr=edges_attr_orig, symmtry=False)
    sparse_A_arr_true = np.array([
        [1, 1, 2],
        [0, 0, 4],
        [0, 5, 5],
        [0, 0, 5],
        [0, 0, 0]], dtype=int)
    sparse_A_attr_arr_true = np.array([
        [2., 2., 4.],
        [0., 0., 7.],
        [0., 6., 5.],
        [0., 0., 3.],
        [0., 0., 0.]
    ])
    assert np.array_equal(sparse_A_arr, sparse_A_arr_true)
    assert np.array_equal(sparse_A_attr_arr, sparse_A_attr_arr_true)

    # Test 7
    sparse_A_arr, sparse_A_attr_arr = edges_and_edges_attr_to_sparse_A_arr_and_sparse_A_attr_arr(
        en, k_max, edges_orig, binary_edges_type=binary_edges_type_orig,
        edges_attr=edges_attr_orig, symmtry=True)
    sparse_A_arr_true = np.array([
        [-2, 1, 1],
        [-1, 0, 4],
        [0, 5, 5],
        [-5, 0, 2],
        [-4, 3, 3]], dtype=int)
    sparse_A_attr_arr_true = np.array([
        [4., 2., 2.],
        [4., 0., 7.],
        [0., 6., 5.],
        [3., 0., 7.],
        [3., 6., 5.]
    ])
    assert np.array_equal(sparse_A_arr, sparse_A_arr_true)
    assert np.array_equal(sparse_A_attr_arr, sparse_A_attr_arr_true)

    # Test 8
    sparse_A_arr, sparse_A_attr_arr = edges_and_edges_attr_to_sparse_A_arr_and_sparse_A_attr_arr(
        en, k_max, edges_orig, binary_edges_type=binary_edges_type_orig,
        edges_attr=edges_attr_orig, symmtry=False)
    sparse_A_arr_true = np.array([
        [-2, 1, 1],
        [0, 0, 4],
        [0, 5, 5],
        [-5, 0, 0],
        [0, 0, 0]], dtype=int)
    sparse_A_attr_arr_true = np.array([
        [4., 2., 2.],
        [0., 0., 7.],
        [0., 6., 5.],
        [3., 0., 0.],
        [0., 0., 0.]
    ])
    assert np.array_equal(sparse_A_arr, sparse_A_arr_true)
    assert np.array_equal(sparse_A_attr_arr, sparse_A_attr_arr_true)

    # Test 9
    sparse_A_arr = edges_and_edges_attr_to_sparse_A_arr_and_sparse_A_attr_arr(
        en, k_max, edges_orig, binary_edges_type=None, edges_attr=None,
        symmtry=True)
    edges = sparse_A_arr_and_sparse_A_attr_arr_to_edges_and_edges_attr(
        sparse_A_arr, return_binary_edges_type=False, sparse_A_attr_arr=None)
    assert np.array_equal(edges, edges_orig)

    # Test 10
    sparse_A_arr = edges_and_edges_attr_to_sparse_A_arr_and_sparse_A_attr_arr(
        en, k_max, edges_orig, binary_edges_type=None, edges_attr=None,
        symmtry=False)
    edges = sparse_A_arr_and_sparse_A_attr_arr_to_edges_and_edges_attr(
        sparse_A_arr, return_binary_edges_type=False, sparse_A_attr_arr=None)
    assert np.array_equal(edges, edges_orig)

    # Test 11
    sparse_A_arr = edges_and_edges_attr_to_sparse_A_arr_and_sparse_A_attr_arr(
        en, k_max, edges_orig, binary_edges_type=binary_edges_type_orig,
        edges_attr=None, symmtry=True)
    edges, binary_edges_type = sparse_A_arr_and_sparse_A_attr_arr_to_edges_and_edges_attr(
        sparse_A_arr, return_binary_edges_type=True, sparse_A_attr_arr=None)
    assert np.array_equal(edges, edges_orig)
    assert np.array_equal(binary_edges_type, binary_edges_type_orig)

    # Test 12
    sparse_A_arr = edges_and_edges_attr_to_sparse_A_arr_and_sparse_A_attr_arr(
        en, k_max, edges_orig, binary_edges_type=binary_edges_type_orig,
        edges_attr=None, symmtry=False)
    edges, binary_edges_type = sparse_A_arr_and_sparse_A_attr_arr_to_edges_and_edges_attr(
        sparse_A_arr, return_binary_edges_type=True, sparse_A_attr_arr=None)
    assert np.array_equal(edges, edges_orig)
    assert np.array_equal(binary_edges_type, binary_edges_type_orig)

    # Test 13
    sparse_A_arr, sparse_A_attr_arr = edges_and_edges_attr_to_sparse_A_arr_and_sparse_A_attr_arr(
        en, k_max, edges_orig, binary_edges_type=None,
        edges_attr=edges_attr_orig, symmtry=True)
    edges, edges_attr = sparse_A_arr_and_sparse_A_attr_arr_to_edges_and_edges_attr(
        sparse_A_arr, return_binary_edges_type=False,
        sparse_A_attr_arr=sparse_A_attr_arr)
    assert np.array_equal(edges, edges_orig)
    assert np.array_equal(edges_attr, edges_attr_orig)

    # Test 14
    sparse_A_arr, sparse_A_attr_arr = edges_and_edges_attr_to_sparse_A_arr_and_sparse_A_attr_arr(
        en, k_max, edges_orig, binary_edges_type=None,
        edges_attr=edges_attr_orig, symmtry=False)
    edges, edges_attr = sparse_A_arr_and_sparse_A_attr_arr_to_edges_and_edges_attr(
        sparse_A_arr, return_binary_edges_type=False,
        sparse_A_attr_arr=sparse_A_attr_arr)
    assert np.array_equal(edges, edges_orig)
    assert np.array_equal(edges_attr, edges_attr_orig)

    # Test 15
    sparse_A_arr, sparse_A_attr_arr = edges_and_edges_attr_to_sparse_A_arr_and_sparse_A_attr_arr(
        en, k_max, edges_orig, binary_edges_type=binary_edges_type_orig,
        edges_attr=edges_attr_orig, symmtry=True)
    edges, binary_edges_type, edges_attr = sparse_A_arr_and_sparse_A_attr_arr_to_edges_and_edges_attr(
        sparse_A_arr, return_binary_edges_type=True,
        sparse_A_attr_arr=sparse_A_attr_arr)
    assert np.array_equal(edges, edges_orig)
    assert np.array_equal(edges_attr, edges_attr_orig)
    assert np.array_equal(binary_edges_type, binary_edges_type_orig)

    # Test 16
    sparse_A_arr, sparse_A_attr_arr = edges_and_edges_attr_to_sparse_A_arr_and_sparse_A_attr_arr(
        en, k_max, edges_orig, binary_edges_type=binary_edges_type_orig,
        edges_attr=edges_attr_orig, symmtry=False)
    edges, binary_edges_type, edges_attr = sparse_A_arr_and_sparse_A_attr_arr_to_edges_and_edges_attr(
        sparse_A_arr, return_binary_edges_type=True,
        sparse_A_attr_arr=sparse_A_attr_arr)
    assert np.array_equal(edges, edges_orig)
    assert np.array_equal(edges_attr, edges_attr_orig)
    assert np.array_equal(binary_edges_type, binary_edges_type_orig)