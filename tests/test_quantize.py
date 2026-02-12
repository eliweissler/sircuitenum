import functools
import sympy as sym
import numpy as np

from sircuitenum import quantize
from sircuitenum import qpackage_interface as pi
from sircuitenum import utils

import time
import itertools


def test__independent_from():
    # Basic: A = {[1,0], [0,1]}, B = {[1,1]}
    A = [sym.Matrix([1, 0]), sym.Matrix([0, 1])]
    B = [sym.Matrix([1, 1])]
    result = quantize._independent_from(A, B)
    assert len(result) == 1
    assert any(v == sym.Matrix([1, 0]) for v in result) or any(v == sym.Matrix([0, 1]) for v in result)

    A = [sym.Matrix([1, 1]), sym.Matrix([0, -1])]
    B = [sym.Matrix([1, 0])]
    result = quantize._independent_from(A, B)
    assert len(result) == 1
    assert any(v == sym.Matrix([1, 1]) for v in result) or any(v == sym.Matrix([0, -1]) for v in result)

    # Subset: A = {[1,0], [0,1]}, B = {[1,0]}
    A = [sym.Matrix([1, 0]), sym.Matrix([0, 1])]
    B = [sym.Matrix([1, 0])]
    result = quantize._independent_from(A, B)
    assert len(result) == 1
    assert result[0] == sym.Matrix([0, 1])

    # All in B: A = {[1,0]}, B = {[1,0]}
    A = [sym.Matrix([1, 0])]
    B = [sym.Matrix([1, 0])]
    result = quantize._independent_from(A, B)
    assert result == []

    # Empty B: A = {[1,0], [0,1]}, B = []
    A = [sym.Matrix([1, 0]), sym.Matrix([0, 1])]
    B = []
    result = quantize._independent_from(A, B)
    assert len(result) == 2
    assert any(v == sym.Matrix([1, 0]) for v in result)
    assert any(v == sym.Matrix([0, 1]) for v in result)

    # Symbolic: A = {[x,0], [0,y]}, B = {[x,y]}
    x, y = sym.symbols('x y')
    A = [sym.Matrix([x, 0]), sym.Matrix([0, y])]
    B = [sym.Matrix([x, y])]
    result = quantize._independent_from(A, B)
    assert len(result) == 1
    assert any(v == sym.Matrix([x, 0]) for v in result) or any(v == sym.Matrix([0, y]) for v in result)


def test__vec_space_overlap():

    vecs1 = [sym.Matrix([1, 0, 1]),
             sym.Matrix([1, 1, 0])]
    
    vecs2 = [sym.Matrix([2, 1, 0]),
             sym.Matrix([-1, 0, 0])]

    overlap, in_v1, in_v2 = quantize._vec_space_overlap(vecs1, vecs2, return_decomp=True)

    assert len(overlap) == 1


def test__remove_permutation_equivalent_transformations():
    """Test that identical matrices after column permutation are filtered."""
    # Simple matrices where permutation creates duplicate
    Z1 = sym.Matrix([[1, 0], [0, 1]])
    Z2 = sym.Matrix([[0, 1], [1, 0]])  # Column permutation of Z1
    
    Z_list = [Z1, Z2]
    perms = [(0, 1), (1, 0)]  # Identity and swap permutations
    
    filt_Z, idx_keep = quantize._remove_permutation_equivalent_transformations(Z_list, perms)
    
    # Should only keep one since they're equivalent under permutation
    assert len(filt_Z) == 1
    assert len(idx_keep) == 1
    assert idx_keep[0] == 0  # Should keep the first one


    a, b = sym.symbols("a b", real=True)
    Z1 = sym.Matrix([[a, 0], [0, b]])
    Z2 = sym.Matrix([[a, b], [b, a]])
    Z3 = sym.Matrix([[2*a, 0], [0, 3*b]])
    
    Z_list = [Z1, Z2, Z3]
    perms = [(0, 1), (1, 0)]
    
    filt_Z, idx_keep = quantize._remove_permutation_equivalent_transformations(Z_list, perms)
    
    # Should keep at least one, and filter duplicates if any
    assert len(filt_Z) == 3
    assert len(idx_keep) == len(filt_Z)
    # Indices should be valid
    assert all(0 <= i < 3 for i in idx_keep)


    a = sym.symbols("a", real=True)
    Z1 = sym.Matrix([[a, 0],
                     [0, a]])
    Z2 = sym.Matrix([[0, -b],
                    [-b, 0]])  # Sign flip and column permutation of Z1
    
    Z_list = [Z1, Z2]
    perms = [(0, 1), (1, 0)]
    
    filt_Z, idx_keep = quantize._remove_permutation_equivalent_transformations(Z_list, perms)
    
    # Should filter one as they're equivalent up to sign
    assert len(filt_Z) == 1
    assert len(idx_keep) == len(filt_Z)
    assert idx_keep[0] == 0  # Should keep the first one


def test__linearly_indep_rows_cols():

    x,y,z = sym.symbols("x,y,z")
    test_cases = [
        # Empty matrix
        (sym.Matrix([]), []),
        
        # Single row matrices
        (sym.Matrix([[1, 2, 3]]), [0]),
        (sym.Matrix([[0, 0, 0]]), []),
        
        # Identity matrix
        (sym.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), [0, 1, 2]),
        
        # Duplicate rows
        (sym.Matrix([[1, 2, 3], [1, 2, 3], [4, 5, 6]]), [0, 2]),
        
        # Scaled rows
        (sym.Matrix([[1, 2, 3], [2, 4, 6], [3, 5, 7]]), [0, 2]),
        
        # Linear combination
        (sym.Matrix([[1, 0, 1], [0, 1, 1], [1, 1, 2]]), [0, 1]),
        
        # Symbolic matrix
        (sym.Matrix([[x, y, z], 
                    [2*x, 2*y, 2*z], 
                    [x+1, y, z]]), [0, 2]),
        
        # Mixed entries
        (sym.Matrix([[1, x, 3], 
                [2, 2*x, 6], 
                [4, 3*x, 5]]), [0, 2]),
        
        # Zero rows interspersed
        (sym.Matrix([[1, 2, 3], [0, 0, 0], [4, 5, 6]]), [0, 2]),
        
        # Rectangular matrix with more rows than columns
        (sym.Matrix([[1, 0], [0, 1], [1, 1], [2, 0]]), [0, 1]),
        
        # Rectangular matrix with more columns than rows
        (sym.Matrix([[1, 0, 1, 2], [0, 1, 1, 3]]), [0, 1]),
        
        # Fractions
        (sym.Matrix([[sym.Rational(1,2), sym.Rational(1,3), sym.Rational(1,4)], 
                [sym.Rational(1,1), sym.Rational(2,1), sym.Rational(3,1)], 
                [sym.Rational(1,3), sym.Rational(2,3), sym.Rational(3,3)]]), [0, 1]),
    ]
    
    # Run all test cases
    for matrix, expected in test_cases:
        row_set = quantize._linearly_indep_rows(matrix)
        col_set = quantize._linearly_indep_cols(matrix.transpose())
        assert row_set == tuple(expected)
        assert col_set == tuple(expected)


def test__linearly_indep_row_col_sets():

    x,y,z = sym.symbols("x,y,z")
    test_cases = [
        # Empty matrix
        (sym.Matrix([]), [()]),
        
        # Single row matrices
        (sym.Matrix([[1, 2, 3]]), [(0,)]),
        (sym.Matrix([[0, 0, 0]]), [()]),
        
        # Identity matrix
        (sym.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), [(0, 1, 2)]),
        
        # Duplicate rows
        (sym.Matrix([[1, 2, 3], [1, 2, 3], [4, 5, 6]]), [(0, 2), (1, 2)]),
        
        # Scaled rows
        (sym.Matrix([[1, 2, 3], [2, 4, 6], [3, 5, 7]]), [(0, 2), (1, 2)]),
        
        # Linear combination
        (sym.Matrix([[1, 0, 1], [0, 1, 1], [1, 1, 2]]), [(0, 1), (0, 2), (1, 2)]),
        
        # Symbolic matrix
        (sym.Matrix([[x, y, z], 
                    [2*x, 2*y, 2*z], 
                    [x+1, y, z]]), [(0, 2), (1, 2)]),
        
        # Mixed entries
        (sym.Matrix([[1, x, 3], 
                [2, 2*x, 6], 
                [4, 3*x, 5]]), [(0, 2), (1, 2)]),
        
        # Zero rows interspersed
        (sym.Matrix([[1, 2, 3], [0, 0, 0], [4, 5, 6]]), [(0, 2)]),
        
        # Rectangular matrix with more rows than columns
        (sym.Matrix([[1, 0], [0, 1], [1, 1], [2, 0]]), [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]),
        
        # Rectangular matrix with more columns than rows
        (sym.Matrix([[1, 0, 1, 2], [0, 1, 1, 3]]), [(0, 1)]),
        
        # Fractions
        (sym.Matrix([[sym.Rational(1,2), sym.Rational(1,3), sym.Rational(1,4)], 
                [sym.Rational(1,1), sym.Rational(2,1), sym.Rational(3,1)], 
                [sym.Rational(1,3), sym.Rational(2,3), sym.Rational(3,3)]]), [(0, 1), (0, 2)]),
    ]
    
    # Run all test cases
    for matrix, expected in test_cases:
        row_sets = quantize._linearly_indep_row_sets(matrix)
        col_sets = quantize._linearly_indep_col_sets(matrix.transpose())
        assert len(col_sets) == len(expected)
        assert len(row_sets) == len(expected)
        assert all(x in expected for x in row_sets)
        assert all(x in expected for x in col_sets)


def test__det_fast():

    for i in range(10):
        mat = sym.randMatrix(i, i, min=0, max=5)
        assert mat.det() == quantize._det_fast(mat)

def test__equiv_cols():

    c1 = sym.Matrix([[1], [1]])
    c2 = sym.Matrix([[1], [-1]])
    assert quantize._equiv_cols(c1, c2) == False

    c1 = sym.Matrix([[1], [0]])
    c2 = sym.Matrix([[0], [1]])
    assert quantize._equiv_cols(c1, c2) == True

    c1 = sym.Matrix([[1], [0], [0]])
    c2 = sym.Matrix([[0], [1], [-1]])
    assert quantize._equiv_cols(c1, c2) == False


    c1 = sym.Matrix([[1], [0], [0]])
    c2 = sym.Matrix([[0], [1], [-1]])
    shift = sym.Matrix([[1], [-1], [1]])*100
    assert quantize._equiv_cols(c1, c2, shifts=[shift]) == True

def test__equal_up_to_column_shift_and_sign():

    shift = sym.ones(2, 1)

    # Shift no columns
    matrix1 = sym.Matrix([[1, 2], [3, 4]])
    matrix2 = sym.Matrix([[1, 2], [3, 4]])
    assert quantize._equal_up_to_column_shift_and_sign(matrix1, matrix2) == True
    
    # Shift one column
    matrix1 = sym.Matrix([[1, 2], [3, 4]])
    matrix2 = sym.Matrix([[1, 2], [3, 4]])
    matrix2[:, 0] += shift
    assert quantize._equal_up_to_column_shift_and_sign(matrix1, matrix2) == True
    
    # Shift both columns
    matrix1 = sym.Matrix([[1, 2], [3, 4]])
    matrix2 = sym.Matrix([[1, 2], [3, 4]])
    matrix2[:, 0] += shift
    matrix2[:, 1] += -1*shift
    assert quantize._equal_up_to_column_shift_and_sign(matrix1, matrix2) == True

    # scale a column
    matrix1 = sym.Matrix([[1, 2], [3, 4]])
    matrix2 = sym.Matrix([[1, 2], [3, 4]])
    matrix2[:, 0]*=2
    assert quantize._equal_up_to_column_shift_and_sign(matrix1, matrix2) == False
    
    # With symbolic entries
    x, y = sym.symbols('x y', positive=True)
    matrix1 = sym.Matrix([[x, 2], [y, 4]])
    matrix2 = sym.Matrix([[x-y, 2], [0, 4]])
    assert quantize._equal_up_to_column_shift_and_sign(matrix1, matrix2) == True
    
    # With fractions
    matrix1 = sym.Matrix([[sym.Rational(1,2), sym.Rational(2,1)], [sym.Rational(3,4), sym.Rational(4,1)]])
    matrix2 = sym.Matrix([[-sym.Rational(1,2), sym.Rational(2,1)], [-sym.Rational(1,4), sym.Rational(4,1)]])
    assert quantize._equal_up_to_column_shift_and_sign(matrix1, matrix2) == True
    
    # Different columns with symbols, first one is *-1 -> shift away
    matrix1 = sym.Matrix([[x, 2], [y, 4]])
    matrix2 = sym.Matrix([[y, 2], [x, 4]])
    assert quantize._equal_up_to_column_shift_and_sign(matrix1, matrix2) == True

    # Different columns with symbols
    matrix1 = sym.Matrix([[x+y, 2], [y, 4]])
    matrix2 = sym.Matrix([[y, 2], [x, 4]])
    assert quantize._equal_up_to_column_shift_and_sign(matrix1, matrix2) == False

    # Identical columns
    matrix1 = sym.Matrix([[1, 2], [3, 4]])
    matrix2 = sym.Matrix([[1, 2], [3, 4]])
    assert quantize._equal_up_to_column_shift_and_sign(matrix1, matrix2) == True
    
    # Negative of column
    matrix1 = sym.Matrix([[1, 2], [3, 4]])
    matrix2 = sym.Matrix([[-1, 2], [-3, 4]])
    assert quantize._equal_up_to_column_shift_and_sign(matrix1, matrix2) == True
    
    # With symbolic entries
    x, y = sym.symbols('x y')
    matrix1 = sym.Matrix([[x, 2], [y, 4]])
    matrix2 = sym.Matrix([[-x, 2], [-y, 4]])
    assert quantize._equal_up_to_column_shift_and_sign(matrix1, matrix2) == True
    
    # With fractions
    matrix1 = sym.Matrix([[sym.Rational(1,2), sym.Rational(2,1)], [sym.Rational(3,4), sym.Rational(4,1)]])
    matrix2 = sym.Matrix([[-sym.Rational(1,2), sym.Rational(2,1)], [-sym.Rational(3,4), sym.Rational(4,1)]])
    assert quantize._equal_up_to_column_shift_and_sign(matrix1, matrix2) == True


def test__find_equiv_cols():

    shift = sym.ones(2, 1)

    # Do nothing
    matrix1 = sym.Matrix([[1], [0]])
    matrix2 = matrix1
    assert quantize._find_equiv_cols(matrix1, [matrix2, shift, matrix2+shift], shifts=[shift]) == [0, 2]

    # just shift
    matrix1 = sym.Matrix([[1], [0]])
    matrix2 = matrix1 - shift
    assert quantize._find_equiv_cols(matrix1, [matrix2, shift], shifts=[shift]) == [0]

    # shift and sign
    matrix1 = sym.Matrix([[1], [0]])
    matrix2 = -matrix1 + shift
    assert quantize._find_equiv_cols(matrix1, [shift, matrix2], shifts=[shift]) == [1]

    # shift and sign and scale
    matrix1 = sym.Matrix([[1], [0]])
    matrix2 = -2*matrix1 + shift
    assert quantize._find_equiv_cols(matrix1, [matrix2], shifts=[shift]) == []

    # different shift
    matrix1 = sym.Matrix([[1], [0]])
    shift = sym.Matrix([[0], [1]])
    matrix2 = -matrix1 + shift
    assert quantize._find_equiv_cols(matrix1, [matrix2], shifts=[shift]) == [0]

    # Failed for some reason
    matrix1 = sym.simplify(sym.Matrix([-1/2, -1/2, 1/2, 1/2]), rational=True)
    matrix2 = sym.simplify(sym.Matrix([1/2, 1/2, -1/2, -1/2]), rational=True)
    assert quantize._find_equiv_cols(matrix1, [matrix2]) == [0]

    matrix1 = sym.simplify(sym.Matrix([-1/2, -1/2, 1/2, 1/2]), rational=True)
    matrix2 = sym.simplify(sym.Matrix([1/2, 0, -1/2, -1/2]), rational=True)
    assert quantize._find_equiv_cols(matrix1, [matrix2]) == []


def test__find_equiv_mats():

    shift = sym.ones(2, 1)

    # Do nothing
    matrix1 = sym.Matrix([[1], [0]])
    matrix2 = matrix1
    matrix3 = sym.Matrix([[1], [-1]])
    matrix4 = -matrix3
    assert quantize._find_equiv_mats(sym.Matrix.hstack(matrix1, matrix2), [sym.Matrix.hstack(matrix3, matrix4)], shifts=[shift]) == []
    assert quantize._find_equiv_mats(sym.Matrix.hstack(matrix1, matrix3), [sym.Matrix.hstack(matrix3, matrix4),
                                                                    sym.Matrix.hstack(matrix4, matrix1)], shifts=[shift]) == [1]


def test__equal_up_to_column_swaps_and_shift_and_sign():

    shift = sym.ones(2, 1)

    # Shift no columns
    matrix1 = sym.Matrix([[1, 1], [1, -1]])
    matrix2 = matrix1[:, ::-1]
    assert quantize._equal_up_to_column_swaps_and_shift_and_sign(matrix1, matrix2) == True
    
    # Shift one column
    matrix1 = sym.Matrix([[1, 1], [1, -1]])
    matrix2 = matrix1[:, ::-1]
    matrix2[:, 0] += shift
    assert quantize._equal_up_to_column_swaps_and_shift_and_sign(matrix1, matrix2) == True
    
    # Shift both columns
    matrix1 = sym.Matrix([[1, 1], [1, -1]])
    matrix2 = matrix1[:, ::-1]
    matrix2[:, 0] += shift
    matrix2[:, 1] += -1*shift
    assert quantize._equal_up_to_column_swaps_and_shift_and_sign(matrix1, matrix2) == True

    # scale a column
    matrix1 = sym.Matrix([[1, 1], [1, -1]])
    matrix2 = matrix1[:, ::-1]
    matrix2[:, 0]*=2
    assert quantize._equal_up_to_column_swaps_and_shift_and_sign(matrix1, matrix2) == False

    # swap columns
    matrix1 = sym.Matrix([[1, 1], [1, -1]])
    matrix2 = matrix1[:, ::-1]
    matrix2[:, 0] += shift
    matrix2[:, 1] += -1*shift
    assert quantize._equal_up_to_column_swaps_and_shift_and_sign(matrix1, matrix2) == True

    # With symbolic entries
    x, y = sym.symbols('x y', positive=True)
    matrix1 = sym.Matrix([[x, 2], [y, 4]])
    matrix2 = matrix1[:, ::-1]
    matrix2[:, 1]*=-1
    matrix2[:, 1]+=7*shift
    assert quantize._equal_up_to_column_swaps_and_shift_and_sign(matrix1, matrix2) == True
    
    # With fractions
    matrix1 = sym.Matrix([[sym.Rational(1,2), sym.Rational(2,1)], [sym.Rational(3,4), sym.Rational(4,1)]])
    matrix2 = matrix1[:, ::-1]
    matrix2[:, 1]*=2
    assert quantize._equal_up_to_column_swaps_and_shift_and_sign(matrix1, matrix2) == False


def test__nonzero_entries_str():

    test = np.array([[1, 0, 2],
                      [0, 0.01, 0],
                      [2, 0, 3]])
    
    res = quantize._nonzero_entries_str(test)
    assert res == "1-010"

    test = sym.Matrix(np.array([[1, 0.01, 2],
                      [-1, 0.01, -.2],
                      [7.3, 0, 3]]))
    
    res = quantize._nonzero_entries_str(test)
    assert res == "3-111"
    res = quantize._nonzero_entries_str(test, prepend_sum=False)
    assert res == "111"



def test__sort_wT():
     
    test = np.array([[1, 0, 1],
                      [-1, 1, 0],
                      [0, 1, 0]])
    assert np.all(quantize._sort_wT(test) == test[(2, 1, 0), :])

    # rows with same nonzero counts but different nonzero column indices
    mat = sym.Matrix([[0, 1, 0],
                      [1, 0, 0],
                      [0, 1, 1],
                      [0, 0, 0]])
    sorted_mat = quantize._sort_wT(mat)
    # keys are sorted lexicographically; expected order computed from key logic
    expected_order = (3, 1, 0, 2)
    expected = np.array(mat)[expected_order, :]
    assert np.array_equal(np.array(sorted_mat), expected)

    # numpy array input with negative entries should behave the same
    arr = np.array([[0, 0], [0, -1], [1, 0]])
    sorted_arr = quantize._sort_wT(arr)
    # compute expected keys and ordering to assert deterministic behavior
    keys = []
    for i in range(arr.shape[0]):
        keys.append(str(sum(arr[i, :] != 0)) + "_" + "-".join(arr[i, :].nonzero()[0].astype(str)))
    order = np.argsort(keys)
    assert np.array_equal(sorted_arr, arr[order, :])


def test__maximize_wT():

    # Map old format string keys to numerical tuples
    num_map = {"0": -4, "1": -3, "2": -2, "3": -1,"4": 0, "5": 1, "6": 2, "7": 3, "8": 4}
    str_to_num = lambda s: tuple(num_map[c] for c in s)

    test = sym.nsimplify(sym.Matrix([
                        [-1, -1,  0, 0],
                        [-1,  0,  1, 0],
                        [-1,  0,  0, 0],
                        [ 0,  1,  1, 0],
                        [ 0,  1,  0, 0],
                        [ 0,  0, -1, 0]]), rational=True)
    wT, best_key, (row_vec, col_vec, row_order) = quantize._maximize_wT(test[:, :-1])
    ans = str_to_num('544454445554545453')
    assert best_key == ans
     
    test = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
    wT, best_key, (row_vec, col_vec, row_order) = quantize._maximize_wT(test)

    assert np.all(wT == test)
    assert np.all(row_order == np.arange(3))
    assert np.all(row_vec == np.ones(3))
    assert np.all(col_vec == np.ones(3))
    assert best_key == str_to_num("".join(str(int(x) + 3) for x in "211121112"))

    test = np.array([[1, 0, 0],
                      [0, -1, 0],
                      [0, -1, 1]])
    wT, best_key, (row_vec, col_vec, row_order) = quantize._maximize_wT(test)
    ans = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 1, 1]])
    assert np.all(wT == ans)
    assert np.all(row_order == np.arange(3))
    assert best_key == str_to_num("".join(str(int(x) + 3) for x in "211121122"))


    test = np.array([ [0, 0, 0, -1],
                      [-1, 1, 0, 0],
                      [1, 1, 0, 0],
                      [0, -1, 1, 0]])
    wT, best_key, (row_vec, col_vec, row_order) = quantize._maximize_wT(test)
    ans = np.array([ [0, 0, 0, 1],
                      [1, 1, 0, 0],
                      [1, -1, 0, 0],
                      [0, 1, 1, 0]])
    assert np.all(wT == ans)
    assert np.all(row_order == np.array([0, 2, 1, 3]))
    assert best_key == str_to_num("".join((ans + 4).flatten().astype(str)))

    test = np.array([[-1,  0,  0,  0,  0],
                    [ 0,  1,  0,  0,  0],
                    [ 1,  0,  0,  0, -1],
                    [ 0, -1,  0,  1,  0],
                    [ 0,  1,  0,  1,  0],
                    [ 0, -1,  0,  0, -1],
                    [ 0,  0,  1,  0, -1],
                    [ 0,  0,  0,  1, -1],
                    [ 1, -1, -1,  0,  0],
                    [-1,  0, -1,  1,  1]])
    import time
    t0 = time.time()
    wT, best_key, (row_vec, col_vec, row_order) = quantize._maximize_wT(test)
    tf = time.time()
    print(tf-t0)
    ans = np.array([[ 1,  0,  0,  0,  0],
                    [ 0,  1,  0,  0,  0],
                    [ 1,  0,  0,  0,  1],
                    [ 0,  1,  0,  1,  0],
                    [ 0,  1,  0, -1,  0],
                    [ 0,  1,  0,  0,  1],
                    [ 0,  0,  1,  0,  1],
                    [ 0,  0,  0,  1,  1],
                    [ 1,  1, -1,  0,  0],
                    [ 1,  0,  1, -1,  1]])
    assert np.all(wT == ans)
    
    
    test = (np.random.random((17, 5)) > 0.75).astype(int) - (np.random.random((17, 5)) > 0.75).astype(int)
    test = quantize._sort_wT(test)
    t0 = time.time()
    wT, best_key, (row_vec, col_vec, row_order) = quantize._maximize_wT(test)
    tf = time.time()
    print(tf-t0)


def test__var_col_perms():

    var_types = {"compact": [0, 1],
                 "extended": [2],
                 "harmonic": [3]}
    perms = quantize._var_col_perms(var_types)
    assert len(perms) == 2
    assert (0,1,2,3) in perms
    assert (1,0,2,3) in perms
    

def test__sub_equal_LC():

    C1, C2, C3 = sym.symbols("C1, C2, C3")
    CJ1, CJ2 = sym.symbols("CJ1, CJ2")
    L1, L2 = sym.symbols("L1, L2")
    X = sym.Matrix([[C1, L1, C2],
                    [C3, C1, L2],
                    [CJ1, CJ2, CJ1]])
    
    test = quantize._sub_equal_LC(X)

    assert test[0,0] == test[0,2] == test[1,0] == test[1,1]
    assert test[0,1] == test[1,2]
    assert test[2,0] == test[2,1] == test[2,2]


def test__find_Z_deterministic():

    # Solution that only has a single thing
    x,y = sym.symbols("x,y", real=True)
    a,b = sym.symbols("a,b", real=True)
    v = [x,y,a,b]

    Z = sym.Matrix([[x,y],
                    [a,b]])
    
    Zsub = quantize._find_Z_instance_deterministic(Z, v)
    assert Z.det().simplify() != 0
    assert sym.im(Zsub).is_zero_matrix


def test__find_Z_min_cost():

    Z00, Z01, Z02, Z10, Z11, Z12, Z20, Z21, Z22 = sym.symbols('Z00, Z01, Z02, Z10, Z11, Z12, Z20, Z21, Z22')
    wJ =  sym.Matrix([[1, 1, 1, 0, 0, 0], [-1, 0, 0, 1, 1, 0], [0, -1, 0, -1, 0, 1], [0, 0, -1, 0, -1, -1]])
    var_list = [Z20, Z10, Z21, Z22]
    nonzero = [1, 2*Z10 + Z20, 4]
    Z = sym.Matrix([[-Z10 - Z20, 3*Z21*(-Z10 + Z20)/(4*(2*Z10 + Z20)) + Z21*(Z10 + 2*Z20)/(4*(2*Z10 + Z20)) - Z21/4, Z22/4, 1/4], [Z10, -Z21*(-Z10 + Z20)/(4*(2*Z10 + Z20)) - 3*Z21*(Z10 + 2*Z20)/(4*(2*Z10 + Z20)) - Z21/4, Z22/4, 1/4], [Z20, -Z21*(-Z10 + Z20)/(4*(2*Z10 + Z20)) + Z21*(Z10 + 2*Z20)/(4*(2*Z10 + Z20)) + 3*Z21/4, Z22/4, 1/4], [0, -Z21*(-Z10 + Z20)/(4*(2*Z10 + Z20)) + Z21*(Z10 + 2*Z20)/(4*(2*Z10 + Z20)) - Z21/4, -3*Z22/4, 1/4]])
    var_types = {'compact': [], 'extended': [0, 1, 2], 'harmonic': [], 'free': [], 'frozen': [], 'sigma': [3]}
    min_cost = quantize._find_Z_min_cost(Z, var_list, var_types, wJ)
    assert min_cost == 5

    Z21, Z12, Z01, Z02, Z11, Z22 = sym.symbols('Z21 Z12 Z01 Z02 Z11 Z22')

    #     wJ.transpose()*Z_final[-1] = 
    # Matrix([
    # [-1, Z11/2,    Z22/2, 0],
    # [-1, Z11/2,  3*Z22/2, 0],
    # [ 0,     0,      Z22, 0],
    # [ 1, Z11/2,   -Z22/2, 0],
    # [ 1, Z11/2, -3*Z22/2, 0]]) -> Z11,Z22 = 1 -> min_cost = 4+1+3+2+1+3 = 14 +(4 compact) = 18
    
    wJ = sym.nsimplify(sym.Matrix([
            [ 1,  1,  0,  0,  0],
            [-1,  0,  1,  1,  0],
            [ 0, -1, -1,  0,  1],
            [ 0,  0,  0, -1, -1]]), rational=True)
    Z = sym.nsimplify(sym.Matrix([
            [-1/2,  Z11/2, Z22/2, 1/4],
            [ 1/2,      0,     0, 1/4],
            [ 1/2,      0,  -Z22, 1/4],
            [-1/2, -Z11/2, Z22/2, 1/4]]))
    var_types = {'compact': [0], 'extended': [1, 2], 'harmonic': [], 'free': [], 'frozen': [], 'sigma': [3]}
    var_list = [Z11, Z22]
    min_cost = quantize._find_Z_min_cost(Z, var_list, var_types, wJ)
    #     wJ.transpose()*Z_final[-1] = 
    # Matrix([
    # [-1, Z11/2,    Z22/2, 0],
    # [-1, Z11/2,  3*Z22/2, 0],
    # [ 0,     0,      Z22, 0],
    # [ 1, Z11/2,   -Z22/2, 0],
    # [ 1, Z11/2, -3*Z22/2, 0]]) -> Z11,Z22 = 1 -> min_cost = 4+1+3+2+1+3 = 14 +(4 compact) = 18
    assert min_cost == 18

    wJ =  sym.Matrix([[1, 1, 1, 0, 0, 0], [-1, 0, 0, 1, 1, 0], [0, -1, 0, -1, 0, 1], [0, 0, -1, 0, -1, -1]])
    var_types =  {'compact': [0], 'extended': [1, 2], 'harmonic': [], 'free': [], 'frozen': [], 'sigma': [3]}

    # Same as test z3 interface, but with 3 entries from compact mode
    Z = sym.Matrix([[-1/4, -Z01/4 - Z21/4, -3*Z02*Z21/(4*(2*Z01 - Z21)) - Z02/4, 1/4], [3/4, 3*Z01/4 - Z21/4, Z02*Z21/(4*(2*Z01 - Z21)) + 3*Z02/4, 1/4], [-1/4, -Z01/4 + 3*Z21/4, Z02*Z21/(4*(2*Z01 - Z21)) - Z02/4, 1/4], [-1/4, -Z01/4 - Z21/4, Z02*Z21/(4*(2*Z01 - Z21)) - Z02/4, 1/4]])
    var_list =  [Z01, Z02, Z21]
    min_cost = quantize._find_Z_min_cost(Z, var_list, var_types, wJ)
    assert min_cost == 9

    Z =  sym.Matrix([[-1/4, 3*Z11/4, -Z22/2, 1/4], [3/4, -Z11/4, Z22/2, 1/4], [-1/4, -Z11/4, Z22/2, 1/4], [-1/4, -Z11/4, -Z22/2, 1/4]])
    wJT =  sym.Matrix([[-1, Z11, -Z22, 0], [0, Z11, -Z22, 0], [0, Z11, 0, 0], [1, 0, 0, 0], [1, 0, Z22, 0], [0, 0, Z22, 0]])
    var_list = [Z11, Z22]
    min_cost = quantize._find_Z_min_cost(Z, var_list, var_types, wJ)
    assert min_cost == 10

    Z =  sym.Matrix([[-1/4, -3*Z21/8, 3*Z12/4, 1/4], [3/4, Z21/8, -Z12/4, 1/4], [-1/4, 5*Z21/8, -Z12/4, 1/4], [-1/4, -3*Z21/8, -Z12/4, 1/4]])
    wJT =  sym.Matrix([[-1, -Z21/2, Z12, 0], [0, -Z21, Z12, 0], [0, 0, Z12, 0], [1, -Z21/2, 0, 0], [1, Z21/2, 0, 0], [0, Z21, 0, 0]])
    var_list = [Z21, Z12]
    min_cost = quantize._find_Z_min_cost(Z, var_list, var_types, wJ)
    assert min_cost == 13


    return

def test__find_Z_instance():

    Z00, Z01, Z02, Z10, Z11, Z12, Z20, Z21, Z22 = sym.symbols('Z00, Z01, Z02, Z10, Z11, Z12, Z20, Z21, Z22')
 
    Z = sym.nsimplify(sym.Matrix([
            [-1/2,  Z11/2, Z22/2, 1/4],
            [ 1/2,      0,     0, 1/4],
            [ 1/2,      0,  -Z22, 1/4],
            [-1/2, -Z11/2, Z22/2, 1/4]]))
    wJ = sym.nsimplify(sym.Matrix([
            [ 1,  1,  0,  0,  0],
            [-1,  0,  1,  1,  0],
            [ 0, -1, -1,  0,  1],
            [ 0,  0,  0, -1, -1]]), rational=True)
    var_list = [Z11, Z22]
    var_types = {'compact': [0], 'extended': [1, 2], 'harmonic': [], 'free': [], 'frozen': [], 'sigma': [3]}
    Zsub = quantize._find_Z_instance(Z, var_list, wJ=wJ, var_types=var_types)
    ans = sym.nsimplify(sym.Matrix([
                        [-1/2, -1, -1, 1/4],
                        [ 1/2,  0,  0, 1/4],
                        [ 1/2,  0,  2, 1/4],
                        [-1/2,  1, -1, 1/4]]), rational=True)
    assert quantize._equal_up_to_column_shift_and_sign(Zsub, ans)

    # Test apriori_sol that doesn't work
    Z = sym.nsimplify(sym.Matrix([
            [-1/2,  Z11/2, Z22/2, 1/4],
            [ 1/2,      0,     0, 1/4],
            [ 1/2,      0,  -Z22, 1/4],
            [-1/2, -Z11/2, Z22/2, 1/4]]))
    wJ = sym.nsimplify(sym.Matrix([
            [ 1,  1,  0,  0,  0],
            [-1,  0,  1,  1,  0],
            [ 0, -1, -1,  0,  1],
            [ 0,  0,  0, -1, -1]]), rational=True)
    var_list = [Z11, Z22]
    var_types = {'compact': [0], 'extended': [1, 2], 'harmonic': [], 'free': [], 'frozen': [], 'sigma': [3]}
    Zsub = quantize._find_Z_instance(Z, var_list, wJ=wJ, var_types=var_types, apriori_sol=4)
    assert Zsub == []

    # solution with one free variable, which is divided by 2 somewhere
    Z11 = sym.symbols("Z11")
    Z = sym.nsimplify(sym.Matrix([
    [ 1/3,  Z11/2, 1/3],
    [ 1/3, -Z11/2, 1/3],
    [-2/3,      0, 1/3]]), rational=True)
    wJ = sym.Matrix([
    [ 1,  1,  0],
    [-1,  0,  1],
    [ 0, -1, -1]])
    var_list = [Z11]
    var_types = {'compact': [0], 'extended': [1], 'harmonic': [], 'free': [], 'frozen': [], 'sigma': [2]}
    nonzero = [2, sym.simplify(-Z11/3, rational=True)]
    Zsub = quantize._find_Z_instance(Z, var_list, wJ=wJ, var_types=var_types)
    ans = sym.nsimplify(sym.Matrix([
                                [ 1/3,  1, 1/3],
                                [ 1/3, -1, 1/3],
                                [-2/3,  0, 1/3]]), rational=True)
    assert quantize._equal_up_to_column_shift_and_sign(Zsub, ans)

    # Solution that only has a single thing
    x,y = sym.symbols("x,y")
    a,b = sym.symbols("a,b")
    v = [x,y,a,b]

    Z = sym.Matrix([[x,1],
                    [y,1]])
    wJ = sym.Matrix([[1],
                     [-1]])
    Zsub = quantize._find_Z_instance(Z, v, wJ=wJ, var_types={"extended":[0], "sigma":[1]})
    
    assert sym.im(Zsub).is_zero_matrix
    wJT_trans = wJ.transpose()*Zsub
    assert all(sym.simplify(x) in range(-4,5) for x in wJT_trans)



def test_compact_well_aligned():

    wT = sym.nsimplify(sym.Matrix([[1, 0, 1, 1],
                                    [0, -1, 0, 1],
                                    [0, 1, 0, 1],
                                    [1, 0, -1, 1]]), rational=True)
    assert quantize.compact_well_aligned(wT, 2)


    wT = sym.nsimplify(sym.Matrix([[1, 0, 1, 1],
                                    [0, -1, 0, 1],
                                    [0, 1, 0, 1],
                                    [1, 0, -1, 1]]), rational=True)
    assert not quantize.compact_well_aligned(wT, 3)

    wT = sym.nsimplify(sym.Matrix([[1, 1, 1],
                        [2/3, 1/3, 1/10],
                        [40, 90, 200]]), rational=True)
    assert not quantize.compact_well_aligned(wT, 2)

def test_compact_alignment_transformation():

    edges = [(1, 2), (2, 3), (3, 4), (4,1)]
    edges = utils.renumber_nodes(edges)
    circuit = [("J",), ("J","C"), ("J",), ("L",)]
    Z = sym.nsimplify((sym.Matrix([[1, 0, 1, 1],
                                    [0, -1/2, 0, 1],
                                    [0, 1/2, 0, 1],
                                    [1, 0, -1, 1]])), rational=True)
    wj = quantize.gen_w(circuit, edges, "J")
    wT = wj.transpose()*Z
    n_c = 2
    for Z2 in quantize.compact_alignment_transformation(wT, n_c):
        assert quantize.compact_well_aligned(wT*Z2, n_c)

    
    for n_c in [1, 2, 3]:
        wT = sym.nsimplify(sym.Matrix([[1, 1, 1],
                        [2/3, 1/3, 1/10],
                        [40, 90, 200]]), rational=True)
        for Z2 in quantize.compact_alignment_transformation(wT, n_c):
            assert quantize.compact_well_aligned(wT*Z2, n_c)


def test_decouple_column():

    # Transmon molecule
    edges = [(1, 2), (3, 4), (1, 3), (2, 4)]
    circuit = [("J",), ("J",), ("C13",), ("C24",)]
    cMat = quantize.gen_cap_mat(circuit, edges)
    Z1 = sym.simplify(sym.Matrix([[1/2, 0, 1, 1],
                                  [-1/2, 0, 1, 1],
                                  [0, 1/2, 0, 1],
                                  [0, -1/2, 0, 1]]),
                                  rational=True)
    nd_mat = Z1[:, 2:]

    c1 = quantize.decouple_column(Z1[:, 0], nd_mat, cMat)
    c2 = quantize.decouple_column(Z1[:, 1], nd_mat, cMat)
    Z2 = sym.Matrix.hstack(c1, c2, nd_mat)
    test2 = sym.simplify(Z2.transpose()*cMat*Z2)
    for i in [0, 1]:
        for j in [2, 3]:
            assert test2[i,j] == test2[j,i] == 0


def test_decoupling_transformation():

    test = sym.Matrix([[1, 2, 3],
                       [2, 3, 5],
                       [3, 5, 3]])
    
    Z2 = quantize.decoupling_transformation(test, 2)
    test2 = Z2.transpose()*test*Z2
    assert test2[1,2] == test2[2,1] == 0
    assert test2[0,2] == test2[2,0] == 0


    # Transmon molecule
    edges = [(1, 2), (3, 4), (1, 3), (2, 4)]
    circuit = [("J",), ("J",), ("C13",), ("C24",)]
    cMat = quantize.gen_cap_mat(circuit, edges)
    Z1 = sym.simplify(sym.Matrix([[1/2, 0, 1, 1],
                                  [-1/2, 0, 1, 1],
                                  [0, 1/2, 0, 1],
                                  [0, -1/2, 0, 1]]),
                                  rational=True)
    cMat = Z1.transpose()*cMat*Z1
    Z2 = quantize.decoupling_transformation(cMat, 2)
    test2 = sym.simplify(Z2.transpose()*cMat*Z2)
    for i in [0, 1]:
        for j in [2, 3]:
            assert test2[i,j] == test2[j,i] == 0
   
num_map = {"0": -4, "1": -3, "2": -2, "3": -1,"4": 0, "5": 1, "6": 2, "7": 3, "8": 4}
inv_num_map = {v: k for k, v in num_map.items()}
def new_to_old(k):
    old_form = ""
    for c in k:
        if isinstance(c, str):
            old_form += c + "_"
        elif isinstance(c, tuple):
            old_form += "".join(inv_num_map[cr] for cr in c) + "_"
        elif isinstance(c, int):
            old_form += str(c) + "_"
    return old_form[:-1]

def test_H_hash():

   
    # Misbehaving case from earlier version
    L, C, C_J = sym.symbols("L, C, C_J", positive=True)
    J_1, J_2, J_3, J_4, J_5, J_6 = sym.symbols("J_1, J_2, J_3, J_4, J_5, J_6", positive=True)
    Z = sym.Matrix([[-1, -1, -1/4, 1/4], [2, 0, -1/4, 1/4], [-1, 1, -1/4, 1/4], [0, 0, 3/4, 1/4]])
    cTransInv = sym.Matrix([[1/(6*C + 24*C_J), 0, 0], [0, 1/(2*C + 8*C_J), 0], [0, 0, 1/(3*C + 3*C_J)]])
    lTrans = sym.Matrix([[24/L, 0, 0, 0], [0, 8/L, 0, 0], [0, 0, 3/L, 0], [0, 0, 0, 0]])
    wJtTrans = sym.Matrix([[-3, -1, 0, 0], [0, -2, 0, 0], [-1, -1, -1, 0], [3, -1, 0, 0], [2, 0, -1, 0], [-1, 1, -1, 0]])
    EJ = (J_1, J_2, J_3, J_4, J_5, J_6)
    var_types = {'compact': [], 'extended': [0, 1, 2], 'harmonic': [], 'free': [], 'frozen': [], 'sigma': [3]}
    hashes = quantize.H_hash(cTransInv, lTrans, wJtTrans, EJ, var_types)
    assert new_to_old(hashes[0]) == '030_3-111_0_0-000_0-000'
    hashes = quantize.H_hash(cTransInv, lTrans, wJtTrans, EJ, var_types, extra_nl=True)
    assert new_to_old(hashes[0]).replace("-","").replace("_","") == '030_3-111-19-3-446564475473535355_0_0-000_0-000'.replace("-","").replace("_","")

    # Transmon
    edges = [(0, 1)]
    circuit = [("J", "C")]
    _, var_types = quantize.var_trans_basis(circuit, edges)
    cMat = quantize.gen_cap_mat(circuit, edges)
    lMat = quantize.gen_ind_mat(circuit, edges)
    Z = sym.Matrix([[1, 1], [0, 1]])
    wJ, EJ = quantize.gen_w(circuit, edges, w_elem="J", return_params=True)
    # hashes = quantize.H_hash(Z, var_types, cMat, lMat, wJ)
    cTransInv = (Z.transpose()*cMat*Z)[:-1, :-1].inv()
    lTrans = Z.transpose()*lMat*Z
    wJtTrans = wJ.transpose()*Z
    hashes = quantize.H_hash(cTransInv, lTrans, wJtTrans, EJ, var_types)
    assert new_to_old(hashes[0]) == "100_0-_0_0-_0-"
    assert quantize._find_equiv_mats(sym.Matrix([[1, 1],
                                                 [0, 1]]), 
                                     [Z[:, hashes[1]]]) == [0]

    # Zero-pi
    circuit = [("J",),("J",), ("L",), ("L",), ("C",), ("C",)]
    edges = [(0, 1), (2, 3), (0, 3), (1, 2), (0, 2), (1, 3)]
    _, var_types = quantize.var_trans_basis(circuit, edges)
    cMat = quantize.gen_cap_mat(circuit, edges)
    lMat = quantize.gen_ind_mat(circuit, edges)
    # wJ = quantize.gen_w(circuit, edges, w_elem="J")
    Z = sym.Matrix([[1, 1, 1, 1],
                    [0, 0, 1, 1],
                    [0, 1, 0, 1],
                    [1, 0, 0, 1]])
    circuit = [("J1",),("J2",), ("L",), ("L",), ("C",), ("C",)]
    wJ, EJ = quantize.gen_w(circuit, edges, w_elem="J", return_params=True)
    # hashes = quantize.H_hash(Z, var_types, cMat, lMat, wJ)
    cTransInv = (Z.transpose()*cMat*Z)[:-1, :-1].inv()
    lTrans = Z.transpose()*lMat*Z
    wJtTrans = wJ.transpose()*Z
    hashes = quantize.H_hash(cTransInv, lTrans, wJtTrans, EJ, var_types)
    assert new_to_old(hashes[0]) == '111_1-100_0_0-000_0-000'
    assert quantize._find_equiv_mats(Z, [Z[:, hashes[1]]]) == [0]
    hashes = quantize.H_hash(cTransInv, lTrans, wJtTrans, EJ, var_types, extra_nl=True)
    assert new_to_old(hashes[0]).replace("-","").replace("_","") == '111_1-100-4-1-5553_0_0-000_0-000'.replace("-","").replace("_","")
    assert quantize._find_equiv_mats(Z, [Z[:, hashes[1]]]) == [0]

    circuit = [("J1",),("J2",), ("L1",), ("L2",), ("C1",), ("C2",)]
    cMat = quantize.gen_cap_mat(circuit, edges)
    lMat = quantize.gen_ind_mat(circuit, edges)
    wJ, EJ = quantize.gen_w(circuit, edges, w_elem="J", return_params=True)
    # hashes = quantize.H_hash(Z, var_types, cMat, lMat, wJ)
    cTransInv = (Z.transpose()*cMat*Z)[:-1, :-1].inv()
    lTrans = Z.transpose()*lMat*Z
    wJtTrans = wJ.transpose()*Z
    hash, _ = quantize.H_hash(cTransInv, lTrans, wJtTrans, EJ, var_types)
    assert new_to_old(hash) == '111_1-100_4_1-001_3-111'


def test_incidence_to_square():

    wTest = sym.Matrix([[1, 0, 1, 1],
                        [0, -1/2, 0, 1],
                        [0, 1/2, 0, 1],
                        [1, 0, -1, 1.0]])
    ans = functools.reduce(lambda a,b: a+b, [wTest[:,i]*wTest[:,i].transpose() for i in range(4)], sym.zeros(4,4))
    wSquare = quantize.incidence_to_square(wTest, [1,1,1,1])
    assert (wSquare - ans).is_zero_matrix


def test_gen_cap_mat():

    edges = [(0, 1)]
    circuit = [("L")]
    try:
        cMat = quantize.gen_cap_mat(circuit, edges)
    except Exception as e:
        assert isinstance(e, ValueError)

    edges = [(0, 1)]
    circuit = [("C", "L")]
    cMat = quantize.gen_cap_mat(circuit, edges)
    C = cMat.free_symbols.pop()
    assert cMat/C == sym.Matrix([[1, -1],
                                 [-1, 1]])


def test_gen_ind_mat():

    edges = [(0, 1)]
    circuit = [("J", "C")]
    lMat = quantize.gen_ind_mat(circuit, edges)
    assert lMat == sym.Matrix([[0, 0],
                               [0, 0]])

    edges = [(0, 1)]
    circuit = [("J", "C", "L")]
    lMat = quantize.gen_ind_mat(circuit, edges)
    L = lMat.free_symbols.pop()
    assert lMat*L == sym.Matrix([[1, -1],
                                 [-1, 1]])
    
def test_gen_w():

    # Transmon
    edges = [(0, 1)]
    circuit = [("J", "C")]
    w = quantize.gen_w(circuit, edges, w_elem="J")
    assert w == sym.Matrix([[1],
                           [-1]])
    edges = [(0, 1)]
    circuit = [("J", "C")]
    w = quantize.gen_w(circuit, edges, w_elem="C")
    assert w == sym.Matrix([[1],
                           [-1]])
    edges = [(0, 1)]
    circuit = [("J", "C")]
    w = quantize.gen_w(circuit, edges, w_elem="L")
    assert w == sym.Matrix([[0],
                            [0]])
    
    # 0-pi
    edges = [(1, 2), (3, 4), (1, 4), (2, 3), (1, 3), (2, 4)]
    circuit = [("J",), ("J",), ("L",), ("L",), ("C",), ("C",)]
    w = quantize.gen_w(circuit, edges, w_elem="J")
    assert w == sym.Matrix([[1, 0],
                           [-1, 0],
                           [0, 1],
                           [0, -1]])


def test_var_trans_basis():


    circuit, edges = ([('J',), ('J', 'L'), ('C', 'J', 'L')], [(0, 1), (0, 2), (1, 2)])
    Z, var_types = quantize.var_trans_basis(circuit, edges)
    assert var_types["extended"] == [0,1]
    assert var_types["sigma"] == [2]
    assert Z.det() != 0
    for i in range(3):
        for j in range(3):
            assert Z[i,j].is_finite

    # Was making singular Z
    circuit = [('J_1',), ('J_2',), ('C_1', 'L_1'), ('L_2',)]
    edges = [(0, 2), (0, 3), (1, 3), (2, 3)]
    Z, var_types = quantize.var_trans_basis(circuit, edges)
    assert Z.det() != 0

    # Well-spaced and decoupled not possible for harmonic
    circuit, edges = ([('C', 'L'), ('L',), ('J',), ('L',)],
                      [(0, 2), (0, 3), (1, 3), (2, 3)])
    trans, var_types = quantize.var_trans_basis(circuit, edges)
    assert var_types["compact"] == [0]
    assert var_types["harmonic"] == [1]
    assert var_types["frozen"] == [2]
    assert var_types["sigma"] == [3]

    circuit, edges = [('J',), ('J',), ('J',)], [(0, 3), (1, 3), (2, 3)]
    trans, var_types = quantize.var_trans_basis(circuit, edges)
    assert var_types["compact"] == [0, 1, 2]
    assert var_types["sigma"] == [3]

    circuit, edges = [('J',), ('J',), ('J',)], [(0, 1), (0, 2), (1, 2)]
    trans, var_types = quantize.var_trans_basis(circuit, edges)
    assert var_types["compact"] == [0, 1]
    assert var_types["sigma"] == [2]
    
    # Bifluxon
    edges = [(1, 2), (1, 3), (2, 3)]
    circuit = [("J",), ("J",), ("L",)]
    trans, var_types = quantize.var_trans_basis(circuit, edges)
    assert var_types["compact"] == [0]
    assert var_types["extended"] == [1]
    assert var_types["sigma"] == [2]
    
    edges = [(1, 2), (1, 3), (2, 3), (1, 4)]
    circuit = [("J",), ("J",), ("L",), ("C", "L")]
    trans, var_types = quantize.var_trans_basis(circuit, edges)
    assert var_types["compact"] == [0]
    assert var_types["extended"] == [1]
    assert var_types["harmonic"] == [2]
    assert var_types["sigma"] == [3]

    edges = [(1, 2), (1, 3), (2, 3), (1, 4)]
    circuit = [("C",), ("L",), ("J",), ("J",)]
    trans, var_types = quantize.var_trans_basis(circuit, edges)
    assert var_types["compact"] == [0, 1]
    assert var_types["harmonic"] == [2]
    assert var_types["sigma"] == [3]

    # Transmon Molecule
    edges = [(1, 2), (3, 4), (1, 3), (2, 4)]
    circuit = [("J",), ("J",), ("C",), ("C",)]
    trans, var_types = quantize.var_trans_basis(circuit, edges)
    assert var_types["compact"] == [0, 1]
    assert var_types["free"] == [2]
    assert var_types["sigma"] == [3]
   
    # Zero pi
    edges = [(1, 2), (3, 4), (1, 4), (2, 3), (1, 3), (2, 4)]
    circuit = [("J",), ("J",), ("L",), ("L",), ("C",), ("C",)]
    trans, var_types = quantize.var_trans_basis(circuit, edges)
    assert var_types["compact"] == [0]
    assert var_types["extended"] == [1]
    assert var_types["harmonic"] == [2]
    assert var_types["sigma"] == [3]
    
    # Fully linear
    edges = [(1, 2), (3, 4), (1, 3), (2, 4)]
    circuit = [("L",), ("L",), ("C",), ("C",)]
    trans, var_types = quantize.var_trans_basis(circuit, edges)
    assert var_types["harmonic"] == [0]
    assert var_types["free"] == [1]
    assert var_types["frozen"] == [2]
    assert var_types["sigma"] == [3]


def test_secondary_decouple():

    # Nothing to decouple 
    circuit = [("L", "C"), ("L", "C"), ("L",), ("J",)]
    edges = [(1, 2), (3, 4), (1, 3), (2, 4)]
    cMat = quantize.gen_cap_mat(circuit, edges)
    lMat = quantize.gen_ind_mat(circuit, edges)
    wJ, EJ = quantize.gen_w(circuit, edges, w_elem="J", return_params=True)
    jMat = quantize.incidence_to_square(wJ, EJ)

    circuit = utils.add_elem_number(circuit)
    Z0, var_types = quantize.var_trans_basis(circuit, edges)
    Z1 = quantize.secondary_decouple(var_types, [Z0.transpose()*jMat*Z0], [False])


    # Debugging example
    edges = [(1, 2), (3, 4), (1, 3), (2, 4)]
    circuit = [("L_1", "C_1"), ("L_2", "C_2"), ("L_3",), ("J_1",)]
    cMat = quantize.gen_cap_mat(circuit, edges)
    lMat = quantize.gen_ind_mat(circuit, edges)
    edges = utils.renumber_nodes(edges)
    wJ, EJ = quantize.gen_w(circuit, edges, w_elem="J", return_params=True)
    Z0, var_types = quantize.var_trans_basis(circuit, edges)
    Z1 = quantize.secondary_decouple(var_types, [Z0.transpose()*lMat*Z0, Z0.transpose()*cMat*Z0], [False, True])
    cT = []
    lT = []
    wT = []
    vals = []
    Z_perm = []
    for Z in Z1:
        Ztot = Z0*Z
        cInvTrans = sym.simplify((Ztot.transpose()*cMat*Ztot)[:-1, :-1].inv())
        lTrans = sym.simplify(Ztot.transpose()*lMat*Ztot)
        wJtTrans = sym.simplify(wJ.transpose()*Ztot)
        cT.append(cInvTrans)
        lT.append(lTrans)
        wT.append(wJtTrans)
        Z = quantize._find_Z_instance_deterministic(Ztot, (Ztot).free_symbols)
        Z = sym.simplify(Z)
        val, Zp = quantize.H_hash(cInvTrans, lTrans, wJtTrans, EJ, var_types)
        vals.append(val)
        Z_perm.append(Zp)
   
    assert new_to_old(vals[0]).replace("-","").replace("_", "") == '012_0-000_3_2-011_1-001'.replace("-","").replace("_", "")

    # Make sure it's not nan
    circuit, edges = ([('J',), ('J', 'L'), ('C', 'J', 'L')], [(0, 1), (0, 2), (1, 2)])
    Z0, var_types = quantize.var_trans_basis(circuit, edges)
    assert var_types["extended"] == [0,1]
    assert var_types["sigma"] == [2]
    assert Z0.det() != 0
    for i in range(3):
        for j in range(3):
            assert Z[i,j].is_finite
    cMat = quantize.gen_cap_mat(circuit, edges)
    lMat = quantize.gen_ind_mat(circuit, edges)
    wJ = quantize.gen_w(circuit, edges, w_elem="J")
    Z2 = quantize.secondary_decouple(var_types, [Z0.transpose()*lMat*Z0, Z0.transpose()*cMat*Z0], [False, True])
    for Z in Z2:
        for i in range(3):
            for j in range(3):
                assert Z[i,j].is_finite or Z[i,j].is_finite is None


    circuit = [("C", "L1"), ("J", "L2")]
    edges = [(1, 2), (1, 3)]
    edges = utils.renumber_nodes(edges)
    cMat = quantize.gen_cap_mat(circuit, edges)
    lMat = quantize.gen_ind_mat(circuit, edges)
    wJ = quantize.gen_w(circuit, edges, "J")
    Z0 = sym.simplify(sym.Matrix([[1/2, 0, 1],
                                [0, 1, 1],
                                [-1/2, 0, 1]]), rational=True)
    var_types =  {"compact": [],
                "extended": [0],
                "harmonic": [1],
                "sigma": [2]}
    Z2 = quantize.secondary_decouple(var_types, [Z0.transpose()*lMat*Z0, Z0.transpose()*cMat*Z0], [False, True])[0]
    Z_comp = quantize._find_Z_instance_deterministic(Z2, Z2.free_symbols)
    Z_ans = sym.nsimplify(sym.Matrix([[2, 0, 0],[1, 1, 0],[0, 0, -1]]), rational=True)
    assert quantize._equal_up_to_column_swaps_and_shift_and_sign(Z_comp,
                                                                 Z_ans,
                                                                 shifts=[sym.ones(3,1)])
    Z = sym.simplify(Z0*Z_comp)
    cInvTrans = (Z.transpose()*cMat*Z)[:-1, :-1].inv()
    lTrans = Z.transpose()*lMat*Z
    wJtTrans = wJ.transpose()*Z
    # cTransInv, lTrans, wJtTrans, EJ, var_types
    val, Z_perm = quantize.H_hash(cInvTrans, lTrans, wJtTrans, EJ, var_types)
    assert new_to_old(val).replace("-","").replace("_", "") == "011_0-0_0_0-0_0-0".replace("-","").replace("_", "")



def test_choose_Z():


    # All compact
    circuit = [('C_1',), ('C_2',), ('J_1',), ('C_3', 'J_2'), ('C_4',), ('C_5',)]
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    Z, var_types, val = quantize.choose_Z(circuit, edges, return_instance=True)
    assert new_to_old(val).replace("_","").replace("-","") == "200_0-0-2-1-5445_1_0-0_1-1".replace("_","").replace("-","")


    # Zero-pi
    circuit = [("J",),("J",), ("L",), ("L",), ("C",), ("C",)]
    edges = [(0, 1), (2, 3), (0, 3), (1, 2), (0, 2), (1, 3)]
    cMat = quantize.gen_cap_mat(circuit, edges)
    lMat = quantize.gen_ind_mat(circuit, edges)
    wJ = quantize.gen_w(circuit, edges, w_elem="J")
    Z, var_types, val = quantize.choose_Z(circuit, edges, return_instance=True)
    wJT = wJ.transpose()*Z[0]
    cTrans = Z[0].transpose()*cMat*Z[0]
    lTrans = Z[0].transpose()*lMat*Z[0]
    Z = Z[0]
    assert quantize._find_equiv_mats(sym.Matrix([[1, 1],
                                                [0, 0],
                                                [0, 1],
                                                [1, 0]]), 
                                    [Z[:,:2]]) == [0]
    assert new_to_old(val).replace("_","").replace("-","")  == '111_1-100-4-1-5553_0_0-000_0-000'.replace("_","").replace("-","")

    circuit, edges = [('J1',), ('J2',), ('J3',)], [(0, 1), (0, 2), (1, 2)]
    cMat = quantize.gen_cap_mat(circuit, edges)
    lMat = quantize.gen_ind_mat(circuit, edges)
    wJ = quantize.gen_w(circuit, edges, w_elem="J")
    Z, var_types, val = quantize.choose_Z(circuit, edges, return_instance=False)
    assert len(Z) == 3
    assert var_types["compact"] == [0, 1]
    assert var_types["sigma"] == [2]
    for Zi in  [sym.nsimplify(sym.Matrix([[2/3, 1/3, 1/3],
                                         [-1/3, -2/3, 1/3],
                                         [-1/3, 1/3, 1/3]]), rational=True),
                    sym.nsimplify(sym.Matrix([[2/3, 1/3, 1/3],
                                              [-1/3, 1/3, 1/3],
                                              [-1/3, -2/3, 1/3]]), rational=True),
                    sym.nsimplify(sym.Matrix([[1/3, 1/3, 1/3],
                                              [1/3, -2/3, 1/3],
                                              [-2/3, 1/3, 1/3]]), rational=True)]:
        assert len(quantize._find_equiv_mats(Zi, Z)) > 0
    assert new_to_old(val).replace("_","").replace("-","") == "200_1-1_1_0-0_1-1".replace("_","").replace("-","")

    for i in range(1):
        edges = [(1, 2), (3, 4), (1, 3), (2, 4)]
        circuit = [("L1", "C1"), ("L2", "C2"), ("L3",), ("J",)]
        # Swap columns to see if valing will pick right column order
        cMat = quantize.gen_cap_mat(circuit, edges)
        lMat = quantize.gen_ind_mat(circuit, edges)
        wJ = quantize.gen_w(circuit, edges, w_elem="J")
        t0 = time.time()
        Z, var_types, val = quantize.choose_Z(circuit, edges, return_instance=False)
        Z = Z[0]
        print(time.time() - t0)
        # Z2, var_types2, val2 = quantize.choose_Z(circuit, edges, return_instance=True)
        # breakpoint()
        assert new_to_old(val).replace("_","").replace("-","") == "012_0-000_3_2-011_1-001".replace("_","").replace("-","") 

    # Was giving inconsistent results in test_enumeration
    # Examine every way to label nodes
    circuit = [("J", "L"), ("C", "J"), ("C", "J", "L")]
    circuit = utils.add_elem_number(circuit)
    edges_og = ((0, 1), (0, 2), (1, 2))
    edges_all = [edges_og]
    for p in itertools.combinations([0, 1, 2], r=2):
        edges_all.append(tuple(utils.swap_nodes(edges_og, p[0], p[1])))
    for i in range(1):
        res = {}
        for edges in edges_all:
            wJ = quantize.gen_w(circuit, edges, w_elem="J")
            Z, var_types, val = quantize.choose_Z(circuit, edges, return_instance=True)
            Z = Z[0]
            assert var_types["extended"] == [0,1]
            assert var_types["sigma"] == [2]
            if edges not in res:
                res[edges] = set()
            res[edges].add(quantize._maximize_wT((wJ.transpose()*Z)[:,:2])[1])
        for e in res:
            res[e] = frozenset(res[e])
        assert len(set(res.values())) == 1
    
    # Should discover a single wJ 1 transformation
    circuit, edges = ([('J',), ('J', 'L'), ('C', 'J', 'L')], [(0, 1), (0, 2), (1, 2)])
    Z, var_types, val = quantize.choose_Z(circuit, edges, return_instance=True)
    assert len(Z) == 1
    cMat = quantize.gen_cap_mat(circuit, edges)
    lMat = quantize.gen_ind_mat(circuit, edges)
    wJ = quantize.gen_w(circuit, edges, w_elem="J")
    Z_test = sym.nsimplify(sym.Matrix([
                    [-1/3, -2/3, 1/3],
                    [ 2/3,  1/3, 1/3],
                    [-1/3,  1/3, 1/3]]), rational=True)
    assert var_types["extended"] == [0,1]
    assert var_types["sigma"] == [2]
    for i in range(3):
        for j in range(3):
            assert Z[0][i,j].is_finite
    assert quantize._equal_up_to_column_swaps_and_shift_and_sign(Z[0],
                                                             Z_test,
                                                             shifts=[sym.ones(3,1)])
    wJT = wJ.transpose()*Z[0]
    for j in range(wJT.cols):
        for i in range(wJT.rows):
            assert wJT[i,j] in [-1, 0, 1]

    # Transmon Molecule + cap
    edges = [(1, 2), (3, 4), (1, 3), (2, 4)]
    circuit = [("J", "C"), ("J","C"), ("C",), ("C",)]
    Z, var_types, val = quantize.choose_Z(circuit, edges, return_instance=False)
    assert len(Z) == 1
    Z = Z[0]
    assert var_types["compact"] == [0, 1]
    assert var_types["free"] == [2]
    assert var_types["sigma"] == [3]
    assert quantize._equal_up_to_column_shift_and_sign(
                    sym.nsimplify(sym.Matrix([[1/2, 0, 1/2, 1],
                                     [-1/2,0, 1/2, 1],
                                     [0, 1/2, -1/2,1],
                                     [0, -1/2,-1/2,1]]), rational=True), Z)
    assert new_to_old(val).replace("_","").replace("-","") == "200_0-0_1_0-0_1-1".replace("_","").replace("-","")


     # Transmon
    edges = [(0, 1)]
    circuit = [("J", "C")]
    Z, var_types, val = quantize.choose_Z(circuit, edges, return_instance=False)
    Z = Z[0]
    assert new_to_old(val).replace("_","").replace("-","") == "100_0-_0_0-_0-".replace("_","").replace("-","")
    assert quantize._find_equiv_mats(sym.Matrix([[1, 1],
                                                 [0, 1]]), 
                                     [Z]) == [0]



    circuit = [("J1",), ("J2",), ("L1",), ("L2",), ("C1",), ("C2",)]
    edges = [(0, 1), (2, 3), (0, 3), (1, 2), (0, 2), (1, 3)]
    Z, var_types, val = quantize.choose_Z(circuit, edges, return_instance=False)
    cMat = quantize.gen_cap_mat(circuit, edges)
    lMat = quantize.gen_ind_mat(circuit, edges)
    assert new_to_old(val).replace("_","").replace("-","") == '111_1-100_4_1-001_3-111'.replace("_","").replace("-","")

    circuit = [("J",),("J",), ("L1",), ("L2",), ("C1",), ("C2",)]
    edges = [(0, 1), (2, 3), (0, 3), (1, 2), (0, 2), (1, 3)]
    cMat = quantize.gen_cap_mat(circuit, edges)
    lMat = quantize.gen_ind_mat(circuit, edges)
    wJ = quantize.gen_w(circuit, edges, w_elem="J")
    Z, var_types, val = quantize.choose_Z(circuit, edges, return_instance=True)
    Z = Z[0]
    assert quantize._find_equiv_mats(sym.Matrix([[1, 1],
                                                [0, 0],
                                                [0, 1],
                                                [1, 0]]), 
                                    [Z[:,:2]]) == [0]
    assert new_to_old(val).replace("_","").replace("-","") == '111_1-100-4-1-5553_2_1-001_1-010'.replace("_","").replace("-","")


    edges = [(1, 2), (3, 4), (1, 3), (2, 4)]
    circuit = [("L1", "C1"), ("L2", "C2"), ("L3",), ("J",)]
    # Swap columns to see if valing will pick right column order
    cMat = quantize.gen_cap_mat(circuit, edges)
    lMat = quantize.gen_ind_mat(circuit, edges)
    wJ = quantize.gen_w(circuit, edges, w_elem="J")
    t0 = time.time()
    Z, var_types, val = quantize.choose_Z(circuit, edges, return_instance=False)
    tf = time.time()
    print(tf-t0)
    assert new_to_old(val).replace("_","").replace("-","") == "012_0-000_3_2-011_1-001".replace("_","").replace("-","")
    

    # Bifluxon -- Tests the equal parameter tiebreaker
    edges = [(1, 2), (1, 3), (2, 3)]
    circuit = [("J",), ("J",), ("L",)]
    Z0, var_types, val = quantize.choose_Z(circuit, edges, return_instance=False)
    assert var_types["compact"] == [0]
    assert var_types["extended"] == [1]
    assert var_types["sigma"] == [2]
    cMat = quantize.gen_cap_mat(circuit, edges)
    lMat = quantize.gen_ind_mat(circuit, edges)
    wJ = quantize.gen_w(circuit, edges, w_elem="J")
    Z = quantize._find_Z_instance(Z0[0], Z0[0].free_symbols, var_types, wJ)
    assert quantize._equal_up_to_column_shift_and_sign(
                    sym.nsimplify(sym.Matrix([[2/3, 0, 1],
                                     [-1/3, -1, 1],
                                     [-1/3, 1, 1]]), rational=True), Z)
    assert new_to_old(val).replace("_","").replace("-","") == "110_1-1_0_0-0_0-0".replace("_","").replace("-","")

    edges = [(1, 2), (1, 3), (2, 3)]
    circuit = [("J1",), ("J2",), ("L",)]
    Z, var_types, val = quantize.choose_Z(circuit, edges, return_instance=False)
    Z = Z[0]
    assert var_types["compact"] == [0]
    assert var_types["extended"] == [1]
    assert var_types["sigma"] == [2]
    cMat = quantize.gen_cap_mat(circuit, edges)
    lMat = quantize.gen_ind_mat(circuit, edges)
    wJ = quantize.gen_w(circuit, edges, w_elem="J")
    assert new_to_old(val).replace("_","").replace("-","") == "110_1-1_1_0-0_1-1".replace("_","").replace("-","")


    # Transmon Molecule
    edges = [(1, 2), (3, 4), (1, 3), (2, 4)]
    circuit = [("J",), ("J",), ("C",), ("C",)]
    Z, var_types, val = quantize.choose_Z(circuit, edges, return_instance=False)
    Z = Z[0]
    assert var_types["compact"] == [0, 1]
    assert var_types["free"] == [2]
    assert var_types["sigma"] == [3]
    assert quantize._equal_up_to_column_shift_and_sign(
                    sym.nsimplify(sym.Matrix([[1/2, 0, 1/2, 1],
                                     [-1/2,0, 1/2, 1],
                                     [0, 1/2, -1/2,1],
                                     [0, -1/2,-1/2,1]]), rational=True), Z)
    assert new_to_old(val).replace("_","").replace("-","") == "200_0-0_1_0-0_1-1".replace("_","").replace("-","")

    
    # Fully linear
    edges = [(1, 2), (3, 4), (1, 3), (2, 4)]
    circuit = [("L",), ("L",), ("C",), ("C",)]
    Z, var_types, val = quantize.choose_Z(circuit, edges, return_instance=False)
    Z = Z[0]
    assert var_types["harmonic"] == [0]
    assert var_types["free"] == [1]
    assert var_types["frozen"] == [2]
    assert var_types["sigma"] == [3]
    assert new_to_old(val).replace("_","").replace("-","") == "001_0-_0_0-_0-".replace("_","").replace("-","")

    
    # Example from secondary transformation section
    circuit = [("C", "L1"), ("J", "L2")]
    edges = [(1, 2), (1, 3)]
    edges = utils.renumber_nodes(edges)
    cMat = quantize.gen_cap_mat(circuit, edges)
    Z, var_types, val = quantize.choose_Z(circuit, edges, return_instance=False)
    Z = Z[0]
    assert var_types["extended"] == [0]
    assert var_types["harmonic"] == [1]
    assert var_types["sigma"] == [2]
    assert new_to_old(val).replace("_","").replace("-","") == "011_0-0_0_0-0_0-0".replace("_","").replace("-","")


    # Inconsistent results in test_enumeration
    vals = []
    for i in range(5):
        circuit = [('J', 'L'), ('J', 'L'), ('J', 'L')]
        edges = [(0, 1), (0, 2), (1, 2)]
        cMat = quantize.gen_cap_mat(circuit, edges)
        lMat = quantize.gen_ind_mat(circuit, edges)
        wJ = quantize.gen_w(circuit, edges, w_elem="J")
        Z, var_types, val = quantize.choose_Z(circuit, edges, return_instance=True)
        vals.append(val)
    assert len(set(vals)) == 1


def test_gen_junc_pot():

    # Transmon
    edges = [(0, 1)]
    circuit = [("J", "C")]
    obj = pi.to_SCqubits(circuit, edges)
    Z = sym.Matrix(obj.transformation_matrix)

    th_vec = sym.Matrix(sym.symbols("p1, p2"))
    J = quantize.gen_junc_pot(circuit, edges, th_vec)
    ans = r'- E_{J} \cos{\left(p_{1} - p_{2} \right)}'
    assert sym.latex(J, order="grlex") == ans

    J = quantize.gen_junc_pot(circuit, edges, th_vec, Z=Z)
    ans = r'- E_{J} \cos{\left(p_{1} \right)}'
    assert sym.latex(J, order="grlex") == ans


def test_num_subs():

    C1, C2, C3 = sym.symbols("C1, C2, C3", positive=True, real=True)
    L1 = sym.symbols("L1", positive=True, real=True)
    X = sym.Matrix([[C1, 1, C2],
                    [C3, C1, L1]])
    test, vals = quantize.num_subs(X, symbol="C", hermitify=False)
    assert test[0,0] == test[1,1] == vals[C1]
    assert test[0,2] == vals[C2]
    assert test[1,0] == vals[C3]
    assert test[0,1] == 1
    assert L1 in test.free_symbols

    X = sym.Matrix([[C1, 1, C2],
                    [C3, C1, L1]])
    test, vals = quantize.num_subs(X, symbol="L", hermitify=False)
    assert test[1,2] == vals[L1]
    assert test[0,1] == 1
    assert all(x in test.free_symbols for x in  [C1, C2, C3])

    vals_in = {C1: 1, C2:2, C3:3}
    test, vals = quantize.num_subs(X, symbol="C", vals_in=vals_in, hermitify=False)
    assert test[0,0] == test[1,1] == vals[C1] == vals_in[C1]
    assert test[0,2] == vals[C2] == vals_in[C2]
    assert test[1,0] == vals[C3] == vals_in[C3]
    assert test[0,1] == 1
    assert L1 in test.free_symbols

    CJ = sym.symbols(r"C_{J}", positive=True, real=True)
    X = sym.Matrix([[CJ, C1],
                    [C1, 1]])
    test, vals = quantize.num_subs(X, symbol="C", exclude="J", hermitify=True)
    assert test[1,0] == test[0,1] == vals[C1]
    assert test[1,1] == 1
    assert CJ in test.free_symbols
    assert test == test.transpose()

    CJ = sym.symbols(r"C_{J}", positive=True, real=True)
    X = sym.Matrix([[CJ, 0],
                    [C1, 1]])
    test, vals = quantize.num_subs(X, symbol=r"C_{J}", hermitify=True)
    assert test[1,0] == test[0,1] == C1/2
    assert test[0,0] == vals[CJ]
    assert all(x in test.free_symbols for x in  [C1])
    assert test == test.transpose()


def test_collect_H_terms():

    q_vec, th_vec = quantize.gen_variables(3, Z = sym.eye(3), periodic=[1])

    a,b,c = sym.symbols("a,b,c", positive=True, real=True)

    H = q_vec[0]**2 + a*q_vec[0]**2 + b*q_vec[1]*q_vec[2] + c*q_vec[1]*q_vec[2] + a*(th_vec[1]-th_vec[2])**2 + c*th_vec[2]**2 + sym.cos(th_vec[0]+th_vec[1]) + sym.cos(th_vec[0]-th_vec[1])

    H_collect, combos_th, combos_q = quantize.collect_H_terms(H, collect_phase=True)

    ans = 'a n_{1}^{2} + a \\left(φ_{2} - φ_{3}\\right)^{2} + b q_{2} q_{3} + c q_{2} q_{3} + c φ_{3}^{2} + n_{1}^{2} + \\cos{\\left(θ_{1} - φ_{2} \\right)} + \\cos{\\left(θ_{1} + φ_{2} \\right)}'

    assert sym.latex(H, order="grlex") == ans

def test_symbolic_hamiltonian():

    # Fluxonium
    edges = [(0, 1)]
    circuit = [("J", "L")]
    obj = pi.to_SCqubits(circuit, edges)
    Z = sym.simplify(sym.Matrix([[1/2, 1/2],
                                [-1/2, 1/2]]), rational=True)
    var_types = {"extended": [1], "sigma": [2]}
    H, qv, tv = quantize.symbolic_hamiltonian(circuit, edges, Z=Z, var_types=var_types,
                                          return_vars=True)
    ans = '- E_{J} \\cos{\\left(φ_{1} \\right)} + \\frac{φ_{1}^{2}}{2 L} + \\frac{q_{1}^{2}}{2 C_{J}}'
    assert sym.latex(H, order="grlex") == ans

    # 0 - pi
    edges = [(1, 2), (3, 4), (1, 4), (2, 3), (1, 3), (2, 4)]
    circuit = [("J",), ("J",), ("L",), ("L",), ("C",), ("C",)]
    Z = sym.Matrix([[1, 1, 1, 1],
                    [0, 0, 1, 1],
                    [0, 1, 0, 1],
                    [1, 0, 0, 1]])
    var_types = {"compact":[1], "extended": [2], "harmonic": [3], "sigma": [4]}

    H, cMat, lMat, wJT = quantize.symbolic_hamiltonian(circuit, edges, Z=Z, var_types=var_types,
                                          return_vars=False, return_mats=True)
    ans = "- 2 E_{J} \\cos{\\left(θ_{1} \\right)} \\cos{\\left(φ_{2} \\right)} + \\frac{n_{1}^{2}}{4 C + 4 C_{J}} + \\frac{φ_{2}^{2}}{L} + \\frac{φ_{3}^{2}}{L} + \\frac{q_{2}^{2}}{4 C_{J}} + \\frac{q_{3}^{2}}{4 C}"
    assert sym.latex(H, order="grlex") == ans

    assert cMat.shape == lMat.shape == (3,3)
    assert wJT.shape == (2,3)

    C = [x for x in H.free_symbols if "C" in str(x) and "J" not in str(x)][0]
    Cj = [x for x in H.free_symbols if "C" in str(x) and "J" in str(x)][0]
    L = [x for x in H.free_symbols if "L" in str(x)][0]

    cMat_exp = sym.Matrix([[2*(C+Cj), 0, 0],
                            [0, 2*Cj, 0],
                            [0, 0, 2*C]])
    lMat_exp = sym.Matrix([[0, 0, 0],
                            [0, 2/L, 0],
                            [0, 0, 2/L]])
    wJT_exp = sym.Matrix([[1,1,0],[-1,1,0]])
    assert cMat == cMat_exp
    assert lMat == lMat_exp                   
    assert quantize._equal_up_to_column_shift_and_sign(wJT, wJT_exp)

    # Transmon with Drive
    edges = [(0, 1)]
    circuit = [("J", "C")]
    Z = sym.simplify(sym.Matrix([[1/2, 1/2],
                                [-1/2, 1/2]]), rational=True)
    Cv = sym.Matrix(np.array([sym.Symbol("C_c", real=True, positive=True), 0]).reshape((2, 1)))
    V = sym.Matrix(np.array([sym.Symbol("V_g", real=True, positive=True)]).reshape((1, 1)))
    var_types = {"compact": [1], "sigma": [2]}
    H, qv, tv = quantize.symbolic_hamiltonian(circuit, edges, Z=Z, var_types=var_types, return_vars=True, Cv=Cv, V=V)
    ans = '- \\frac{C_{c} V_{g} n_{1}}{C + C_{J}} - E_{J} \\cos{\\left(θ_{1} \\right)} + \\frac{n_{1}^{2}}{2 C + 2 C_{J}}'
    assert sym.latex(H, order="grlex") == ans



def main():
    
    # from sircuitenum import enum

    # entry = utils.get_circuit_data_batch("../circuits.db", n_nodes=4, filter_str="WHERE unique_key LIKE 'n4_g5_c42871'").iloc[0]
    # circuit, edges = entry.circuit, entry.edges
    # obj = pi.to_SCqubits(circuit, edges, sym_cir=True, initiate_sym_calc=False)
    # Z, var_class = obj.variable_transformation_matrix()
    # var_class["free"] = [utils.get_num_nodes(edges)]
    # H, trans, H_class = enum.gen_hamiltonian(entry.circuit, entry.edges,
    #                                         cob=Z, var_class = var_class, symmetric=False)

    # test__wT_key()
    # test__sort_wT()
    # test__sub_equal_LC()
    # test_num_subs()
    # test_collect_H_terms()

    # test__to_eq_list()
    # test_decoupling_transformation()
    # test_decouple_column()

    # test_symbolic_hamiltonian()
    # test__var_col_perms()

    # test_unique_compact()
    # test__find_equiv_mats()
    # test__find_equiv_cols()
    # test__nonzero_entries_str()
    # test__maximize_wT()
    # test_H_hash()
    # test__find_Z_instance_deterministic()
    # test__fully_compatible_set()
    # test__unique_products()
    # test__fully_compatible_set()
    # test__sol_indep_of_vars()

    # test__maximally_compatible_set()
    # test__remove_permutation_equivalent_transformations()
    # test_choose_Z()

    # test__vec_space_overlap()
    # test__independent_from()

    test__find_Z_deterministic()
    test_secondary_decouple()

    # test_H_hash()
    # test__find_Z_instance()
    test_choose_Z()
    # test__find_Z_min_cost()

    # test__wT_key()
    # test_var_trans_basis()
    # test_incidence_to_square()
    # test_gen_cap_mat()
    # test_var_trans_basis()
    # test__cached_solve()

    return

if __name__ == "__main__":

    main()