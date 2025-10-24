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
    assert len(in_v1) == len(in_v2) == 1

    assert overlap[0] == sym.Matrix([1, 1, 0])
    assert in_v1[0] == sym.Matrix([0, 1])
    assert in_v2[0] == sym.Matrix([1, 1])

    vecs1 = [sym.Matrix([1, 0, 1]),
             sym.Matrix([1, 1, 0]),
             sym.Matrix([1, -1, -1])]
    
    vecs2 = [sym.Matrix([4, 1, 0]),
             sym.Matrix([-1, 0, 0]),
             sym.Matrix([0, 0, 1])]

    overlap, in_v1, in_v2 = quantize._vec_space_overlap(vecs1, vecs2, return_decomp=True)

    assert len(overlap) == len(in_v1) == len(in_v2) == 3

    for v in vecs2:
        assert v in overlap

    assert sym.Matrix([1, 2, 1]) in in_v1
    assert sym.Matrix([sym.Rational(-1, 3),
                       sym.Rational(-1, 3),
                       sym.Rational(-1, 3)]) in in_v1
    assert sym.Matrix([sym.Rational(2, 3),
                       sym.Rational(-1, 3),
                       sym.Rational(-1, 3)]) in in_v1
    assert sym.Matrix([1, 0, 0]) in in_v2
    assert sym.Matrix([0, 1, 0]) in in_v2
    assert sym.Matrix([0, 0, 1]) in in_v2

    vecs1 = [sym.Matrix([1, 2, 1]),
             sym.Matrix([1, 1, -1])]
    vecs2 = [sym.Matrix([-2, 0, 0]),
             sym.Matrix([0, 0, 2])]
    support = sym.Matrix([0, 1, 0])
    
    overlap, in_v1, in_v2, in_sup = quantize._vec_space_overlap(vecs1, vecs2, support=support, return_decomp=True)
    assert len(overlap) == len(in_v1) == len(in_v2) == len(in_sup) == 2
    assert sym.Matrix([-1, -1]) in in_v1
    assert sym.Matrix([1, -1]) in in_v1
    assert sym.Matrix([-3]) in in_sup
    assert sym.Matrix([1]) in in_sup

    support = sym.Matrix([0, 0, 1])
    overlap, in_v1, in_v2, in_sup = quantize._vec_space_overlap(vecs1, vecs2, support=support, return_decomp=True)
    assert len(overlap) == len(in_v1) == len(in_v2) == len(in_sup) == 1
    assert sym.Matrix([2, -4]) in in_v1
    assert sym.Matrix([6]) in in_sup


    vecs1 = [sym.Matrix([0, 2, 0])]
    vecs2 = [sym.Matrix([-2, 0, 0]),
             sym.Matrix([0, 0, 2])]
    support = sym.Matrix([1, 0, 0])
    overlap, in_v1, in_v2, in_sup = quantize._vec_space_overlap(vecs1, vecs2, support=support, return_decomp=True)
    assert len(overlap) == len(in_v1) == len(in_v2) == len(in_sup) == 0

    vecs2 = [sym.nsimplify(sym.Matrix([
                        [ 1/2],
                        [   0],
                        [   0],
                        [-1/2]]), rational=True)]
    vecs1 = [sym.Matrix([
                [1],
                [1],
                [1],
                [0]]),
             sym.Matrix([
                [0],
                [0],
                [0],
                [1]])]
    support = sym.Matrix([0, 1, 1, 0])
    overlap, in_v1, in_v2, in_sup = quantize._vec_space_overlap(vecs1, vecs2, support=support, return_decomp=True)
    assert len(overlap) == len(in_v1) == len(in_v2) == len(in_sup) == 1
    assert overlap[0] == sym.nsimplify(sym.Matrix([
                        [ 1/2],
                        [ 1/2],
                        [ 1/2],
                        [-1/2]]), rational=True)


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
     
    test = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
    wT, best_key, (row_vec, col_vec, row_order) = quantize._maximize_wT(test)

    assert np.all(wT == test)
    assert np.all(row_order == np.arange(3))
    assert np.all(row_vec == np.ones(3))
    assert np.all(col_vec == np.ones(3))
    assert best_key == "".join(str(int(x) + 3) for x in "211121112")

    test = np.array([[1, 0, 0],
                      [0, -1, 0],
                      [0, -1, 1]])
    wT, best_key, (row_vec, col_vec, row_order) = quantize._maximize_wT(test)
    ans = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 1, 1]])
    assert np.all(wT == ans)
    assert np.all(row_order == np.arange(3))
    assert best_key == "".join(str(int(x) + 3) for x in "211121122")


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
    assert best_key == "".join((ans + 4).flatten().astype(str))

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


def test__to_frozenset():

    a, b, c, d, e = sym.symbols("a b c d e")
    s1 = [sym.Eq(a,b), sym.Eq(c,d)]
    s2 = [sym.Eq(c,d), sym.Eq(a,b)]
    s3 = [sym.Eq(a,b), sym.Eq(c,e)]
    fs1 = quantize._to_frozenset(s1)
    fs2 = quantize._to_frozenset(s2)
    fs3 = quantize._to_frozenset(s3)
    assert fs1 == fs2
    assert fs1 != fs3


def test__to_eq_list():

    # list of frozenset of frozenset -> eq list
    Z11 = sym.symbols("Z11")
    test = [frozenset({frozenset({0, Z11 + 1})})]
    l1 = quantize._to_eq_list(test[0])
    assert len(l1) == 1

    # Dictionary -> eq list
    a, b, c, d, e = sym.symbols("a b c d e")
    s1 = {a: b, c: d}
    s2 = {c: d, a: b}
    s3 = {a: b, c: e}
    l1 = quantize._to_eq_list(s1)
    l2 = quantize._to_eq_list(s2)
    l3 = quantize._to_eq_list(s3)
    assert all(x in l1 for x in l2) and all(x in l2 for x in l1)
    assert not (all(x in l1 for x in l3) and all(x in l3 for x in l1))

    # Eq list -> frozenset -> Eq list
    s1 = [sym.Eq(a,b), sym.Eq(c,d)]
    s2 = [sym.Eq(c,d), sym.Eq(a,b)]
    s3 = [sym.Eq(a,b), sym.Eq(c,e)]
    fs1 = quantize._to_frozenset(s1)
    fs2 = quantize._to_frozenset(s2)
    fs3 = quantize._to_frozenset(s3)
    s1_b = quantize._to_eq_list(fs1)
    s2_b = quantize._to_eq_list(fs2)
    s3_b = quantize._to_eq_list(fs3)
    assert all(x in s1 or sym.Eq(x.rhs, x.lhs) in s1 for x in s1_b)
    assert all(x in s2 or sym.Eq(x.rhs, x.lhs) in s2 for x in s2_b)
    assert all(x in s3 or sym.Eq(x.rhs, x.lhs) in s3 for x in s3_b)

    # list of dict -> eq list
    l1 = quantize._to_eq_list(s1+s2+s3)
    assert len(l1) == 6
    assert all(x in l1 for x in [sym.Eq(a,b), sym.Eq(c,d), sym.Eq(c,e)])

    # expression -> eq list
    expr1 = a - b
    expr2 = c - d
    l1 = quantize._to_eq_list([expr1, expr2])
    assert len(l1) == 2
    assert all(x in l1 for x in [sym.Eq(a-b,0), sym.Eq(c-d,0)])


def test__sols_set_to_dict():
    x, y, z = sym.symbols("x y z")
    # First sol: numeric for x, Complexes for y (should be skipped), numeric for z
    sols_set = [(1, sym.Complexes, 3), (x, y, z)]
    res = quantize._sols_set_to_dict(sols_set, [x, y, z])
    assert isinstance(res, list)
    assert res[0] == {x: 1, z: 3}
    # second solution maps each var to itself -> dict should be empty
    assert res[1] == {}


def test__eq_as_numer_denom():
    x, y = sym.symbols("x y")
    # simple rational eq
    numer, denom = quantize._eq_as_numer_denom(sym.Eq(1/(x + 1), 0))
    assert numer == 1
    assert denom == x + 1
    # linearized fraction
    numer, denom = quantize._eq_as_numer_denom(sym.Eq(x/y, 2))
    assert sym.simplify(numer - (x - 2*y)) == 0
    assert denom == y
    # two rational sides -> common denom is product
    numer, denom = quantize._eq_as_numer_denom(sym.Eq((x + 1)/(x - 1), (y + 1)/(y - 1)))
    assert denom == (x - 1)*(y - 1)


def test__eq_list_to_dict():
    a, b, c = sym.symbols("a b c")

    # simple list of Eq -> dict
    eqs = [sym.Eq(a, b), sym.Eq(b, c)]
    d = quantize._eq_list_to_dict(eqs)
    assert isinstance(d, dict)
    assert d[a] == b and d[b] == c

    # reversed numeric side -> should flip to var: number
    d = quantize._eq_list_to_dict([sym.Eq(2, a)])
    assert d == {a: 2}

    # conflicting assignments for same var -> None
    assert quantize._eq_list_to_dict([sym.Eq(a, b), sym.Eq(a, c)]) is None

def test__maximally_compatible_set():

    Z00, Z11, Z12, Z21, Z20, Z02, Z22, Z10 = sym.symbols("Z00, Z11, Z12, Z21, Z20, Z02, Z22, Z10", real=True)
    C1, C2, C_J = sym.symbols("C1, C2, C_{J}", real=True, positive=True)
    var_list = [Z00, Z11, Z12, Z21, Z20, Z02, Z22, Z10]
    # coupling_c =  [[{Z00: Z10 - Z12*Z20/Z22}], [{Z00: Z10 - Z11*Z20/Z21}],
    #                [{Z00: Z10, Z12: 0, Z21: 0}, {Z00: Z10, Z11: 0, Z22: 0}]]
    # coupling_l =  [[{Z10: Z20, Z11: 0}, {Z00: Z10, Z11: Z21, Z20: 0}, {Z10: 0, Z11: 0, Z20: 0}, {Z00: Z20, Z10: Z20, Z21: 0}],
    #                [{Z10: 0, Z12: 0, Z20: 0}, {Z00: Z10, Z12: Z22, Z20: 0}, {Z00: Z20, Z10: Z20, Z22: 0}]]
    coupling_c =  [[{Z00: Z10 - Z12*Z20/Z22}], [{Z00: Z10 - Z11*Z20/Z21}], [{Z00: Z10 - Z11*Z20/Z21, Z12: 0}, {Z00: Z10, Z12: 0}, {Z00: Z10, Z11: 0, Z22: 0}]]
    coupling_l =  [[{Z00: Z10, Z11: Z21, Z20: 0}, {Z10: 0, Z11: 0, Z20: 0}, {Z00: Z20, Z10: Z20, Z21: 0}], [{Z10: Z20, Z12: 0}, {Z00: Z10, Z12: Z22, Z20: 0}, {Z10: 0, Z12: 0, Z20: 0}, {Z00: Z20, Z10: Z20, Z22: 0}]]
    dets =  [Z00*(Z11*Z22 - Z12*Z21), C1*C2*C_J*Z00**2*(Z11**2*Z22**2 - 2*Z11*Z12*Z21*Z22 + Z12**2*Z21**2)]
    terms = coupling_c + coupling_l
    keys, subs = quantize._maximally_compatible_set(terms, var_list=var_list, nonzero=dets)

    x, y = sym.symbols("x y", real=True)

    # Basic: two independent terms should be selectable together
    terms = [
        [{x: 0}, {x: 1}],
        [{y: 0}, {y: 1}],
    ]
    keys, subs = quantize._maximally_compatible_set(terms, [x, y])
    assert (0, 1) in keys
    assert any(isinstance(s, dict) and x in s and y in s for s in subs)

    # Incompatibility: two terms force different values for same var
    terms = [
        [{x: 0}],
        [{x: 1}],
        [{y: 0}],
    ]
    keys, subs = quantize._maximally_compatible_set(terms, [x, y])
    # no key should contain both incompatible terms 0 and 1
    assert all(not (0 in k and 1 in k) for k in keys)
    # maximal compatible set size should be 2 (the y-term plus one x-term)
    assert any(len(k) == 2 for k in keys)

    # Tiebreaker: prefer the set with smaller sum of indices (02 over 12)
    tiebreaker = lambda ks: sum(ks) if ks else 20
    keys, subs = quantize._maximally_compatible_set(terms, [x, y], tiebreaker_fn=tiebreaker)
    assert any(k == (0, 2) for k in keys)


def test__fully_compatible_set():

    # ensure order does not matter
    Z00, Z11, Z12, Z21, Z20, Z02, Z22, Z10 = sym.symbols("Z00, Z11, Z12, Z21, Z20, Z02, Z22, Z10", real=True)
    solve_vars = (Z00, Z11, Z12, Z21, Z20, Z02, Z22, Z10)
    ord1 = [[{Z12: 0}, {Z00: Z10}], [{Z20: 0}, {Z22: 0}], [{Z12: Z22}, {Z10: Z20}]]
    ord2 =  [[{Z12: Z22}, {Z10: Z20}], [{Z20: 0}, {Z22: 0}], [{Z12: 0}, {Z00: Z10}]]
    res1 = quantize._fully_compatible_set(ord1, solve_vars, depth_first=False)
    res2 = quantize._fully_compatible_set(ord2, solve_vars, depth_first=False)
    res1_set = frozenset(quantize._to_frozenset(quantize._to_eq_list(r)) for r in res1)
    res2_set = frozenset(quantize._to_frozenset(quantize._to_eq_list(r)) for r in res2)
    print(res1_set)
    print(res2_set)
    assert res1_set == res2_set
    

    C1, C2 = sym.symbols("C1, C2", real=True, positive=True)
    expr = (-4*Z00*Z11*Z12*Z22 - 2*Z00*Z11*Z22**2 +
            4*Z00*Z12**2*Z21 + 2*Z00*Z12*Z21*Z22 - 
            4*Z10*Z11*Z22**2 + 4*Z10*Z12*Z21*Z22 + 
            4*Z11*Z12*Z20*Z22 - 4*Z12**2*Z20*Z21)
    assumptions = [({expr: 0},)]
    res_all = quantize._fully_compatible_set(assumptions, solve_vars, depth_first=False)
    assert len(res_all) == 2
    assert {Z00: sym.simplify(2*(-Z10*Z22 + Z12*Z20)/(2*Z12 + Z22))} in res_all
    assert {Z11: Z12*Z21/Z22} in res_all

    res_one = quantize._fully_compatible_set(assumptions, solve_vars, depth_first=True)
    assert len(res_one) == 1
    assert {Z00: sym.simplify(2*(-Z10*Z22 + Z12*Z20)/(2*Z12 + Z22))} in res_one or {Z11: Z12*Z21/Z22} in res_one

    # Produces infite denominator with the first expression applied to last
    assumptions = [({Z11: -1, Z12:-Z22/2},), ({Z00: sym.simplify(2*(-Z10*Z22 + Z12*Z20)/(2*Z12 + Z22)),},)]
    res_one = quantize._fully_compatible_set(assumptions, solve_vars, depth_first=False)
    print("res one", res_one)
    assert len(res_one) == 0

    assumptions = [({Z11: -1, Z12:-Z22/2}, {Z12: Z22}), ({Z00: sym.simplify(2*(-Z10*Z22 + Z12*Z20)/(2*Z12 + Z22)),},)]
    res_all = quantize._fully_compatible_set(assumptions, solve_vars, depth_first=False)
    assert len(res_all) == 1

    assumptions = [({Z11: -1, Z12:-Z22/2}, {Z12: Z22}), ({Z11: Z12*Z21/Z22},)]
    res_all = quantize._fully_compatible_set(assumptions, solve_vars, depth_first=False)
    assert len(res_all) == 2
    
    assumptions = [({Z11: -1, Z12:-Z22/2},), ({Z00: sym.simplify(2*(-Z10*Z22 + Z12*Z20)/(2*Z12 + Z22))},
                                                         {Z11: Z12*Z21/Z22})]
    res_all = quantize._fully_compatible_set(assumptions, solve_vars, depth_first=False)
    assert len(res_all) == 1


def test__cached_solve():

    eps = 1e-06
    for i in range(10):
        x, y, z = sym.symbols("x,y,z")
        eq1 = x + 2*y + 3*z - 6*np.random.random()
        eq2 = 2*x + 3*y + z - 5*np.random.random()
        eq3 = x - y + z - 2*np.random.random()
        eqs = [eq1, eq2, eq3]
        vars = [x, y, z]
        sol_good = sym.solve(eqs, vars, dict=True, simplify=True)[0]
        t0 = time.time()
        sol1 = quantize._cached_solve(eqs, vars)[0]
        t1 = time.time()
        sol2 = quantize._cached_solve(eqs, vars)[0]
        t2 = time.time()
        assert t1 - t0 > t2 - t1  # second call should be faster due to caching
        for v in vars:
            assert abs(sol1[v] - sol_good[v]) < eps
            assert abs(sol2[v] - sol_good[v]) < eps


def test__sol_indep_of_vars():

    # From real example
    C1, C2, C_J = sym.symbols("C1, C2, C_J", real=True, positive=True)
    L1, L2, L3 = sym.symbols("L1, L2, L3", real=True, positive=True)
    Z00, Z11, Z12, Z21, Z20, Z02, Z22, Z10 = sym.symbols("Z00, Z11, Z12, Z21, Z20, Z02, Z22, Z10", real=True)
    solve_vars = [Z00, Z11, Z12, Z21, Z20, Z02, Z22, Z10]
    dets =  [Z00*(Z11*Z22 - Z12*Z21), C1*C2*C_J*Z00**2*(Z11**2*Z22**2 - 2*Z11*Z12*Z21*Z22 + Z12**2*Z21**2)]

    eqL02 =  Z12*(-Z20/L3 - Z00/L1 + Z10*(L1 + L3)/(L1*L3)) + Z22*(-Z10/L3 + Z20*(L2 + L3)/(L2*L3))
    varsL02 =  [Z00, Z10, Z12, Z20, Z22]
    # solL02 =  [{Z10: 0, Z12: 0, Z20: 0}, {Z00: Z10, Z12: Z22, Z20: 0}, {Z00: Z20, Z10: Z20, Z22: 0}]
    resL02 = quantize._sol_indep_of_vars(eqL02, varsL02, nonzero=dets)
    assert sum(abs(sym.simplify(eqL02.subs(resL02[i])) == 0) for i in range(len(resL02)))
    assert len(resL02) == 3

    eqC01 =  (-(C1*Z12**2 + C2*Z22**2)*(C2*Z20*Z21 + Z11*(-C1*Z00 + C1*Z10)) + (C1*Z11*Z12 + C2*Z21*Z22)*(C2*Z20*Z22 + Z12*(-C1*Z00 + C1*Z10)))/((C1*Z11**2 + C2*Z21**2)*(C1*Z12**2 + C2*Z22**2)*(C2*Z20**2 + Z00*(-C1*Z10 + Z00*(C1 + C_J)) + Z10*(-C1*Z00 + C1*Z10)) - (C1*Z11**2 + C2*Z21**2)*(C2*Z20*Z22 + Z12*(-C1*Z00 + C1*Z10))*(-C1*Z00*Z12 + C1*Z10*Z12 + C2*Z20*Z22) - (C1*Z12**2 + C2*Z22**2)*(C2*Z20*Z21 + Z11*(-C1*Z00 + C1*Z10))*(-C1*Z00*Z11 + C1*Z10*Z11 + C2*Z20*Z21) - (C1*Z11*Z12 + C2*Z21*Z22)**2*(C2*Z20**2 + Z00*(-C1*Z10 + Z00*(C1 + C_J)) + Z10*(-C1*Z00 + C1*Z10)) + (C1*Z11*Z12 + C2*Z21*Z22)*(C2*Z20*Z21 + Z11*(-C1*Z00 + C1*Z10))*(-C1*Z00*Z12 + C1*Z10*Z12 + C2*Z20*Z22) + (C1*Z11*Z12 + C2*Z21*Z22)*(C2*Z20*Z22 + Z12*(-C1*Z00 + C1*Z10))*(-C1*Z00*Z11 + C1*Z10*Z11 + C2*Z20*Z21))
    varsC01 =  [Z00, Z10, Z11, Z12, Z20, Z21, Z22]
    solC01 =  [{Z00: Z10 - Z12*Z20/Z22}]
    resC01 = quantize._sol_indep_of_vars(eqC01, varsC01, nonzero=dets)
    assert sum(abs(sym.simplify(eqC01.subs(resC01[i])) == 0) for i in range(len(resC01)))
    assert len(resC01) == 1

    eqL01 =  Z11*(-Z20/L3 - Z00/L1 + Z10*(L1 + L3)/(L1*L3)) + Z21*(-Z10/L3 + Z20*(L2 + L3)/(L2*L3))
    varsL01 =  [Z00, Z10, Z11, Z20, Z21]
    solL01 =  [{Z10: 0, Z11: 0, Z20: 0}, {Z00: Z10, Z11: Z21, Z20: 0}, {Z00: Z20, Z10: Z20, Z21: 0}]
    resL01 = quantize._sol_indep_of_vars(eqL01, varsL01, nonzero=dets)
    assert sum(abs(sym.simplify(eqL01.subs(resL01[i])) == 0) for i in range(len(resL01)))
    assert len(resL01) == 3

    eqC02 =  (-(C1*Z11**2 + C2*Z21**2)*(C2*Z20*Z22 + Z12*(-C1*Z00 + C1*Z10)) + (C1*Z11*Z12 + C2*Z21*Z22)*(C2*Z20*Z21 + Z11*(-C1*Z00 + C1*Z10)))/((C1*Z11**2 + C2*Z21**2)*(C1*Z12**2 + C2*Z22**2)*(C2*Z20**2 + Z00*(-C1*Z10 + Z00*(C1 + C_J)) + Z10*(-C1*Z00 + C1*Z10)) - (C1*Z11**2 + C2*Z21**2)*(C2*Z20*Z22 + Z12*(-C1*Z00 + C1*Z10))*(-C1*Z00*Z12 + C1*Z10*Z12 + C2*Z20*Z22) - (C1*Z12**2 + C2*Z22**2)*(C2*Z20*Z21 + Z11*(-C1*Z00 + C1*Z10))*(-C1*Z00*Z11 + C1*Z10*Z11 + C2*Z20*Z21) - (C1*Z11*Z12 + C2*Z21*Z22)**2*(C2*Z20**2 + Z00*(-C1*Z10 + Z00*(C1 + C_J)) + Z10*(-C1*Z00 + C1*Z10)) + (C1*Z11*Z12 + C2*Z21*Z22)*(C2*Z20*Z21 + Z11*(-C1*Z00 + C1*Z10))*(-C1*Z00*Z12 + C1*Z10*Z12 + C2*Z20*Z22) + (C1*Z11*Z12 + C2*Z21*Z22)*(C2*Z20*Z22 + Z12*(-C1*Z00 + C1*Z10))*(-C1*Z00*Z11 + C1*Z10*Z11 + C2*Z20*Z21))
    varsC02 =  [Z00, Z10, Z11, Z12, Z20, Z21, Z22]
    solC02 =  [{Z00: Z10 - Z11*Z20/Z21}]
    resC02 = quantize._sol_indep_of_vars(eqC02, varsC02, nonzero=dets)
    assert sum(abs(sym.simplify(eqC02.subs(resC02[i])) == 0) for i in range(len(resC02)))
    assert len(resC02) == 1

    eqC12 =  (-(C1*Z11*Z12 + C2*Z21*Z22)*(C2*Z20**2 + Z00*(-C1*Z10 + Z00*(C1 + C_J)) + Z10*(-C1*Z00 + C1*Z10)) + (C2*Z20*Z22 + Z12*(-C1*Z00 + C1*Z10))*(-C1*Z00*Z11 + C1*Z10*Z11 + C2*Z20*Z21))/((C1*Z11**2 + C2*Z21**2)*(C1*Z12**2 + C2*Z22**2)*(C2*Z20**2 + Z00*(-C1*Z10 + Z00*(C1 + C_J)) + Z10*(-C1*Z00 + C1*Z10)) - (C1*Z11**2 + C2*Z21**2)*(C2*Z20*Z22 + Z12*(-C1*Z00 + C1*Z10))*(-C1*Z00*Z12 + C1*Z10*Z12 + C2*Z20*Z22) - (C1*Z12**2 + C2*Z22**2)*(C2*Z20*Z21 + Z11*(-C1*Z00 + C1*Z10))*(-C1*Z00*Z11 + C1*Z10*Z11 + C2*Z20*Z21) - (C1*Z11*Z12 + C2*Z21*Z22)**2*(C2*Z20**2 + Z00*(-C1*Z10 + Z00*(C1 + C_J)) + Z10*(-C1*Z00 + C1*Z10)) + (C1*Z11*Z12 + C2*Z21*Z22)*(C2*Z20*Z21 + Z11*(-C1*Z00 + C1*Z10))*(-C1*Z00*Z12 + C1*Z10*Z12 + C2*Z20*Z22) + (C1*Z11*Z12 + C2*Z21*Z22)*(C2*Z20*Z22 + Z12*(-C1*Z00 + C1*Z10))*(-C1*Z00*Z11 + C1*Z10*Z11 + C2*Z20*Z21))
    varsC12 =  [Z00, Z10, Z11, Z12, Z20, Z21, Z22]
    solC12 =  [{Z00: Z10, Z11: 0, Z22: 0}, {Z00: Z10, Z12: 0, Z21: 0}]
    resC12 = quantize._sol_indep_of_vars(eqC12, varsC12, nonzero=dets)
    assert sum(abs(sym.simplify(eqC12.subs(resC12[i])) == 0) for i in range(len(resC12)))
    assert len(resC12) == 2

    eqL12 =  Z12*(-Z21/L3 + Z11*(L1 + L3)/(L1*L3)) + Z22*(-Z11/L3 + Z21*(L2 + L3)/(L2*L3))
    varsL12 =  [Z11, Z12, Z21, Z22]
    solL12 =  []
    resL12 = quantize._sol_indep_of_vars(eqL12, varsL12, nonzero=dets)
    assert len(resL12) == 0

    eq =  Z11*(Z20/L1 + Z00*(L1 - L2)/(2*L1*L2) + Z10*(L1 + L2)/(L1*L2)) + Z21*(-Z00/(2*L1) + Z10/L1 + Z20*(L1 + L3)/(L1*L3))
    sols = quantize._sol_indep_of_vars(eq, solve_vars)
    for x in [{Z21: 0, Z11: 0},
              {Z00: 0, Z10: 0, Z20: 0},
              {Z00: 2*Z10, Z11: 0, Z20: 0},
              {Z00: -2*Z10, Z20: -2*Z10, Z21: 0},
              {Z00: -2*Z10, Z11: -Z21, Z20: 0}]:     
        any_true = False
        for sol in sols:
             all_eq = [sym.Eq(x[0], x[1]) for x in sol.items()]
             sol_rewrite = sym.solve(all_eq, list(x.keys()), dict=True, simplify=True)
             if x in sol_rewrite:
                 any_true = True
                 break
        assert any_true

    x,y = sym.symbols("x,y", real=True)
    a,b = sym.symbols("a,b", real=True)

    # No solution indep of bad vars
    eq = (x+y/2)*a*b + b
    SOLVE_CACHE={}
    sol = quantize._sol_indep_of_vars(eq, [x,y])
    assert sol == []

    # Failing for some reason
    Z10, Z20 = sym.symbols("Z10, Z20", real=True)
    L_3 = sym.symbols("L_3", real=True, positive=True)
    expr = Z10*Z20/L_3
    sol = quantize._sol_indep_of_vars(expr, [Z10, Z20])
    assert {Z10: 0} in sol and {Z20: 0} in sol


    # Buggy one
    j_var = sym.symbols("J_1, J_2, J_3, J_4", real=True, positive=True)
    J_1, J_2, J_3, J_4 = j_var
    Z_var = sym.symbols("Z21, Z22, Z02, Z11, Z01, Z12", real=True)
    Z21, Z22, Z02, Z11, Z01, Z12 = Z_var
    Z_var = set([Z01, Z02, Z11, Z12, Z21, Z22])
    expr = J_3*Z21*Z22 + Z02*(-J_4*Z11 + Z01*(J_1 + J_4)) + Z12*(-J_4*Z01 + Z11*(J_2 + J_4))
    sol = quantize._sol_indep_of_vars(expr, Z_var)
    good_subs = {Z02: 0, Z12: 0, Z21: 0}
    assert any(s == good_subs for s in sol)


    # No bad vars present
    eq = x-y
    sol = quantize._sol_indep_of_vars(eq, [x,y])
    assert len(sol) == 1
    sol = sol[0]
    if x in sol:
        assert sol[x] == y
    else:
        assert sol[y] == x

    # Solution exists independent of bad vars
    eq = (x+y/2)*a*b
    sol = quantize._sol_indep_of_vars(eq, [x,y])
    assert len(sol) == 1
    sol = sol[0]
    if x in sol:
        assert sol[x] == -y/2
    else:
        assert sol[y] == -2*x
   
    # There is if you remove b
    eq = (x+y/2)*a + b
    sol = quantize._sol_indep_of_vars(eq, [x,y,b])
    assert len(sol) == 1
    sol = sol[0]
    if x in sol:
        assert sol[x] == -y/2
    else:
        assert sol[y] == -2*x
    assert sol[b] == 0

    # Multiple solutions
    eq = x*a*b
    sol = quantize._sol_indep_of_vars(eq, [x,y,b])
    assert len(sol) == 2
    if x in sol[0]:
        assert sol[0][x] == 0 and sol[1][b] == 0
    else:
        assert sol[1][x] == 0 and sol[0][b] == 0


    x, y, a = sym.symbols("x y a", real=True)
    # product giving two simple independent solutions: y = 0 OR x = -2
    eq = (x + 2)*y
    sols = quantize._sol_indep_of_vars(eq, [x, y])
    # expect both solutions present (order not important)
    assert any(s == {y: 0} for s in sols)
    assert any(s.get(x) == -2 for s in sols)

    # no solve vars present in expression -> no solutions
    eq2 = a + 1
    sols2 = quantize._sol_indep_of_vars(eq2, [x, y])
    assert sols2 == []



def test__unique_products():

    Z00, Z11, Z12, Z12, Z12, Z21, Z20, Z02, Z22, Z10 = sym.symbols("Z00, Z11, Z12, Z12, Z12, Z21, Z20, Z02, Z22, Z10", real=True)
    C1, C2 = sym.symbols("C1, C2", real=True, positive=True)

    expr = (-4*C1*C2*Z00*Z11*Z12*Z22 - 2*C1*C2*Z00*Z11*Z22**2 +
            4*C1*C2*Z00*Z12**2*Z21 + 2*C1*C2*Z00*Z12*Z21*Z22 - 
            4*C1*C2*Z10*Z11*Z22**2 + 4*C1*C2*Z10*Z12*Z21*Z22 + 
            4*C1*C2*Z11*Z12*Z20*Z22 - 4*C1*C2*Z12**2*Z20*Z21)
    

    prods = quantize._unique_products(expr, [Z00, Z11, Z12, Z12, Z12, Z21, Z20, Z02, Z22, Z10])
    assert len(prods) == 1
    assert prods[C1*C2] == sym.simplify(expr/(C1*C2))


    # Solution that only has a single thing
    x,y = sym.symbols("x,y", real=True)
    a,b = sym.symbols("a,b", real=True)

    eq = (x+y/2)*a*b + b
    prods = quantize._unique_products(eq, [x,y])
    assert prods == {a*b: x + y/2, b: 1}

    eq = x*y
    prods = quantize._unique_products(eq, [a,b])
    assert eq in prods
    assert prods[eq] == 1

    eq = x*y
    prods = quantize._unique_products(eq, [x,y])
    assert 1 in prods
    assert prods[1] == eq


    eq = x + y
    prods = quantize._unique_products(eq, [x,y])
    assert prods[1] == eq

    eq = (x + y)*(a + b)**2 + a - y*b
    prods = quantize._unique_products(eq, [x,y])
    for pr in [a**2, a*b, b**2, a, b]:
        assert pr in prods

    eq = (x + y)*(a + 1/b)**2 + 1/a - y*b
    prods = quantize._unique_products(eq, [x,y])
    for pr in [a**2, a/b, 1/b**2, 1/a, b]:
        assert pr in prods


def test__expr_valid():

    x,y = sym.symbols("x,y", real=True)
    a,b = sym.symbols("a,b", real=True)

    eq = (x+y/2)*a*b + b
    assert quantize._expr_valid(eq, {x:y}) == True
    assert quantize._expr_valid(eq, {x:y, a:b}) == True


    eq = (x+y/2)*a*b + 1/b
    assert quantize._expr_valid(eq, {x:y}) == True
    assert quantize._expr_valid(eq, {x:y, b:0}) == False


def test__extract_denom():

    x,y = sym.symbols("x,y", real=True)
    a,b = sym.symbols("a,b", real=True)

    M = sym.Matrix([[1/x, 0, 2],
                    [1/x + 1/y, sym.Rational(1,100), 0],
                    [(a + b)/(x*y + a + b), 0, 3/y]])
    denom = quantize._extract_denom(M)
    assert len(denom) == 4
    for entry in [x, y, x*y + a + b, x*y]:
        assert entry in denom


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


def test__find_Z_instance():

    # TWO FROM DIFFERENT RUNS
    # wJT Matrix([[-Z10*Z11/(Z01 - Z11), Z01 - Z11, 0], [Z10*(Z01 - 2*Z11)/(Z01 - Z11), Z01, 0], [Z10, Z11, 0]])
    # wJT Matrix([[-Z10*Z11/(Z01 - Z11), Z01 - Z11, 0], [Z10*(Z01 - 2*Z11)/(Z01 - Z11), Z01, 0], [Z10, Z11, 0]])
    # Matrix([[2*Z10*(Z01 - 2*Z11)/(3*(Z01 - Z11)) - Z10/3, 2*Z01/3 - Z11/3, 1/3], [-Z10*(Z01 - 2*Z11)/(3*(Z01 - Z11)) + 2*Z10/3, -Z01/3 + 2*Z11/3, 1/3], [-Z10*(Z01 - 2*Z11)/(3*(Z01 - Z11)) - Z10/3, -Z01/3 - Z11/3, 1/3]])
    # Matrix([[2*Z10*(Z01 - 2*Z11)/(3*(Z01 - Z11)) - Z10/3, 2*Z01/3 - Z11/3, 1/3], [-Z10*(Z01 - 2*Z11)/(3*(Z01 - Z11)) + 2*Z10/3, -Z01/3 + 2*Z11/3, 1/3], [-Z10*(Z01 - 2*Z11)/(3*(Z01 - Z11)) - Z10/3, -Z01/3 - Z11/3, 1/3]])



    # The troublesome one
    # Z10, Z01, Z11 = sym.symbols("Z10, Z01, Z11", real=True)
    
    
    # Z = sym.Matrix([[2*Z10*(Z01 - 2*Z11)/(3*(Z01 - Z11)) - Z10/3, 2*Z01/3 - Z11/3, 1/3], [-Z10*(Z01 - 2*Z11)/(3*(Z01 - Z11)) + 2*Z10/3, -Z01/3 + 2*Z11/3, 1/3], [-Z10*(Z01 - 2*Z11)/(3*(Z01 - Z11)) - Z10/3, -Z01/3 - Z11/3, 1/3]])
    # wJ = sym.Matrix([[1, 1, 0], [-1, 0, 1], [0, -1, -1]])
    # var_types =  {'compact': [], 'extended': [0, 1], 'harmonic': [], 'free': [], 'frozen': [], 'sigma': [2]}
    # var_list =  [Z01, Z11, Z10]
    # Z_hash2 = quantize._find_Z_instance(Z, var_list, var_types=var_types, wJ=wJ, nonzero=quantize._extract_denom(Z))
    # Z_hash2 = sym.nsimplify(Z_hash2, rational=True)
    # # wJT Matrix([[-Z10*Z11/(Z01 - Z11), Z01 - Z11, 0], [Z10*(Z01 - 2*Z11)/(Z01 - Z11), Z01, 0], [Z10, Z11, 0]])



    # Z =  sym.Matrix([[2*Z10*(Z01 - 2*Z11)/(3*(Z01 - Z11)) - Z10/3, 2*Z01/3 - Z11/3, 1/3], [-Z10*(Z01 - 2*Z11)/(3*(Z01 - Z11)) + 2*Z10/3, -Z01/3 + 2*Z11/3, 1/3], [-Z10*(Z01 - 2*Z11)/(3*(Z01 - Z11)) - Z10/3, -Z01/3 - Z11/3, 1/3]])
    # var_types =  {'compact': [], 'extended': [0, 1], 'harmonic': [], 'free': [], 'frozen': [], 'sigma': [2]}
    # # var_list =  [Z11, Z10, Z01]
    # wJ =  sym.Matrix([[1, 1, 0], [-1, 0, 1], [0, -1, -1]])
    # Z_hash2 = quantize._find_Z_instance(Z, var_list, var_types=var_types, wJ=wJ, nonzero=quantize._extract_denom(Z))
    # Z_hash2 = sym.nsimplify(Z_hash2, rational=True)
    # _, key, _ = quantize._maximize_wT((wJ.transpose()*Z_hash2)[:, :2])
    # assert key == "544555"


    # Solution that only has a single thing
    x,y = sym.symbols("x,y", real=True)
    a,b = sym.symbols("a,b", real=True)
    v = [x,y,a,b]

    Z = sym.Matrix([[x,1],
                    [a,1]])
    wJ = sym.Matrix([[1],
                     [-1]])
    Zsub = quantize._find_Z_instance(Z, v, wJ=wJ, var_types={"extended":[0], "sigma":[1]})
    
    assert Z.det().simplify() != 0
    assert sym.im(Zsub).is_zero_matrix
    wJT_trans = wJ.transpose()*Zsub
    assert all(sym.simplify(x) in quantize.WJ_VALS for x in wJT_trans)

    vals = [0, 2, -2]
    Zsub = quantize._find_Z_instance(Z, v, vals=vals, wJ=wJ, var_types={"extended":[0], "sigma":[1]})
    assert Z.det().simplify() != 0
    assert sym.im(Zsub).is_zero_matrix
    wJT_trans = wJ.transpose()*Zsub
    assert all(sym.simplify(x) in vals for x in wJT_trans)


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
   

def test_H_hash():


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
    assert hashes[0] == "100_0-_0_0-_0-"
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
    assert hashes[0] == '111_1-100_0_0-000_0-000'
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
    assert hash == '111_1-100_4_1-001_3-111'
    hash, _ = quantize.H_hash(cTransInv, lTrans, wJtTrans, EJ, var_types, ordering_matters=False)
    assert hash == '111_1_4_1_3'


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

    # Debugging example
    edges = [(1, 2), (3, 4), (1, 3), (2, 4)]
    circuit = [("L_1", "C_1"), ("L_2", "C_2"), ("L_3",), ("J_1",)]
    cMat = quantize.gen_cap_mat(circuit, edges)
    lMat = quantize.gen_ind_mat(circuit, edges)
    edges = utils.renumber_nodes(edges)
    wJ, EJ = quantize.gen_w(circuit, edges, w_elem="J", return_params=True)
    Z0, var_types = quantize.var_trans_basis(circuit, edges)
    Z = quantize.secondary_decouple(Z0, var_types, cMat, lMat)
    Z = quantize._find_Z_instance(Z, Z.free_symbols, wJ=wJ, var_types=var_types)
    Z = sym.simplify(Z0*Z)
    cInvTrans = (Z.transpose()*cMat*Z)[:-1, :-1].inv()
    lTrans = Z.transpose()*lMat*Z
    wJtTrans = wJ.transpose()*Z
    # cTransInv, lTrans, wJtTrans, EJ, var_types
    val, Z_perm = quantize.H_hash(cInvTrans, lTrans, wJtTrans, EJ, var_types)
    assert val == '012_0-000_3_2-011_1-001'

    # Make sure it's not nan
    circuit, edges = ([('J',), ('J', 'L'), ('C', 'J', 'L')], [(0, 1), (0, 2), (1, 2)])
    Z, var_types = quantize.var_trans_basis(circuit, edges)
    assert var_types["extended"] == [0,1]
    assert var_types["sigma"] == [2]
    assert Z.det() != 0
    for i in range(3):
        for j in range(3):
            assert Z[i,j].is_finite
    cMat = quantize.gen_cap_mat(circuit, edges)
    lMat = quantize.gen_ind_mat(circuit, edges)
    wJ = quantize.gen_w(circuit, edges, w_elem="J")
    Z2 = quantize.secondary_decouple(Z, var_types, cMat, lMat, return_instance=True)
    for i in range(3):
        for j in range(3):
            assert Z2[i,j].is_finite



    # Very slow right now with identical parameters
    # edges = [(1, 2), (3, 4), (1, 3), (2, 4)]
    # circuit = [("L", "C"), ("L", "C"), ("L",), ("J",)]
    # cMat = quantize.gen_cap_mat(circuit, edges)
    # lMat = quantize.gen_ind_mat(circuit, edges)
    # Z0, var_types = quantize.var_trans_basis(circuit, edges)
    # Z = Z0*quantize.secondary_decouple(Z, var_types, cMat, lMat)
    # cMat = quantize.gen_cap_mat(circuit, edges)
    # lMat = quantize.gen_ind_mat(circuit, edges)
    # wJ = quantize.gen_w(circuit, edges, w_elem="J")
    # hashes = quantize.H_hash(quantize._find_Z_instance(Z, Z.free_symbols), var_types, cMat, lMat, wJ)
    # assert hashes[0] == "012_0-000_3_0-000_3-111"


    # Example from secondary transformation section
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
    Z2 = quantize.secondary_decouple(Z0, var_types, cMat, lMat)
    Z_comp = quantize._find_Z_instance(Z2, Z2.free_symbols, wJ=wJ, var_types=var_types)
    assert quantize._equal_up_to_column_swaps_and_shift_and_sign(Z_comp,
                                                                 sym.simplify(sym.Matrix([[1.0, 0, 0],[1/2, 1/2, 0],[0, 0, 1]]), rational=True),
                                                                 shifts=[sym.ones(3,1)])
    Z = sym.simplify(Z0*Z_comp)
    cInvTrans = (Z.transpose()*cMat*Z)[:-1, :-1].inv()
    lTrans = Z.transpose()*lMat*Z
    wJtTrans = wJ.transpose()*Z
    # cTransInv, lTrans, wJtTrans, EJ, var_types
    val, Z_perm = quantize.H_hash(cInvTrans, lTrans, wJtTrans, EJ, var_types)
    assert val == "011_0-0_0_0-0_0-0"



def test_choose_Z():

    for i in range(1):
        edges = [(1, 2), (3, 4), (1, 3), (2, 4)]
        circuit = [("L1", "C1"), ("L2", "C2"), ("L3",), ("J",)]
        # Swap columns to see if hashing will pick right column order
        cMat = quantize.gen_cap_mat(circuit, edges)
        lMat = quantize.gen_ind_mat(circuit, edges)
        wJ = quantize.gen_w(circuit, edges, w_elem="J")
        t0 = time.time()
        Z, var_types, hash = quantize.choose_Z(circuit, edges, return_instance=False)
        Z = Z[0]
        print(time.time() - t0)
        # Z2, var_types2, hash2 = quantize.choose_Z(circuit, edges, return_instance=True)
        # breakpoint()
        print(hash)
        assert hash == "012_0-000_3_2-011_1-001"

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
            Z, var_types, hash = quantize.choose_Z(circuit, edges, return_instance=True)
            Z = Z[0]
            assert var_types["extended"] == [0,1]
            assert var_types["sigma"] == [2]
            if edges not in res:
                res[edges] = set()
            res[edges].add(quantize._maximize_wT((wJ.transpose()*Z)[:,:2])[1])
        for e in res:
            res[e] = frozenset(res[e])

        assert len(set(res.values())) == 1
    
    
    circuit, edges = ([('J',), ('J', 'L'), ('C', 'J', 'L')], [(0, 1), (0, 2), (1, 2)])
    Z, var_types, hash = quantize.choose_Z(circuit, edges, return_instance=True)
    Z = Z[0]
    assert var_types["extended"] == [0,1]
    assert var_types["sigma"] == [2]
    for i in range(3):
        for j in range(3):
            assert Z[i,j].is_finite

    # Transmon Molecule + cap
    edges = [(1, 2), (3, 4), (1, 3), (2, 4)]
    circuit = [("J", "C"), ("J","C"), ("C",), ("C",)]
    Z, var_types, hash = quantize.choose_Z(circuit, edges)
    Z = Z[0]
    assert var_types["compact"] == [0, 1]
    assert var_types["free"] == [2]
    assert var_types["sigma"] == [3]
    assert quantize._equal_up_to_column_shift_and_sign(
                    sym.nsimplify(sym.Matrix([[1/2, 0, 1/2, 1],
                                     [-1/2,0, 1/2, 1],
                                     [0, 1/2, -1/2,1],
                                     [0, -1/2,-1/2,1]]), rational=True), Z)
    assert hash == "200_0-0_1_0-0_1-1"


     # Transmon
    edges = [(0, 1)]
    circuit = [("J", "C")]
    Z, var_types, hash = quantize.choose_Z(circuit, edges)
    Z = Z[0]
    assert hash == "100_0-_0_0-_0-"
    assert quantize._find_equiv_mats(sym.Matrix([[1, 1],
                                                 [0, 1]]), 
                                     [Z]) == [0]

    # Zero-pi
    circuit = [("J",),("J",), ("L",), ("L",), ("C",), ("C",)]
    edges = [(0, 1), (2, 3), (0, 3), (1, 2), (0, 2), (1, 3)]
    cMat = quantize.gen_cap_mat(circuit, edges)
    lMat = quantize.gen_ind_mat(circuit, edges)
    wJ = quantize.gen_w(circuit, edges, w_elem="J")
    Z, var_types, hash = quantize.choose_Z(circuit, edges, return_instance=True)
    Z = Z[0]
    assert quantize._find_equiv_mats(sym.Matrix([[1, 1],
                                                [0, 0],
                                                [0, 1],
                                                [1, 0]]), 
                                    [Z[:,:2]]) == [0]
    assert hash == '111_1-100_0_0-000_0-000'

    circuit = [("J1",), ("J2",), ("L1",), ("L2",), ("C1",), ("C2",)]
    Z, var_types, hash = quantize.choose_Z(circuit, edges)
    cMat = quantize.gen_cap_mat(circuit, edges)
    lMat = quantize.gen_ind_mat(circuit, edges)
    assert hash == '111_1-100_4_1-001_3-111'

    circuit = [("J",),("J",), ("L1",), ("L2",), ("C1",), ("C2",)]
    edges = [(0, 1), (2, 3), (0, 3), (1, 2), (0, 2), (1, 3)]
    cMat = quantize.gen_cap_mat(circuit, edges)
    lMat = quantize.gen_ind_mat(circuit, edges)
    wJ = quantize.gen_w(circuit, edges, w_elem="J")
    Z, var_types, hash = quantize.choose_Z(circuit, edges)
    Z = Z[0]
    assert quantize._find_equiv_mats(sym.Matrix([[1, 1],
                                                [0, 0],
                                                [0, 1],
                                                [1, 0]]), 
                                    [Z[:,:2]]) == [0]
    assert hash == '111_1-100_2_1-001_1-010'


    edges = [(1, 2), (3, 4), (1, 3), (2, 4)]
    circuit = [("L1", "C1"), ("L2", "C2"), ("L3",), ("J",)]
    # Swap columns to see if hashing will pick right column order
    cMat = quantize.gen_cap_mat(circuit, edges)
    lMat = quantize.gen_ind_mat(circuit, edges)
    wJ = quantize.gen_w(circuit, edges, w_elem="J")
    t0 = time.time()
    Z, var_types, hash = quantize.choose_Z(circuit, edges)
    tf = time.time()
    print(tf-t0)
    assert hash == "012_0-000_3_2-011_1-001"



    circuit, edges = [('J',), ('J',), ('J',)], [(0, 1), (0, 2), (1, 2)]
    cMat = quantize.gen_cap_mat(circuit, edges)
    lMat = quantize.gen_ind_mat(circuit, edges)
    wJ = quantize.gen_w(circuit, edges, w_elem="J")
    Z, var_types, hash = quantize.choose_Z(circuit, edges)
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
    assert hash == "200_1-1_1_0-0_1-1"
    

    # Bifluxon -- Tests the equal parameter tiebreaker
    edges = [(1, 2), (1, 3), (2, 3)]
    circuit = [("J",), ("J",), ("L",)]
    Z, var_types, hash = quantize.choose_Z(circuit, edges)
    Z = Z[0]
    assert var_types["compact"] == [0]
    assert var_types["extended"] == [1]
    assert var_types["sigma"] == [2]
    cMat = quantize.gen_cap_mat(circuit, edges)
    lMat = quantize.gen_ind_mat(circuit, edges)
    wJ = quantize.gen_w(circuit, edges, w_elem="J")
    assert quantize._equal_up_to_column_shift_and_sign(
                    sym.nsimplify(sym.Matrix([[2/3, 0, 1],
                                     [-1/3, -1, 1],
                                     [-1/3, 1, 1]]), rational=True), Z)
    assert hash == "110_1-1_0_0-0_0-0"

    edges = [(1, 2), (1, 3), (2, 3)]
    circuit = [("J1",), ("J2",), ("L",)]
    Z, var_types, hash = quantize.choose_Z(circuit, edges)
    Z = Z[0]
    assert var_types["compact"] == [0]
    assert var_types["extended"] == [1]
    assert var_types["sigma"] == [2]
    cMat = quantize.gen_cap_mat(circuit, edges)
    lMat = quantize.gen_ind_mat(circuit, edges)
    wJ = quantize.gen_w(circuit, edges, w_elem="J")
    assert hash == "110_1-1_1_0-0_1-1"


    # Transmon Molecule
    edges = [(1, 2), (3, 4), (1, 3), (2, 4)]
    circuit = [("J",), ("J",), ("C",), ("C",)]
    Z, var_types, hash = quantize.choose_Z(circuit, edges)
    Z = Z[0]
    assert var_types["compact"] == [0, 1]
    assert var_types["free"] == [2]
    assert var_types["sigma"] == [3]
    assert quantize._equal_up_to_column_shift_and_sign(
                    sym.nsimplify(sym.Matrix([[1/2, 0, 1/2, 1],
                                     [-1/2,0, 1/2, 1],
                                     [0, 1/2, -1/2,1],
                                     [0, -1/2,-1/2,1]]), rational=True), Z)
    assert hash == "200_0-0_1_0-0_1-1"

    
    # Fully linear
    edges = [(1, 2), (3, 4), (1, 3), (2, 4)]
    circuit = [("L",), ("L",), ("C",), ("C",)]
    Z, var_types, hash = quantize.choose_Z(circuit, edges)
    Z = Z[0]
    assert var_types["harmonic"] == [0]
    assert var_types["free"] == [1]
    assert var_types["frozen"] == [2]
    assert var_types["sigma"] == [3]
    assert hash == "001_0-_0_0-_0-"

    
    # Example from secondary transformation section
    circuit = [("C", "L1"), ("J", "L2")]
    edges = [(1, 2), (1, 3)]
    edges = utils.renumber_nodes(edges)
    cMat = quantize.gen_cap_mat(circuit, edges)
    Z, var_types, hash = quantize.choose_Z(circuit, edges)
    Z = Z[0]
    assert var_types["extended"] == [0]
    assert var_types["harmonic"] == [1]
    assert var_types["sigma"] == [2]
    assert hash == "011_0-0_0_0-0_0-0"


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
    # test_H_hash()
    # test__find_Z_instance_deterministic()
    # test__fully_compatible_set()
    # test__find_Z_instance()
    # test__unique_products()
    # test__fully_compatible_set()
    # test__sol_indep_of_vars()

    # test__maximally_compatible_set()
    test_choose_Z()

    # test__vec_space_overlap()
    # test__independent_from()
    # test__nonzero_entries_str()
    # test_choose_Z()

    # test_secondary_decouple()

    # test__maximize_wT()
    # test__wT_key()
    # test_var_trans_basis()


    # test__to_eq_list()
    # test_H_hash()
    # test_incidence_to_square()
    # test_gen_cap_mat()
    # test_var_trans_basis()
    # test__cached_solve()

    return

if __name__ == "__main__":

    main()