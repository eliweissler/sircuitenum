import sympy as sym
import numpy as np

from sircuitenum import quantize
from sircuitenum import qpackage_interface as pi
from sircuitenum import utils


def test_gen_cap_mat():

    # Transmon
    edges = [(0, 1)]
    circuit = [("J", "C")]
    obj = pi.to_SCqubits(circuit, edges)
    Z = sym.Matrix(obj.transformation_matrix)

    C = quantize.gen_cap_mat(circuit, edges)
    ans = r'\left[\begin{matrix}C + C_{J} & - C - C_{J}\\- C - C_{J} & C + C_{J}\end{matrix}\right]'
    assert sym.latex(C, order="grlex") == ans


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

    J = quantize.gen_junc_pot(circuit, edges, th_vec, cob=Z)
    ans = r'- E_{J} \cos{\left(p_{1} \right)}'
    assert sym.latex(J, order="grlex") == ans


def test_gen_ind_mat():

    # Transmon
    edges = [(0, 1)]
    circuit = [("J", "C")]
    obj = pi.to_SCqubits(circuit, edges)
    Z = sym.Matrix(obj.transformation_matrix)

    L = quantize.gen_ind_mat(circuit, edges)
    ans = r'\left[\begin{matrix}0 & 0\\0 & 0\end{matrix}\right]'
    assert sym.latex(L, order="grlex") == ans

    # Fluxonium
    edges = [(0, 1)]
    circuit = [("J", "L")]
    obj = pi.to_SCqubits(circuit, edges)
    Z = sym.Matrix(obj.transformation_matrix)
    L = quantize.gen_ind_mat(circuit, edges)
    ans = r'\left[\begin{matrix}\frac{1}{L} & - \frac{1}{L}\\- \frac{1}{L} & \frac{1}{L}\end{matrix}\right]'

    assert sym.latex(L, order="grlex") == ans


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
    assert w == sym.Matrix([[]])
    
    # 0-pi
    edges = [(1, 2), (3, 4), (1, 4), (2, 3), (1, 3), (2, 4)]
    circuit = [("J",), ("J",), ("L",), ("L",), ("C",), ("C",)]
    w = quantize.gen_w(circuit, edges, w_elem="J")
    assert w == sym.Matrix([[1, 0],
                           [-1, 0],
                           [0, 1],
                           [0, -1]])


def test__indices_to_arr():

    assert quantize._indices_to_arr(3, [1]) == sym.Matrix([[0.], [1.], [0.]])

def test__islands_to_vectors():

    edges = [(1, 2), (3, 4), (1, 4), (2, 3), (1, 3), (2, 4)]
    circuit = [("J",), ("J",), ("L",), ("L",), ("C",), ("C",)]
    vecs = quantize._islands_to_vectors(circuit, edges, ["J", "L"])
    v1 = sym.Matrix([[1.], [0.], [1.], [0.]])
    v2 = sym.Matrix([[0.], [1.], [0.], [1.]])
    assert len(vecs) == 1
    assert vecs[0] in [v1, v2]

def test_find_islands():

    # Transmon
    edges = [(0, 1)]
    circuit = [("J", "C")]
    islands = quantize.find_islands(circuit, edges, links = ["J"])
    assert islands == [(0, 1)]

    
    # 0-pi
    edges = [(1, 2), (3, 4), (1, 4), (2, 3), (1, 3), (2, 4)]
    circuit = [("J",), ("J",), ("L",), ("L",), ("C",), ("C",)]
    islands = quantize.find_islands(circuit, edges, links = ["J"])
    assert islands == [(1, 2, 3, 4)]
    islands = quantize.find_islands(circuit, edges, links = ["J", "C"])
    assert all(x in [(1, 4), (2, 3)] for x in islands)
    assert len(islands) == 2
    islands = quantize.find_islands(circuit, edges, links = ["J", "L"])
    assert all(x in [(1, 3), (2, 4)] for x in islands)
    assert len(islands) == 2


    edges = [(1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7)]
    circuit = [("J",), ("J",), ("J",), ("L",), ("J",), ("L",)]
    islands = quantize.find_islands(circuit, edges, links = ["L"])
    for x in islands:
        assert x in [(1, 2, 3, 4), (5, 6), (7,)]
    assert len(islands) == 3

    # Fancy circuit
    circuit = [("J",), ("J",), ("L",), ("C",), ("C",), ("J",)]
    edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4)]


def test_gen_var_trans():

    edges = [(1, 2), (3, 4), (1, 4), (2, 3), (1, 3), (2, 4)]
    circuit = [("J",), ("J",), ("L",), ("L",), ("C",), ("C",)]
    Z = quantize.gen_var_trans(circuit, edges)
    quantize.quantize_circuit(circuit, edges, cob=Z, free=[4])


def test_decoupling_transformation():

    assert False


def test_compact_well_aligned():

    wT = sym.nsimplify((sym.Matrix([[1, 0, 1, 1],
                                    [0, -1, 0, 1],
                                    [1, 1, 0, -1],
                                    [1, 1, -1, 1]])), rational=True)
    assert quantize.compact_well_aligned(wT, 0) == True
    assert quantize.compact_well_aligned(wT, 1) == True
    assert quantize.compact_well_aligned(wT, 2) == True
    assert quantize.compact_well_aligned(wT, 3) == False
    assert quantize.compact_well_aligned(wT, 4) == False


    wT = sym.nsimplify((sym.Matrix([[1, 0, 0, 1/2],
                                    [0, 1/2, 0, 1],
                                    [0, 0, 1/2, -1],
                                    [1, 1/3, -1, 1]])), rational=True)
    assert quantize.compact_well_aligned(wT, 0) == True
    assert quantize.compact_well_aligned(wT, 1) == True
    assert quantize.compact_well_aligned(wT, 2) == False
    assert quantize.compact_well_aligned(wT, 3) == False
    assert quantize.compact_well_aligned(wT, 4) == False

    wT = sym.nsimplify((sym.Matrix([[1, 0, 0, 1/2],
                                    [0, 0, 1/2, 1],
                                    [0, 0, 1, -1],
                                    [0, -1, 0, 1]])), rational=True)
    assert quantize.compact_well_aligned(wT, 0) == True
    assert quantize.compact_well_aligned(wT, 1) == True
    assert quantize.compact_well_aligned(wT, 2) == True
    assert quantize.compact_well_aligned(wT, 3) == True
    assert quantize.compact_well_aligned(wT, 4) == False

def test_compact_alignment_transformation():

    edges = [(1, 2), (2, 3), (3, 4), (4,1)]
    edges = utils.zero_start_edges(edges)
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


def test__linearly_indep_rows():

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


def test__linearly_indep_row_sets():

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

def test__equal_up_to_column_sign():

    # Identical columns
    matrix1 = sym.Matrix([[1, 2], [3, 4]])
    matrix2 = sym.Matrix([[1, 2], [3, 4]])
    assert quantize._equal_up_to_column_sign(matrix1, matrix2) == True
    
    # Negative of column
    matrix1 = sym.Matrix([[1, 2], [3, 4]])
    matrix2 = sym.Matrix([[-1, 2], [-3, 4]])
    assert quantize._equal_up_to_column_sign(matrix1, matrix2) == True
    
    # With symbolic entries
    x, y = sym.symbols('x y')
    matrix1 = sym.Matrix([[x, 2], [y, 4]])
    matrix2 = sym.Matrix([[-x, 2], [-y, 4]])
    assert quantize._equal_up_to_column_sign(matrix1, matrix2) == True
    
    # With fractions
    matrix1 = sym.Matrix([[sym.Rational(1,2), sym.Rational(2,1)], [sym.Rational(3,4), sym.Rational(4,1)]])
    matrix2 = sym.Matrix([[-sym.Rational(1,2), sym.Rational(2,1)], [-sym.Rational(3,4), sym.Rational(4,1)]])
    assert quantize._equal_up_to_column_sign(matrix1, matrix2) == True
    
    # Different columns with symbols
    matrix1 = sym.Matrix([[x, 2], [y, 4]])
    matrix2 = sym.Matrix([[y, 2], [x, 4]])
    assert quantize._equal_up_to_column_sign(matrix1, matrix2) == False


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
    x, y = sym.symbols('x y')
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


def test_quantize_circuit():

    # Fluxonium
    edges = [(0, 1)]
    circuit = [("J", "L")]
    obj = pi.to_SCqubits(circuit, edges)
    Z = sym.Matrix(obj.transformation_matrix)

    H, qv, tv = quantize.quantize_circuit(circuit, edges, cob=Z, free=[2],
                                          return_vars=True)
    ans = '- E_{J} \\cos{\\left(\\hat{φ}_{1} \\right)} + \\frac{\\hat{φ}_{1}^{2}}{2 L} + \\frac{\\hat{q}_{1}^{2}}{2 C_{J}}'

    assert sym.latex(H, order="grlex") == ans

    # 0 - pi
    edges = [(1, 2), (3, 4), (1, 4), (2, 3), (1, 3), (2, 4)]
    circuit = [("J",), ("J",), ("L",), ("L",), ("C",), ("C",)]
    obj = pi.to_SCqubits(circuit, utils.zero_start_edges(edges))
    circuit = [("J1",), ("J2",), ("L1",), ("L2",), ("C1",), ("C2",)]
    Z = sym.Matrix(obj.transformation_matrix)

    H, qv, tv = quantize.quantize_circuit(circuit, edges, cob=Z, **obj.var_categories | {"frozen": [4]},
                                          return_vars=True)

    ans = '\\left(- E_{J1} - E_{J2}\\right) \\cos{\\left(\\hat{θ}_{1} \\right)} \\cos{\\left(\\hat{φ}_{3} \\right)} + \\left(E_{J1} - E_{J2}\\right) \\sin{\\left(\\hat{θ}_{1} \\right)} \\sin{\\left(\\hat{φ}_{3} \\right)} + \\frac{\\hat{n}_{1}^{2} \\left(C_{1} C_{J1} + C_{1} C_{J2} + C_{2} C_{J1} + C_{2} C_{J2}\\right)}{8 C_{1} C_{2} C_{J1} + 8 C_{1} C_{2} C_{J2} + 8 C_{1} C_{J1} C_{J2} + 8 C_{2} C_{J1} C_{J2}} + \\frac{\\hat{n}_{1} \\hat{q}_{2} \\left(C_{1} C_{J1} + C_{1} C_{J2} - C_{2} C_{J1} - C_{2} C_{J2}\\right)}{8 C_{1} C_{2} C_{J1} + 8 C_{1} C_{2} C_{J2} + 8 C_{1} C_{J1} C_{J2} + 8 C_{2} C_{J1} C_{J2}} + \\frac{\\hat{n}_{1} \\hat{q}_{3} \\left(- C_{1} C_{J1} + C_{1} C_{J2} - C_{2} C_{J1} + C_{2} C_{J2}\\right)}{4 C_{1} C_{2} C_{J1} + 4 C_{1} C_{2} C_{J2} + 4 C_{1} C_{J1} C_{J2} + 4 C_{2} C_{J1} C_{J2}} + \\frac{\\hat{q}_{2}^{2} \\left(C_{1} C_{J1} + C_{1} C_{J2} + C_{2} C_{J1} + C_{2} C_{J2} + 4 C_{J1} C_{J2}\\right)}{32 C_{1} C_{2} C_{J1} + 32 C_{1} C_{2} C_{J2} + 32 C_{1} C_{J1} C_{J2} + 32 C_{2} C_{J1} C_{J2}} + \\frac{\\hat{q}_{2} \\hat{q}_{3} \\left(- C_{1} C_{J1} + C_{1} C_{J2} + C_{2} C_{J1} - C_{2} C_{J2}\\right)}{8 C_{1} C_{2} C_{J1} + 8 C_{1} C_{2} C_{J2} + 8 C_{1} C_{J1} C_{J2} + 8 C_{2} C_{J1} C_{J2}} + \\frac{\\hat{q}_{3}^{2} \\left(4 C_{1} C_{2} + C_{1} C_{J1} + C_{1} C_{J2} + C_{2} C_{J1} + C_{2} C_{J2}\\right)}{8 C_{1} C_{2} C_{J1} + 8 C_{1} C_{2} C_{J2} + 8 C_{1} C_{J1} C_{J2} + 8 C_{2} C_{J1} C_{J2}} + \\frac{\\hat{φ}_{2}^{2} \\left(2 L_{1} + 2 L_{2}\\right)}{L_{1} L_{2}} + \\frac{\\hat{φ}_{2} \\hat{φ}_{3} \\left(2 L_{1} - 2 L_{2}\\right)}{L_{1} L_{2}} + \\frac{\\hat{φ}_{3}^{2} \\left(L_{1} + L_{2}\\right)}{2 L_{1} L_{2}}'

    assert sym.latex(H, order="grlex") == ans

    # Transmon with Drive
    edges = [(0, 1)]
    circuit = [("J", "C")]
    obj = pi.to_SCqubits(circuit, edges)

    Z = sym.Matrix(obj.transformation_matrix)
    Cv = sym.Matrix(np.array([sym.Symbol("C_c", real=True, positive=True), 0]).reshape((2, 1)))
    V = sym.Matrix(np.array([sym.Symbol("V_g", real=True, positive=True)]).reshape((1, 1)))
    H, qv, tv = quantize.quantize_circuit(circuit, edges, cob=Z, free=[2], return_vars=True, Cv=Cv, V=V)

    ans = '- \\frac{C_{c} V_{g} \\hat{q}_{1}}{C + C_{J}} - E_{J} \\cos{\\left(\\hat{φ}_{1} \\right)} + \\frac{\\hat{q}_{1}^{2}}{2 C + 2 C_{J}}'

    assert sym.latex(H, order="grlex") == ans


def test_remove_col_():

    Cv = sym.Matrix(np.array([sym.Symbol("C_c", real=True, positive=True), 0]).reshape((1, 2)))
    assert quantize.remove_col_(Cv, 1).shape[1] == 1


def test_remove_row_():

    Cv = sym.Matrix(np.array([sym.Symbol("C_c", real=True, positive=True), 0]).reshape((2, 1)))
    assert quantize.remove_row_(Cv, 1).shape[0] == 1


def test_H_hash():

    circuit = [("J",),("J",), ("L",), ("L",), ("C",), ("C",)]
    edges = [(0, 1), (2, 3), (0, 3), (1, 2), (0, 2), (1, 3)]

    # Check for correctness and consistency
    for i in range(10):
        assert quantize.H_hash(circuit, edges) == '3_272_02'
        assert quantize.H_hash(circuit, edges, symmetric=False) == '3_511_012'




if __name__ == "__main__":

    # from sircuitenum import enum

    # entry = utils.get_circuit_data_batch("../circuits.db", n_nodes=4, filter_str="WHERE unique_key LIKE 'n4_g5_c42871'").iloc[0]
    # circuit, edges = entry.circuit, entry.edges
    # obj = pi.to_SCqubits(circuit, edges, sym_cir=True, initiate_sym_calc=False)
    # Z, var_class = obj.variable_transformation_matrix()
    # var_class["free"] = [utils.get_num_nodes(edges)]
    # H, trans, H_class = enum.gen_hamiltonian(entry.circuit, entry.edges,
    #                                         cob=Z, var_class = var_class, symmetric=False)

    # test__linearly_indep_rows()
    # test__linearly_indep_row_sets()
    # test__equal_up_to_column_sign()
    test__equal_up_to_column_shift_and_sign()
    # test_compact_well_aligned()
    # test_compact_alignment_transformation()
        