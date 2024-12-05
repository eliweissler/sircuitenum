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

    test_quantize_circuit()
        