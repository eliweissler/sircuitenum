import time
import pytest

import numpy as np
import sympy as sym

from sircuitenum import equationset as eqs


class TestMaximallyCompatibleSol:
    """Tests for the maximally_compatible_sol function."""

    def test_01_fractions_raise_error(self):
        """Test that inputs with fractions raise ValueError immediately."""
        x, y = sym.symbols('x y')
        terms = [[x / y - 1]]
        
        with pytest.raises(ValueError, match="Equations with fractions are not supported"):
            eqs.maximally_compatible_sol(terms)

    def test_02_basic_maximization_all_compatible(self):
        """
        Scenario: 3 terms. All are compatible with each other.
        Expected: Returns indices [(0, 1, 2)] and the solution.
        """
        x, y, z = sym.symbols('x y z')
        terms = [[x - 1], [y - 2], [z - 3]]
        
        keys, sols = eqs.maximally_compatible_sol(terms)
            
        # Assert
        assert keys == [(0, 1, 2)]
        assert len(sols) == 1
        assert sols[0][0] == {x: 1, y: 2, z: 3}

    def test_03_pairwise_conflict_pruning(self):
        """
        Scenario: T0 (x=1) and T1 (x=2) are incompatible.
        Expected: Max size is 2. Solutions: [(0, 2), (1, 2)].
        """
        x, y = sym.symbols('x y')
        terms = [[x - 1], [x - 2], [y - 1]]
        
            
        keys, sols = eqs.maximally_compatible_sol(terms)
        
        # Assert: (0,1) is invalid. (0,2) and (1,2) are valid.
        assert (0, 2) in keys
        assert (1, 2) in keys
        assert (0, 1) not in keys
        assert len(keys) == 2

        x, y = sym.symbols('x y')
        terms = [[x - 1], [x - 2], [y*x - 1]]
        
        # Two different terms, same incompatibility, different variable combinations
        keys, sols = eqs.maximally_compatible_sol(terms)
        assert len(sols) == len(keys) == 2
        for sol in [{x: 1, y: 1}, {x: 2, y: sym.nsimplify(1/2, rational=True)}]:
            assert sol == sols[0][0] or sol == sols[1][0]
        for key in [(0, 2), (1, 2)]:
            assert key in keys


    def test_04_nonzero_constraints(self):
        """
        Scenario: Enforce constraint 'y != 0'.
        Checks if Rabinowitsch equation is added and temp var cleaned up.
        """
        x, y, nzVar = sym.symbols('x y nzVar')
        terms = [[x - 1]]
        constraints = [y]
        
        keys, sols = eqs.maximally_compatible_sol(terms, nonzero_constraints=constraints)
        
        # Assert 2: Output cleaned
        result_sol = sols[0][0]
        assert nzVar not in result_sol

    def test_05_nonzero_post_check_filtering(self):
        """
        Scenario: Solution found, but violates 'must be nonzero' list.
        """
        x = sym.symbols('x')
        terms = [[x - 1]]
        must_be_nonzero = [x - 1]
        
        keys, sols = eqs.maximally_compatible_sol(terms, nonzero=must_be_nonzero)
        
        # Assert: Result empty
        assert keys == []
        assert sols == []

    def test_06_individual_unsolvable_skipped(self):
        """
        Scenario: T1 is inherently unsolvable. It should be pruned immediately.
        """
        x = sym.symbols('x')
        terms = [[x], [sym.sympify(1)]]
            
        keys, sols = eqs.maximally_compatible_sol(terms)
        
        # Assert: Only T0 returned
        assert keys == [(0,)]


def test_eq_indep_of_vars():

    C1, C2, C_J = sym.symbols("C1, C2, C_J", real=True, positive=True)
    L1, L2, L3 = sym.symbols("L1, L2, L3", real=True, positive=True)
    Z00, Z11, Z12, Z21, Z20, Z02, Z22, Z10 = sym.symbols("Z00, Z11, Z12, Z21, Z20, Z02, Z22, Z10", real=True)
    solve_vars = [Z00, Z11, Z12, Z21, Z20, Z02, Z22, Z10]
    dets = [Z00*(Z11*Z22 - Z12*Z21), C1*C2*C_J*Z00**2*(Z11**2*Z22**2 - 2*Z11*Z12*Z21*Z22 + Z12**2*Z21**2)]

    eqL02 = Z12*(-Z20/L3 - Z00/L1 + Z10*(L1 + L3)/(L1*L3)) + Z22*(-Z10/L3 + Z20*(L2 + L3)/(L2*L3))
    varsL02 = [Z00, Z10, Z12, Z20, Z22]
    equations, denom = eqs.eq_indep_of_vars(eqL02, varsL02)
    assert len(equations) == 3
    for factor in [L1,L2,L3]:
        assert factor in denom.free_symbols
    for eq in [Z10*Z12 - Z10*Z22 - Z12*Z20 + Z20*Z22, Z20*Z22, Z12*(-Z00 + Z10)]:
        assert any(sym.simplify(eq - e) == 0 for e in equations)





def test_sol_indep_of_vars():
    C1, C2, C_J = sym.symbols("C1, C2, C_J", real=True, positive=True)
    L1, L2, L3 = sym.symbols("L1, L2, L3", real=True, positive=True)
    Z00, Z11, Z12, Z21, Z20, Z02, Z22, Z10 = sym.symbols("Z00, Z11, Z12, Z21, Z20, Z02, Z22, Z10")
    solve_vars = [Z00, Z11, Z12, Z21, Z20, Z02, Z22, Z10]
    dets = [Z00*(Z11*Z22 - Z12*Z21), C1*C2*C_J*Z00**2*(Z11**2*Z22**2 - 2*Z11*Z12*Z21*Z22 + Z12**2*Z21**2)]

    eqL02 = Z12*(-Z20/L3 - Z00/L1 + Z10*(L1 + L3)/(L1*L3)) + Z22*(-Z10/L3 + Z20*(L2 + L3)/(L2*L3))
    varsL02 = [Z00, Z10, Z12, Z20, Z22]
    resL02 = eqs.sol_indep_of_vars(eqL02, varsL02, nonzero=dets)
    assert sum(abs(sym.simplify(eqL02.subs(resL02[i])) == 0) for i in range(len(resL02)))
    assert len(resL02) == 3

    eqC01 = (-(C1*Z12**2 + C2*Z22**2)*(C2*Z20*Z21 + Z11*(-C1*Z00 + C1*Z10)) + (C1*Z11*Z12 + C2*Z21*Z22)*(C2*Z20*Z22 + Z12*(-C1*Z00 + C1*Z10)))/((C1*Z11**2 + C2*Z21**2)*(C1*Z12**2 + C2*Z22**2)*(C2*Z20**2 + Z00*(-C1*Z10 + Z00*(C1 + C_J)) + Z10*(-C1*Z00 + C1*Z10)) - (C1*Z11**2 + C2*Z21**2)*(C2*Z20*Z22 + Z12*(-C1*Z00 + C1*Z10))*(-C1*Z00*Z12 + C1*Z10*Z12 + C2*Z20*Z22) - (C1*Z12**2 + C2*Z22**2)*(C2*Z20*Z21 + Z11*(-C1*Z00 + C1*Z10))*(-C1*Z00*Z11 + C1*Z10*Z11 + C2*Z20*Z21) - (C1*Z11*Z12 + C2*Z21*Z22)**2*(C2*Z20**2 + Z00*(-C1*Z10 + Z00*(C1 + C_J)) + Z10*(-C1*Z00 + C1*Z10)) + (C1*Z11*Z12 + C2*Z21*Z22)*(C2*Z20*Z21 + Z11*(-C1*Z00 + C1*Z10))*(-C1*Z00*Z12 + C1*Z10*Z12 + C2*Z20*Z22) + (C1*Z11*Z12 + C2*Z21*Z22)*(C2*Z20*Z22 + Z12*(-C1*Z00 + C1*Z10))*(-C1*Z00*Z11 + C1*Z10*Z11 + C2*Z20*Z21))
    varsC01 = [Z00, Z10, Z11, Z12, Z20, Z21, Z22]
    resC01 = eqs.sol_indep_of_vars(eqC01, varsC01, nonzero=dets)
    assert sum(abs(sym.simplify(eqC01.subs(resC01[i])) == 0) for i in range(len(resC01)))
    assert not any(d.subs(resC01[i]) == 0 for d in dets for i in range(len(resC01)))
    assert len(resC01) == 2

    eqL01 = Z11*(-Z20/L3 - Z00/L1 + Z10*(L1 + L3)/(L1*L3)) + Z21*(-Z10/L3 + Z20*(L2 + L3)/(L2*L3))
    varsL01 = [Z00, Z10, Z11, Z20, Z21]
    resL01 = eqs.sol_indep_of_vars(eqL01, varsL01, nonzero=dets)
    assert sum(abs(sym.simplify(eqL01.subs(resL01[i])) == 0) for i in range(len(resL01)))
    assert not any(d.subs(resC01[i]) == 0 for d in dets for i in range(len(resC01)))
    assert len(resL01) == 3

    eqC02 = (-(C1*Z11**2 + C2*Z21**2)*(C2*Z20*Z22 + Z12*(-C1*Z00 + C1*Z10)) + (C1*Z11*Z12 + C2*Z21*Z22)*(C2*Z20*Z21 + Z11*(-C1*Z00 + C1*Z10)))/((C1*Z11**2 + C2*Z21**2)*(C1*Z12**2 + C2*Z22**2)*(C2*Z20**2 + Z00*(-C1*Z10 + Z00*(C1 + C_J)) + Z10*(-C1*Z00 + C1*Z10)) - (C1*Z11**2 + C2*Z21**2)*(C2*Z20*Z22 + Z12*(-C1*Z00 + C1*Z10))*(-C1*Z00*Z12 + C1*Z10*Z12 + C2*Z20*Z22) - (C1*Z12**2 + C2*Z22**2)*(C2*Z20*Z21 + Z11*(-C1*Z00 + C1*Z10))*(-C1*Z00*Z11 + C1*Z10*Z11 + C2*Z20*Z21) - (C1*Z11*Z12 + C2*Z21*Z22)**2*(C2*Z20**2 + Z00*(-C1*Z10 + Z00*(C1 + C_J)) + Z10*(-C1*Z00 + C1*Z10)) + (C1*Z11*Z12 + C2*Z21*Z22)*(C2*Z20*Z21 + Z11*(-C1*Z00 + C1*Z10))*(-C1*Z00*Z12 + C1*Z10*Z12 + C2*Z20*Z22) + (C1*Z11*Z12 + C2*Z21*Z22)*(C2*Z20*Z22 + Z12*(-C1*Z00 + C1*Z10))*(-C1*Z00*Z11 + C1*Z10*Z11 + C2*Z20*Z21))
    varsC02 = [Z00, Z10, Z11, Z12, Z20, Z21, Z22]
    resC02 = eqs.sol_indep_of_vars(eqC02, varsC02, nonzero=dets)
    assert sum(abs(sym.simplify(eqC02.subs(resC02[i])) == 0) for i in range(len(resC02)))
    assert not any(d.subs(resC01[i]) == 0 for d in dets for i in range(len(resC01)))
    assert len(resC02) == 2

    eqC12 = (-(C1*Z11*Z12 + C2*Z21*Z22)*(C2*Z20**2 + Z00*(-C1*Z10 + Z00*(C1 + C_J)) + Z10*(-C1*Z00 + C1*Z10)) + (C2*Z20*Z22 + Z12*(-C1*Z00 + C1*Z10))*(-C1*Z00*Z11 + C1*Z10*Z11 + C2*Z20*Z21))/((C1*Z11**2 + C2*Z21**2)*(C1*Z12**2 + C2*Z22**2)*(C2*Z20**2 + Z00*(-C1*Z10 + Z00*(C1 + C_J)) + Z10*(-C1*Z00 + C1*Z10)) - (C1*Z11**2 + C2*Z21**2)*(C2*Z20*Z22 + Z12*(-C1*Z00 + C1*Z10))*(-C1*Z00*Z12 + C1*Z10*Z12 + C2*Z20*Z22) - (C1*Z12**2 + C2*Z22**2)*(C2*Z20*Z21 + Z11*(-C1*Z00 + C1*Z10))*(-C1*Z00*Z11 + C1*Z10*Z11 + C2*Z20*Z21) - (C1*Z11*Z12 + C2*Z21*Z22)**2*(C2*Z20**2 + Z00*(-C1*Z10 + Z00*(C1 + C_J)) + Z10*(-C1*Z00 + C1*Z10)) + (C1*Z11*Z12 + C2*Z21*Z22)*(C2*Z20*Z21 + Z11*(-C1*Z00 + C1*Z10))*(-C1*Z00*Z12 + C1*Z10*Z12 + C2*Z20*Z22) + (C1*Z11*Z12 + C2*Z21*Z22)*(C2*Z20*Z22 + Z12*(-C1*Z00 + C1*Z10))*(-C1*Z00*Z11 + C1*Z10*Z11 + C2*Z20*Z21))
    varsC12 = [Z00, Z10, Z11, Z12, Z20, Z21, Z22]
    resC12 = eqs.sol_indep_of_vars(eqC12, varsC12, nonzero=dets)
    assert sum(abs(sym.simplify(eqC12.subs(resC12[i])) == 0) for i in range(len(resC12)))
    assert len(resC12) == 4
    for r in resC12:
        assert not any(d.subs(r) == 0 for d in dets)
        assert sym.simplify(eqC12.subs(r)) == 0

    eqL12 = Z12*(-Z21/L3 + Z11*(L1 + L3)/(L1*L3)) + Z22*(-Z11/L3 + Z21*(L2 + L3)/(L2*L3))
    varsL12 = [Z11, Z12, Z21, Z22]
    resL12 = eqs.sol_indep_of_vars(eqL12, varsL12, nonzero=dets)
    assert len(resL12) == 0

    eq = Z11*(Z20/L1 + Z00*(L1 - L2)/(2*L1*L2) + Z10*(L1 + L2)/(L1*L2)) + Z21*(-Z00/(2*L1) + Z10/L1 + Z20*(L1 + L3)/(L1*L3))
    sols = eqs.sol_indep_of_vars(eq, solve_vars)
    x_sols = [{Z21: 0, Z11: 0}, {Z00: 0, Z10: 0, Z20: 0}, {Z00: 2*Z10, Z11: 0, Z20: 0}, {Z00: Z20, Z10: -Z20*sym.Rational(1,2), Z21: 0}, {Z00: -2*Z10, Z11: -Z21, Z20: 0}]
    for sol in sols:
        assert sym.simplify(eq.subs(sol)) == 0
    for x_sol in x_sols:
        assert x_sol in sols
    for sol in sols:
        assert sol in x_sols

    x, y = sym.symbols("x,y")
    a, b = sym.symbols("a,b")

    eq_bad = (x + y/2)*a*b + b
    eqs.SOLVE_CACHE.clear()
    sol = eqs.sol_indep_of_vars(eq_bad, [x, y])
    assert sol == []

    Z10, Z20 = sym.symbols("Z10, Z20")
    L_3 = sym.symbols("L_3")
    expr = Z10*Z20/L_3
    sol = eqs.sol_indep_of_vars(expr, [Z10, Z20])
    assert {Z10: 0} in sol and {Z20: 0} in sol

    j_var = sym.symbols("J_1, J_2, J_3, J_4")
    J_1, J_2, J_3, J_4 = j_var
    Z_var = sym.symbols("Z21, Z22, Z02, Z11, Z01, Z12")
    Z21, Z22, Z02, Z11, Z01, Z12 = Z_var
    Z_var_set = {Z01, Z02, Z11, Z12, Z21, Z22}
    expr = J_3*Z21*Z22 + Z02*(-J_4*Z11 + Z01*(J_1 + J_4)) + Z12*(-J_4*Z01 + Z11*(J_2 + J_4))
    sol = eqs.sol_indep_of_vars(expr, Z_var_set)
    good_subs = {Z02: 0, Z12: 0, Z21: 0}
    assert any(s == good_subs for s in sol)

    eq = x - y
    sol = eqs.sol_indep_of_vars(eq, [x, y])
    assert len(sol) == 1
    sol = sol[0]
    if x in sol:
        assert sol[x] == y
    else:
        assert sol[y] == x

    eq = (x + y/2)*a*b
    sol = eqs.sol_indep_of_vars(eq, [x, y])
    assert len(sol) == 1
    sol = sol[0]
    if x in sol:
        assert sol[x] == -y/2
    else:
        assert sol[y] == -2*x

    eq = (x + y/2)*a + b
    sol = eqs.sol_indep_of_vars(eq, [x, y, b])
    assert len(sol) == 1
    sol = sol[0]
    if x in sol:
        assert sol[x] == -y/2
    else:
        assert sol[y] == -2*x
    assert sol[b] == 0

    eq = x*a*b
    sol = eqs.sol_indep_of_vars(eq, [x, y, b])
    assert len(sol) == 2
    if x in sol[0]:
        assert sol[0][x] == 0 and sol[1][b] == 0
    else:
        assert sol[1][x] == 0 and sol[0][b] == 0

    eq = (x + 2)*y
    sols = eqs.sol_indep_of_vars(eq, [x, y])
    assert any(s == {y: 0} for s in sols)
    assert any(s.get(x) == -2 for s in sols)

    eq2 = a + 1
    sols2 = eqs.sol_indep_of_vars(eq2, [x, y])
    assert sols2 == []


def test_unique_products():
    Z00, Z11, Z12, Z21, Z20, Z02, Z22, Z10 = sym.symbols("Z00, Z11, Z12, Z21, Z20, Z02, Z22, Z10", real=True)
    C1, C2 = sym.symbols("C1, C2", real=True, positive=True)

    expr = (-4*C1*C2*Z00*Z11*Z12*Z22 - 2*C1*C2*Z00*Z11*Z22**2 +
            4*C1*C2*Z00*Z12**2*Z21 + 2*C1*C2*Z00*Z12*Z21*Z22 -
            4*C1*C2*Z10*Z11*Z22**2 + 4*C1*C2*Z10*Z12*Z21*Z22 +
            4*C1*C2*Z11*Z12*Z20*Z22 - 4*C1*C2*Z12**2*Z20*Z21)

    prods = eqs._unique_products(expr, [Z00, Z11, Z12, Z21, Z20, Z02, Z22, Z10])
    assert len(prods) == 1
    assert sym.simplify(prods[C1*C2]) == sym.simplify(expr/(C1*C2))

    x, y = sym.symbols("x,y", real=True)
    a, b = sym.symbols("a,b", real=True)

    eq = (x + y/2)*a*b + b
    prods = eqs._unique_products(eq, [x, y])
    assert prods == {a*b: x + y/2, b: 1}

    eq = x*y
    prods = eqs._unique_products(eq, [a, b])
    assert eq in prods
    assert prods[eq] == 1

    eq = x*y
    prods = eqs._unique_products(eq, [x, y])
    assert 1 in prods
    assert prods[1] == eq

    eq = x + y
    prods = eqs._unique_products(eq, [x, y])
    assert prods[1] == eq

    eq = (x + y)*(a + b)**2 + a - y*b
    prods = eqs._unique_products(eq, [x, y])
    for pr in [a**2, a*b, b**2, a, b]:
        assert pr in prods

    eq = (x + y)*(a + 1/b)**2 + 1/a - y*b
    prods = eqs._unique_products(eq, [x, y])
    for pr in [a**2, a/b, 1/b**2, 1/a, b]:
        assert pr in prods

    eq = 0.0
    prods = eqs._unique_products(eq)
    assert prods == {}


def test_extract_denom():
    x, y = sym.symbols("x,y", real=True)
    a, b = sym.symbols("a,b", real=True)

    M = sym.Matrix([[1/x, 0, 2],
                    [1/x + 1/y, sym.Rational(1, 100), 0],
                    [(a + b)/(x*y + a + b), 0, 3/y]])
    denom = eqs.extract_denom(M)
    assert len(denom) == 4
    for entry in [x, y, x*y + a + b, x*y]:
        assert entry in denom


if __name__ == "__main__":
    # test = TestMaximallyCompatibleSol()
    # test.test_01_fractions_raise_error()
    # test.test_02_basic_maximization_all_compatible()
    # test.test_03_pairwise_conflict_pruning()
    # test.test_04_nonzero_constraints()
    # test.test_05_nonzero_post_check_filtering()
    # test.test_06_individual_unsolvable_skipped()
    # test_sol_indep_of_vars()
    # test_eq_indep_of_vars()
    test_eq_indep_of_vars()