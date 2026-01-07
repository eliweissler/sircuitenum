import time

import numpy as np
import sympy as sym

from sircuitenum import equationset as eqs


def test_equationset_from_any_and_equality():
    """Test EquationSet.from_any() canonicalization and equality."""
    a, b, c, d, e = sym.symbols("a b c d e")
    
    # Order-independence: same equations in different order
    s1 = eqs.EquationSet.from_any([sym.Eq(a, b), sym.Eq(c, d)])
    s2 = eqs.EquationSet.from_any([sym.Eq(c, d), sym.Eq(a, b)])
    s3 = eqs.EquationSet.from_any([sym.Eq(a, b), sym.Eq(c, e)])
    assert s1 == s2
    assert s1 != s3

    # From dict
    s1_dict = eqs.EquationSet.from_any({a: b, c: d})
    s2_dict = eqs.EquationSet.from_any({c: d, a: b})
    assert s1_dict == s2_dict
    assert s1 == s1_dict  # dict and list representations are equal

    # From frozenset of frozensets
    Z11 = sym.symbols("Z11")
    test = frozenset({frozenset({0, Z11 + 1})})
    s_frozen = eqs.EquationSet.from_any(test)
    assert len(s_frozen.as_eq_list()) == 1

    # From expressions (converted to equations)
    expr1 = a - b
    expr2 = c - d
    s_expr = eqs.EquationSet.from_any([expr1, expr2])
    assert len(s_expr.as_eq_list()) == 2
    assert all(eq.rhs == 0 for eq in s_expr.as_eq_list())


def test_equationset_as_frozenset():
    """Test EquationSet.as_frozenset() method."""
    a, b, c, d = sym.symbols("a b c d")
    s1 = eqs.EquationSet.from_any([sym.Eq(a, b), sym.Eq(c, d)])
    s2 = eqs.EquationSet.from_any([sym.Eq(c, d), sym.Eq(a, b)])
    
    # Frozensets should be equal for equal EquationSets
    assert s1.as_frozenset() == s2.as_frozenset()
    
    # Can be used as dict key
    cache = {s1.as_frozenset(): "value"}
    assert cache[s2.as_frozenset()] == "value"


def test_equationset_as_eq_list():
    """Test EquationSet.as_eq_list() method."""
    a, b, c, d = sym.symbols("a b c d")
    
    # Basic conversion
    s1 = eqs.EquationSet.from_any([sym.Eq(a, b), sym.Eq(c, d)])
    eq_list = s1.as_eq_list()
    assert len(eq_list) == 2
    assert all(isinstance(eq, sym.Eq) for eq in eq_list)
    
    # Round-trip preservation
    s2 = eqs.EquationSet.from_any(eq_list)
    assert s1 == s2


def test_equationset_as_dict():
    """Test EquationSet.as_dict() method."""
    a, b, c = sym.symbols("a b c")

    # Simple conversion
    s1 = eqs.EquationSet.from_any([sym.Eq(a, b), sym.Eq(b, c)])
    dct = s1.as_dict()
    assert isinstance(dct, dict)
    assert dct[a] == b and dct[b] == c

    # Reversed numeric side -> should flip to var: number
    s2 = eqs.EquationSet.from_any([sym.Eq(2, a)])
    dct = s2.as_dict()
    assert dct == {a: 2}

    # Conflicting assignments -> None
    s3 = eqs.EquationSet.from_any([sym.Eq(a, b), sym.Eq(a, c)])
    assert s3.as_dict() is None


def test_equationset_constant_conflict_true():
    """EquationSet flags constant conflicts when same var gets different numbers."""
    x = sym.symbols("x")
    s = eqs.EquationSet.from_any([sym.Eq(x, 1), sym.Eq(x, 2)])
    assert s.as_dict() is None
    assert getattr(s, "_constant_conflict") is True


def test_equationset_constant_conflict_false():
    """No constant conflict when assignments are consistent."""
    x, y = sym.symbols("x y")
    s = eqs.EquationSet.from_any([sym.Eq(x, 1), sym.Eq(y, 2)])
    assert s.as_dict() == {x: 1, y: 2}
    assert getattr(s, "_constant_conflict") is False


def test_equationset_nonconstant_conflict_no_flag():
    """Conflicting symbolic assignments return None but do not set constant_conflict."""
    x, y, z = sym.symbols("x y z")
    s = eqs.EquationSet.from_any([sym.Eq(x, y), sym.Eq(x, z)])
    assert s.as_dict() is None
    assert getattr(s, "_constant_conflict") is False


def test_equationset_constant_conflict_from_dicts_and_order():
    """Detect conflicts from list of dicts and ensure order independence."""
    x = sym.symbols("x")
    s1 = eqs.EquationSet.from_any([{x: 1}, {x: 2}])
    s2 = eqs.EquationSet.from_any([{x: 2}, {x: 1}])
    assert s1.as_dict() is None and s2.as_dict() is None
    assert getattr(s1, "_constant_conflict") is True
    assert getattr(s2, "_constant_conflict") is True


def test_equationset_union():
    a, b, c, d = sym.symbols("a b c d")
    s1 = eqs.EquationSet.from_any([sym.Eq(a, b)])
    s2 = eqs.EquationSet.from_any([sym.Eq(c, d)])
    combined = s1 | s2
    assert isinstance(combined, eqs.EquationSet)
    
    # Union should contain both equations
    expected = eqs.EquationSet.from_any([sym.Eq(a, b), sym.Eq(c, d)])
    assert combined == expected
    
    # Verify via frozenset too
    assert combined.as_frozenset() == expected.as_frozenset()

    s1 = eqs.EquationSet.from_any((sym.Eq(a, b), sym.Eq(c, 0)))
    s2 = eqs.EquationSet.from_any((sym.Eq(a, 0),))
    combined = s1 | s2
    # Union concatenates equations without solving, so we get {a=b, c=0, a=0}
    expected = eqs.EquationSet.from_any((sym.Eq(a, b), sym.Eq(c, 0), sym.Eq(a, 0)))
    assert combined == expected
    # The dict representation is None because of conflicting symbolic assignments
    assert combined._constant_conflict is False

    s1 = eqs.EquationSet.from_any((sym.Eq(a, 1), sym.Eq(c, 0)))
    s2 = eqs.EquationSet.from_any((sym.Eq(a, 0),))
    combined = s1 | s2
    # Union concatenates equations without solving, so we get {a=b, c=0, a=0}
    expected = eqs.EquationSet.from_any((sym.Eq(a, 1), sym.Eq(c, 0), sym.Eq(a, 0)))
    assert combined == expected
    # The dict representation should still be None due to conflicting assignments
    assert combined._constant_conflict is True


def test_sols_set_to_dict():
    x, y, z = sym.symbols("x y z")
    sols_set = [(1, sym.Complexes, 3), (x, y, z)]
    res = eqs._sols_set_to_dict(sols_set, [x, y, z])
    assert isinstance(res, list)
    assert res[0] == {x: 1, z: 3}
    assert res[1] == {}


def test_eq_as_numer_denom():
    x, y = sym.symbols("x y")
    numer, denom = eqs._eq_as_numer_denom(sym.Eq(1/(x + 1), 0))
    assert numer == 1
    assert denom == x + 1

    numer, denom = eqs._eq_as_numer_denom(sym.Eq(x/y, 2))
    assert sym.simplify(numer - (x - 2*y)) == 0
    assert denom == y

    numer, denom = eqs._eq_as_numer_denom(sym.Eq((x + 1)/(x - 1), (y + 1)/(y - 1)))
    assert denom == (x - 1)*(y - 1)


def test_maximally_compatible_set():

    Z00, Z11, Z12, Z21, Z20, Z02, Z22, Z10 = sym.symbols("Z00, Z11, Z12, Z21, Z20, Z02, Z22, Z10", real=True)
    C1, C2, C_J = sym.symbols("C1, C2, C_{J}", real=True, positive=True)
    var_list = [Z00, Z11, Z12, Z21, Z20, Z02, Z22, Z10]
    coupling_c = [[{Z00: Z10 - Z12*Z20/Z22}], [{Z00: Z10 - Z11*Z20/Z21}], [{Z00: Z10 - Z11*Z20/Z21, Z12: 0}, {Z00: Z10, Z12: 0}, {Z00: Z10, Z11: 0, Z22: 0}]]
    coupling_l = [[{Z00: Z10, Z11: Z21, Z20: 0}, {Z10: 0, Z11: 0, Z20: 0}, {Z00: Z20, Z10: Z20, Z21: 0}], [{Z10: Z20, Z12: 0}, {Z00: Z10, Z12: Z22, Z20: 0}, {Z10: 0, Z12: 0, Z20: 0}, {Z00: Z20, Z10: Z20, Z22: 0}]]
    dets = [Z00*(Z11*Z22 - Z12*Z21), C1*C2*C_J*Z00**2*(Z11**2*Z22**2 - 2*Z11*Z12*Z21*Z22 + Z12**2*Z21**2)]
    terms = coupling_c + coupling_l
    keys, subs = eqs.maximally_compatible_set(terms, solve_vars=var_list, nonzero=dets)
    assert keys
    assert subs

    x, y = sym.symbols("x y", real=True)

    terms = [[{x: 0}, {x: 1}], [{y: 0}, {y: 1}]]
    keys, subs = eqs.maximally_compatible_set(terms, [x, y])
    assert (0, 1) in keys
    assert any(isinstance(s, dict) and x in s and y in s for s in subs)

    terms = [[{x: 0}], [{x: 1}], [{y: 0}]]
    keys, subs = eqs.maximally_compatible_set(terms, [x, y])
    assert all(not (0 in k and 1 in k) for k in keys)
    assert any(len(k) == 2 for k in keys)

    tiebreaker = lambda ks: sum(ks) if ks else 20
    keys, subs = eqs.maximally_compatible_set(terms, [x, y], tiebreaker_fn=tiebreaker)
    assert any(k == (0, 2) for k in keys)

    # From choose Z example
    resL01 =  [{Z00: Z20, Z10: Z20, Z21: 0}, {Z00: Z10, Z11: Z21, Z20: 0}, {Z10: 0, Z11: 0, Z20: 0}]
    resC02 =  [{Z00: Z10 - Z11*Z20/Z21}, {Z20: 0, Z21: 0}]
    resC01 =  [{Z00: Z10 - Z12*Z20/Z22}, {Z20: 0, Z22: 0}]
    dets = [Z00*(Z11*Z22 - Z12*Z21), C1*C2*C_J*Z00**2*(Z11**2*Z22**2 - 2*Z11*Z12*Z21*Z22 + Z12**2*Z21**2)]
    keys, subs = eqs.maximally_compatible_set([resL01, resC02, resC01], solve_vars=var_list, nonzero=dets)
    assert keys == [(0, 1, 2), (0, 1, 2)]
    for s in [{Z00: Z10, Z11: Z21, Z20: 0}, {Z00: Z10, Z11: Z21, Z20: 0, Z22:0}]:
        assert s in subs



def test_cached_solve():

    Z10, Z01, Z11, Z12, Z22, Z21, Z00, Z20, Z02 = sym.symbols('Z10 Z01 Z11 Z12 Z22 Z21 Z00 Z20 Z02', real=True)
    solve_vars = (Z10, Z01, Z11, Z12, Z22, Z21, Z00, Z20, Z02)
    eq_set = eqs.EquationSet.from_any([sym.Eq(Z11, Z12*Z21/Z22)])
    sols = eqs.cached_solve(eq_set, solve_vars)

    eq_set = [sym.Eq(Z00*Z11 - 2*Z10*Z11 + Z10*Z21 + Z11*Z20 - 2*Z20*Z21, 0),
              sym.Eq(Z00*Z22*(Z12 - Z22) + 2*Z12**2*Z20 - 2*Z12*Z20*Z22 + 2*Z20*Z22**2, 0),
              sym.Eq(Z10*Z22*(Z12 - Z22) + Z12**2*Z20 - Z12*Z20*Z22 + 2*Z20*Z22**2, 0),
              sym.Eq(Z11*(2*Z12 - Z22) - Z12*Z21 + 2*Z21*Z22, 0)]
    

    # Test all combinations of 4 variables to make sure caching works correctly for subsets of variables
    v_in_eq = [v for v in solve_vars if any(v in eq.free_symbols for eq in eq_set)]
    import itertools
    new_sols = []
    new_sols2 = []
    for combos in itertools.combinations(v_in_eq, 4):
        sol_subset1 = eqs.cached_solve(eq_set, combos)
        sol_subset2 = eqs.unique_solutions(eqs._sols_set_to_dict(sym.simplify(sym.nonlinsolve(eq_set, combos)), combos))
        assert len(sol_subset1) == len(sol_subset2)
        for s1 in sol_subset1:
            for var in s1:
                assert any(sym.simplify(s1[var] - s2.get(var, 0)) == 0 for s2 in sol_subset2)

    # Test from variable transformation equations
    eq_set = [sym.Eq(Z00, Z10 - Z11*Z20/Z21), sym.Eq(Z00, Z10), sym.Eq(Z11, Z21), sym.Eq(Z20, 0)]
    eq_set2 = [eq.subs({Z20:0}) for eq in eq_set[:-1]]  # remove last eq since we substitute it in
    sol_sympy = eqs._sols_set_to_dict(sym.nonlinsolve(eq_set, solve_vars), solve_vars)
    sol_cached = eqs.cached_solve(eq_set, solve_vars)
    assert len(sol_cached) == len(sol_sympy)
    assert len(sol_cached) == 1
    assert eqs.EquationSet.from_any(sol_cached[0]) == eqs.EquationSet.from_any(sol_sympy[0])


    # basic functionality test
    Z10, Z01, Z11, Z12, Z22, Z21 = sym.symbols('Z10 Z01 Z11 Z12 Z22 Z21')
    eq_set = eqs.EquationSet.from_any((sym.Eq(Z12, 0), sym.Eq(Z22, 0)))
    sols = eqs.cached_solve(eq_set, solve_vars)
    assert sols == [{Z12: 0, Z22: 0}]

    eq_set = [sym.Eq(Z00*Z12 - Z10*Z12 - Z20*Z22, 0), 
                sym.Eq(Z11*Z12 + Z21*Z22, 0), 
                sym.Eq(Z00*Z11 - 2*Z10*Z11 + Z10*Z21 + Z11*Z20 - 2*Z20*Z21, 0)]
    eq_set = [eq.lhs for eq in eq_set]
    sols = eqs.cached_solve(eq_set)
    assert len(sols) == 12



    # speed and accuracy test
    eps = 1e-06
    for _ in range(10):
        x, y, z = sym.symbols("x,y,z")
        eq1 = x + 2*y + 3*z - 6*np.random.random()
        eq2 = 2*x + 3*y + z - 5*np.random.random()
        eq3 = x - y + z - 2*np.random.random()
        eqs_list = [eq1, eq2, eq3]
        vars_list = [x, y, z]
        sol_good = sym.solve(eqs_list, vars_list, dict=True, simplify=True)[0]
        t0 = time.time()
        sol1 = eqs.cached_solve(eqs_list, vars_list)[0]
        t1 = time.time()
        sol2 = eqs.cached_solve(eqs_list, vars_list)[0]
        t2 = time.time()
        assert t1 - t0 > t2 - t1
        for v in vars_list:
            assert abs(sol1[v] - sol_good[v]) < eps
            assert abs(sol2[v] - sol_good[v]) < eps
    
    # identify different branches
    eq = sym.Eq(0, - Z10*Z01 - Z10*Z11 - Z01)
    sols = eqs.cached_solve([eq], [Z10, Z01, Z11])
    good_sols = [{Z01: -Z10*Z11/(Z10 + 1)}, {Z10: -1, Z11: 0}]
    assert len(sols) == len(good_sols)
    for gs in good_sols:
        assert gs in sols
    

def test_expand_singular_branches():


    C1, C2, C_J = sym.symbols("C1, C2, C_J", real=True, positive=True)
    L1, L2, L3 = sym.symbols("L1, L2, L3", real=True, positive=True)
    Z00, Z11, Z12, Z21, Z20, Z02, Z22, Z10 = sym.symbols("Z00, Z11, Z12, Z21, Z20, Z02, Z22, Z10", real=True)
    solve_vars = [Z00, Z11, Z12, Z21, Z20, Z02, Z22, Z10]
    dets = [Z00*(Z11*Z22 - Z12*Z21), C1*C2*C_J*Z00**2*(Z11**2*Z22**2 - 2*Z11*Z12*Z21*Z22 + Z12**2*Z21**2)]


    eq = eqs.EquationSet.from_any([sym.Eq(Z11, Z12*Z21/Z22)])
    sols = eqs._expand_singular_branches(eq, [{Z11: Z12*Z21/Z22}], solve_vars)
    assert len(sols) == 1

    eq = eqs.EquationSet.from_any((sym.Eq(-Z00**2*Z21*Z22 + 2*Z00*Z10*Z21*Z22 - Z00*Z11*Z20*Z22 - Z00*Z12*Z20*Z21 - Z10**2*Z21*Z22 + Z10*Z11*Z20*Z22 + Z10*Z12*Z20*Z21 - Z11*Z12*Z20**2, 0),))
    sols = [{Z11: Z12*Z21/Z22}, {Z00: Z10 - Z11*Z20/Z21}]
    sols = eqs._expand_singular_branches(eq, sols, solve_vars)
    assert len(sols) == 11


def test_sol_indep_of_vars():
    C1, C2, C_J = sym.symbols("C1, C2, C_J", real=True, positive=True)
    L1, L2, L3 = sym.symbols("L1, L2, L3", real=True, positive=True)
    Z00, Z11, Z12, Z21, Z20, Z02, Z22, Z10 = sym.symbols("Z00, Z11, Z12, Z21, Z20, Z02, Z22, Z10", real=True)
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

    x, y = sym.symbols("x,y", real=True)
    a, b = sym.symbols("a,b", real=True)

    eq_bad = (x + y/2)*a*b + b
    eqs.SOLVE_CACHE.clear()
    sol = eqs.sol_indep_of_vars(eq_bad, [x, y])
    assert sol == []

    Z10, Z20 = sym.symbols("Z10, Z20", real=True)
    L_3 = sym.symbols("L_3", real=True, positive=True)
    expr = Z10*Z20/L_3
    sol = eqs.sol_indep_of_vars(expr, [Z10, Z20])
    assert {Z10: 0} in sol and {Z20: 0} in sol

    j_var = sym.symbols("J_1, J_2, J_3, J_4", real=True, positive=True)
    J_1, J_2, J_3, J_4 = j_var
    Z_var = sym.symbols("Z21, Z22, Z02, Z11, Z01, Z12", real=True)
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
    assert prods[C1*C2] == sym.simplify(expr/(C1*C2))

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


def test_unique_solutions():
    """Test _unique_solutions filters out duplicate solutions."""
    x, y, z = sym.symbols("x y z")
    
    # Test with duplicate solutions
    sols = [
        {x: 1, y: 2},
        {y: 2, x: 1},  # Duplicate (different order)
        {x: 1, y: 3},  # Different solution
        {x: 1, y: 2},  # Another duplicate
    ]
    
    unique = eqs.unique_solutions(sols)
    assert len(unique) == 2
    assert {x: 1, y: 2} in unique
    assert {x: 1, y: 3} in unique
    
    # Test with all unique solutions
    sols_unique = [
        {x: 1},
        {x: 2},
        {y: 3},
    ]
    unique2 = eqs.unique_solutions(sols_unique)
    assert len(unique2) == 3
    
    # Test with empty list
    assert eqs.unique_solutions([]) == []
    
    # Test that Eq(a,b) and Eq(b,a) are treated as same
    sols_sym = [
        {x: y},
        {y: x},  # Equivalent to x: y
    ]
    unique3 = eqs.unique_solutions(sols_sym)
    assert len(unique3) == 1

    # Subset solutions
    x, y, z = sym.symbols("x y z")
    sols = [
        {x: 1, y: 2, z: 3},
        {x: 1, y: 2},
        {x: 1},
    ]
    minimized = eqs.unique_solutions(sols)
    assert minimized == [{x: 1}]


    x, y = sym.symbols("x y")
    sols = [
        {x: 1},
        {y: 2},
        {x: 1, y: 2},  # superset of both
    ]
    minimized = eqs.unique_solutions(sols)
    # Order of kept items should be sorted by size then first-seen
    assert minimized == [{x: 1}, {y: 2}]

    x, y = sym.symbols("x y")
    sols = [
        {x: 1, y: 2},
        {y: 2, x: 1},  # duplicate
        {x: 1},        # subset of the first
    ]
    minimized = eqs.unique_solutions(sols)
    # {x:1} is minimal; duplicates of {x:1,y:2} are removed and supersets dropped
    assert minimized == [{x: 1}]


    x, y = sym.symbols("x y")
    sols1 = [{x: 1}, {x: 1, y: 2}]
    sols2 = [{x: 1, y: 2}, {x: 1}]
    m1 = eqs.unique_solutions(sols1)
    m2 = eqs.unique_solutions(sols2)
    assert m1 == m2 == [{x: 1}]


def test_fully_compatible_set():
    """Test fully_compatible_set function."""
    # ensure order does not matter
    Z00, Z11, Z12, Z21, Z20, Z02, Z22, Z10 = sym.symbols("Z00, Z11, Z12, Z21, Z20, Z02, Z22, Z10", real=True)
    solve_vars = (Z00, Z11, Z12, Z21, Z20, Z02, Z22, Z10)


    # length 1 solutions
    sols = [{Z11: Z12*Z21/Z22}, {Z00: Z10 - Z12*Z20/Z22}, {Z12: 0, Z22: 0}, {Z20: 0, Z22: 0}, {Z21: 0, Z22: 0}]
    res1 = eqs.fully_compatible_set([sols], solve_vars, depth_first=False)
    assert len(res1) == len(sols)
    assert all(s in res1 for s in sols)
    assert all(s in sols for s in res1)



    ord1 = [[{Z12: 0}, {Z00: Z10}], [{Z20: 0}, {Z22: 0}], [{Z12: Z22}, {Z10: Z20}]]
    ord2 =  [[{Z12: Z22}, {Z10: Z20}], [{Z20: 0}, {Z22: 0}], [{Z12: 0}, {Z00: Z10}]]
    res1 = eqs.fully_compatible_set(ord1, solve_vars, depth_first=False)
    res2 = eqs.fully_compatible_set(ord2, solve_vars, depth_first=False)
    res1_set = frozenset(eqs.EquationSet.from_any(r).as_frozenset() for r in res1)
    res2_set = frozenset(eqs.EquationSet.from_any(r).as_frozenset() for r in res2)
    assert res1_set == res2_set
    

    C1, C2 = sym.symbols("C1, C2", real=True, positive=True)
    expr = (-4*Z00*Z11*Z12*Z22 - 2*Z00*Z11*Z22**2 +
            4*Z00*Z12**2*Z21 + 2*Z00*Z12*Z21*Z22 - 
            4*Z10*Z11*Z22**2 + 4*Z10*Z12*Z21*Z22 + 
            4*Z11*Z12*Z20*Z22 - 4*Z12**2*Z20*Z21)
    assumptions = [({expr: 0},)]
    res_all = eqs.fully_compatible_set(assumptions, solve_vars, depth_first=False)
    assert len(res_all) == 7
    assert {Z00: sym.simplify(2*(-Z10*Z22 + Z12*Z20)/(2*Z12 + Z22))} in res_all
    assert {Z11: Z12*Z21/Z22} in res_all
    # res_targ = [{Z00: 2*(-Z10*Z22 + Z12*Z20)/(2*Z12 + Z22)}, {Z11: Z12*Z21/Z22}, {Z11: -Z21/2, Z12: -Z22/2}, {Z10: -Z20/2, Z12: -Z22/2}, {Z12: 0, Z22: 0}, {Z21: 0, Z22: 0}, {Z00: Z20, Z22: 0}]
    for sol in res_all:
        assert sym.simplify(expr.subs(sol)) == 0

    res_one = eqs.fully_compatible_set(assumptions, solve_vars, depth_first=True)
    assert len(res_one) >= 1
    assert {Z00: sym.simplify(2*(-Z10*Z22 + Z12*Z20)/(2*Z12 + Z22))} in res_one or {Z11: Z12*Z21/Z22} in res_one

    # Produces infite denominator with the first expression applied to last
    assumptions = [({Z11: -1, Z12:-Z22/2},), ({Z00: sym.simplify(2*(-Z10*Z22 + Z12*Z20)/(2*Z12 + Z22)),},)]
    res_one = eqs.fully_compatible_set(assumptions, solve_vars, depth_first=False)
    assert len(res_one) == 0

    assumptions = [({Z11: -1, Z12:-Z22/2}, {Z12: Z22}), ({Z00: sym.simplify(2*(-Z10*Z22 + Z12*Z20)/(2*Z12 + Z22)),},)]
    res_all = eqs.fully_compatible_set(assumptions, solve_vars, depth_first=False)
    assert len(res_all) == 1

    assumptions = [({Z11: -1, Z12:-Z22/2}, {Z12: Z22}), ({Z11: Z12*Z21/Z22},)]
    res_all = eqs.fully_compatible_set(assumptions, solve_vars, depth_first=False)
    t1 = eqs.cached_solve(eqs.EquationSet.from_any([assumptions[0][0], assumptions[1][0]]), solve_vars)
    t2 = eqs.cached_solve(eqs.EquationSet.from_any([assumptions[0][1], assumptions[1][0]]), solve_vars)
    assert len(res_all) == 2
    for tk in t1 + t2:
        assert tk in res_all
    
    assumptions = [({Z11: -1, Z12:-Z22/2},), ({Z00: sym.simplify(2*(-Z10*Z22 + Z12*Z20)/(2*Z12 + Z22))},
                                                         {Z11: Z12*Z21/Z22})]
    res_all = eqs.fully_compatible_set(assumptions, solve_vars, depth_first=False)
    assert len(res_all) == 1

    assumptions = [[{Z20: 0}, {Z21: 0}], [{Z11: 0}, {Z00: -2*Z10}], [{Z11: -Z21}, {Z00: 2*Z10 + 2*Z20}]]
    res_all = eqs.fully_compatible_set(assumptions, solve_vars, depth_first=False)
    assert len(res_all) == 8


if __name__ == "__main__":
    # test_fully_compatible_set()
    # test_maximally_compatible_set()
    # test_sol_indep_of_vars()
    # test_equationset_union()
    test_cached_solve()
    # test_equationset_nonconstant_conflict_no_flag()
    # test_expand_singular_branches()
    test_sol_indep_of_vars()
    # test_unique_solutions()
    test_fully_compatible_set()