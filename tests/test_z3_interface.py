import pytest
import sympy as sym
from sircuitenum.z3_interface import find_rational_vars_integer_results, _calc_min_cost, _heuristic_upper_bound

# Define symbols for re-use across tests
@pytest.fixture
def vars():
    return sym.symbols('Z1 Z2 Z3')
def vars2():
    return sym.symbols('Z1 Z2 Z3')

def test_basic_rational_constraints(vars):
    """
    Test that variables can take on rational values (e.g. 1/2, 3/2)
    to satisfy integer constraints on expressions (e.g. 2*Z1).
    """
    Z1, Z2, _ = vars
    
    # If Z1/2 is an integer, Z1 must be something like 2, 4, 6... 
    # BUT since variables are rational, Z1 could be 2.0, 4.0 etc.
    # Let's test a constraint that forces a fraction:
    # 2 * Z1 = 1 (Integer) -> Z1 could be 1/2
    
    constraints = [
        2 * Z1,  # Must be integer
        3 * Z2   # Must be integer
    ]
    
    # We force Z1 to be 0.5 effectively by checking if 2*Z1 results in 1
    # But let's just check the solver finds valid integer outputs
    results = find_rational_vars_integer_results(
        integer_constraints=constraints,
        nonzero_constraints=[],
        zero_constraints=[],
        variables=[Z1, Z2],
        max_result_range=2
    )
    
    assert len(results) > 0
    
    for sol in results:
        # Check consistency: The result values must actually be integers
        res_vals = sol["results"]
        assert all(isinstance(x, int) for x in res_vals)
        
        # Check logic: Substitute variables back into expression
        var_map = sol["variables"]
        expr_val = (2 * Z1).subs(var_map)
        assert expr_val == res_vals[0] # Should match the solver's output

def test_half_type_handling(vars):
    """
    Regression test: Ensure 'Z1 / 2' (which creates sympy.Half) 
    does not crash the recursive converter.
    """
    Z1, _, _ = vars
    
    constraints = [
        Z1 / 2,        # This introduces the Rational(1, 2) type
        Z1 * sym.Rational(1, 3) # Test explicit Rational object too
    ]
    
    try:
        results = find_rational_vars_integer_results(
            integer_constraints=constraints,
            nonzero_constraints=[],
            zero_constraints=[],
            variables=[Z1],
            max_result_range=2
        )
        assert len(results) > 0
    except ValueError as e:
        pytest.fail(f"Solver crashed on Rational types: {e}")

def test_nonzero_constraints(vars):
    """
    Ensure nonzero constraints successfully prune solutions like 0.
    """
    Z1, Z2, _ = vars
    
    # Constraint: Z1/1 is integer (Z1 is integer)
    constraints = [Z1, Z2]
    
    # Nonzero: Product cannot be 0
    nonzero = [Z1 * Z2]
    
    results = find_rational_vars_integer_results(
        integer_constraints=constraints,
        nonzero_constraints=nonzero,
        zero_constraints=[],
        variables=[Z1, Z2],
        max_result_range=1
    )
    
    # Should NOT find Z1=0, Z2=0
    for sol in results:
        z1_val = sol["variables"][Z1]
        z2_val = sol["variables"][Z2]
        assert z1_val * z2_val != 0

def test_return_format_types(vars):
    """
    Verify that the 'variables' dict contains SymPy objects,
    not strings or Z3 types.
    """
    Z1, _, _ = vars
    constraints = [Z1] # Simple identity constraint
    
    results = find_rational_vars_integer_results(
        integer_constraints=constraints,
        nonzero_constraints=[],
        zero_constraints=[],
        variables=[Z1],
        max_result_range=1
    )
    
    assert len(results) > 0
    first_sol = results[0]
    
    var_map = first_sol["variables"]
    
    # Check Key Type
    assert Z1 in var_map
    assert isinstance(list(var_map.keys())[0], sym.Symbol)
    
    # Check Value Type
    val = var_map[Z1]
    # Should be sym.Integer or sym.Rational or sym.Float, but definitely a SymPy Basic
    assert isinstance(val, sym.Basic) 
    assert val.is_number

def test_unsatisfiable_system(vars):
    """
    Ensure empty list is returned for impossible constraints.
    """
    Z1, _, _ = vars
    
    # Z1 must be 1/2 AND Z1 must be Integer
    # (Since we are solving for Rational Variables, Z1=1/2 is valid for the variable,
    # but if we constrain the result "Z1" to be integer, it must be integer).
    
    # Let's try a direct contradiction:
    # 1. Z1 is an integer
    # 2. Z1 = 1.5 (Impossible)
    
    constraints = [Z1]
    # We add a hard equality constraint that Z1 must be 3/2
    
    results = find_rational_vars_integer_results(
        integer_constraints=constraints,
        nonzero_constraints=[],
        zero_constraints=[Z1-sym.Rational(3,2)],
        variables=[Z1],
        max_result_range=5
    )
    
    assert len(results) == 0

def test_l1_minimization(vars):
    """
    Ensure the solver finds the minimal norm solutions first.
    """
    Z1, Z2, _ = vars
    
    # Two valid solutions for Z1 + Z2 = 0:
    # 1. 0, 0 (Norm 0)
    # 2. 1, -1 (Norm 2)
    # If we forbid 0,0, we should get 1,-1 (or similar small ints)
    
    constraints = [Z1, Z2]
    nonzero = [Z1**2 + Z2**2] # Norm squared != 0 implies not (0,0)
    
    results = find_rational_vars_integer_results(
        integer_constraints=constraints,
        nonzero_constraints=nonzero,
        zero_constraints=[],
        variables=[Z1, Z2],
        max_result_range=5
    )
    
    assert len(results) > 0
    
    # Check that we found the smallest non-zero integers (likely 1 or -1)
    # rather than jumping straight to 5
    first_sol_results = results[0]["results"]
    l1_norm = sum(abs(x) for x in first_sol_results)
    
    # The minimal non-zero L1 norm for two integers is likely 1 (e.g. 1, 0) 
    # or 2 (1, 1) depending on constraints. 
    # It should definitively be small.
    assert l1_norm <= 2


def test_as_long_error():

    Z00, Z10, Z11 = sym.symbols('Z00 Z10 Z11')
    integer_constraints = [Z00 - Z10, Z11*(-Z00 - Z10)/Z00, Z00, -Z10*Z11/Z00, Z10, Z11]
    nonzero_constraints = [sym.simplify(x, rational=True) for x in [3, 3*Z00, Z11*(Z00**2 + Z10**2)/(3*Z00)]]
    variables = [Z00, Z10, Z11]
    max_range = 4

    solutions = find_rational_vars_integer_results(
        integer_constraints=integer_constraints,
        nonzero_constraints=nonzero_constraints,
        zero_constraints=[],
        variables=variables,
        max_result_range=max_range,
        symmetry_map=lambda x: [x, [-v for v in x]]
    )
    assert len(solutions) == 2
    assert sum(abs(s) for s in solutions[0]['results']) == 4

    solutions = find_rational_vars_integer_results(
        integer_constraints=integer_constraints,
        nonzero_constraints=nonzero_constraints,
        variables=variables,
        zero_constraints=[],
        max_result_range=max_range,
        symmetry_map=lambda x: [x]
    )
    assert len(solutions) == 4
    assert sum(abs(s) for s in solutions[0]['results']) == 4


def test_calc_min_cost():

    Z21, Z12, Z01, Z02, Z11, Z22 = sym.symbols('Z21 Z12 Z01 Z02 Z11 Z22')
    integer_constraints =  [-Z21/2, Z12, -Z21, Z12, Z12, -Z21/2, Z21/2, Z21]
    nonzero_constraints =  [8, 4, Z12*Z21/4]
    variables =  [Z12, Z21]
    min_cost, _, _,_,_ = _calc_min_cost(integer_constraints, nonzero_constraints,[],
                                        variables)
    assert min_cost == 10

    integer_constraints =  [-Z01, -2*Z01*Z02/(2*Z01 - Z21), -Z21, -Z02*Z21/(2*Z01 - Z21), -Z02*Z21/(2*Z01 - Z21), Z01 - Z21, Z02, Z01, Z02, Z21]
    nonzero_constraints =  [4*Z01 - 2*Z21, 4, -Z02*Z21**2/(8*Z01 - 4*Z21)]
    variables =  [Z01, Z02, Z21]
    min_cost, _, _,_,_ = _calc_min_cost(integer_constraints, nonzero_constraints,[],
                                        variables)
    assert min_cost == 6

    integer_constraints =  [Z11, -Z22, Z11, -Z22, Z11, Z22, Z22]
    nonzero_constraints =  [2, 4, -Z11*Z22/4]
    variables =  [Z11, Z22]
    min_cost, _, _,_,_ = _calc_min_cost(integer_constraints, nonzero_constraints,[],
                                        variables)
    assert min_cost == 7


def test_rational_vars_zero():

    Z00, Z01, Z02, Z10, Z11, Z12, Z20, Z21, Z22 = sym.symbols('Z00, Z01, Z02, Z10, Z11, Z12, Z20, Z21, Z22')
    vars = [Z00, Z01, Z02, Z10, Z11, Z12, Z20, Z21, Z22]
    is_nonzero = sum([Z00*Z11*Z22/4, -Z00*Z12*Z21/4, -Z01*Z10*Z22/4, -Z02*Z11*Z20/4, Z01*Z12*Z20/4, Z02*Z10*Z21/4])
    integer_constraints = [Z00 - Z10, Z01 - Z11, Z02 - Z12, Z00 - Z20, Z01 - Z21, Z02 - Z22, Z00, Z01, Z02, Z10 - Z20, Z11 - Z21, Z12 - Z22, Z10, Z11, Z12, Z20, Z21, Z22]

    res = find_rational_vars_integer_results(integer_constraints, [is_nonzero],
                                             [], variables=vars)

    assert sum(abs(x) for x in res[0]["results"]) == 9


def test_consistent_results_multiple_runs():
    """
    Regression test: Ensure that multiple runs of the solver produce
    consistent results. This tests for non-determinism issues in Z3.
    """
    Z10, Z11 = sym.symbols('Z10 Z11')
    
    integer_constraints = [-2*Z10, -Z10, Z11, Z10, Z11]
    nonzero_constraints = [1, 3, -2*Z10*Z11/3]
    
    # Run the solver multiple times and check for consistency
    results_list = []
    for _ in range(5):
        results = find_rational_vars_integer_results(
            integer_constraints=integer_constraints,
            nonzero_constraints=nonzero_constraints,
            zero_constraints=[],
            variables=[Z10, Z11],
            max_result_range=4,
            symmetry_map=lambda x: [x]
        )
        results_list.append(results)
    
    # All runs should produce the same results
    first_results = results_list[0]
    for i, results in enumerate(results_list[1:], start=1):
        assert len(results) == len(first_results), f"Run {i} produced different number of solutions"
        for j, (r1, r2) in enumerate(zip(first_results, results)):
            assert r1["results"] == r2["results"], (
                f"Run {i}, solution {j}: results differ. "
                f"Expected {r1['results']}, got {r2['results']}"
            )


def test_cost_constraint_preserved_during_enumeration():
    """
    Regression test: Ensure that when enumerating solutions, all solutions
    have the same (minimal) cost. This catches bugs where the cost constraint
    is popped before enumeration.
    """
    Z01, Z10 = sym.symbols('Z01 Z10')
    
    # This constraint set has multiple solutions at different costs
    integer_constraints = [-Z10, Z01, Z01, Z10]
    nonzero_constraints = [3, -Z01*Z10/3]
    
    results = find_rational_vars_integer_results(
        integer_constraints=integer_constraints,
        nonzero_constraints=nonzero_constraints,
        zero_constraints=[],
        variables=[Z01, Z10],
        max_result_range=4,
        enumerate_sols=True
    )
    
    assert len(results) > 0
    
    # All solutions should have the same cost
    costs = [sum(abs(x) for x in r["results"]) for r in results]
    assert all(c == costs[0] for c in costs), (
        f"Solutions have inconsistent costs: {costs}"
    )
    
    # The cost should be the minimum (4 in this case)
    assert costs[0] == 4, f"Expected minimum cost 4, got {costs[0]}"


def test_heuristic_upper_bound_returns_optimal():
    """
    Regression test: Ensure that when the heuristic finds a solution at the
    lower bound, it is correctly returned without searching higher costs.
    """
    Z10, Z11 = sym.symbols('Z10 Z11')
    
    integer_constraints = [-2*Z10, -Z10, Z11, Z10, Z11]
    nonzero_constraints = [1, 3, -2*Z10*Z11/3]
    
    # Get the minimum cost first
    min_cost, _, _, _, _ = _calc_min_cost(
        integer_constraints, nonzero_constraints, [],
        [Z10, Z11], max_result_range=4
    )
    
    # Now solve with heuristic
    results = find_rational_vars_integer_results(
        integer_constraints=integer_constraints,
        nonzero_constraints=nonzero_constraints,
        zero_constraints=[],
        variables=[Z10, Z11],
        max_result_range=4,
        heuristic_upper=True
    )
    
    assert len(results) > 0
    
    # The result cost should be the minimum achievable
    result_cost = sum(abs(x) for x in results[0]["results"])
    assert result_cost == min_cost, (
        f"Heuristic returned cost {result_cost}, but minimum is {min_cost}"
    )


def test_heuristic_does_not_modify_input_constraints():
    """
    Regression test: Ensure that _heuristic_upper_bound does not modify
    the input nonzero_constraints list.
    """
    Z01, Z10 = sym.symbols('Z01 Z10')
    
    integer_constraints = [Z01, Z10]
    nonzero_constraints = [3, -Z01*Z10/3]
    original_nonzero = nonzero_constraints.copy()
    
    _heuristic_upper_bound(
        integer_constraints=integer_constraints,
        nonzero_constraints=nonzero_constraints,
        zero_constraints=[],
        variables=[Z01, Z10],
        max_result_range=4
    )
    
    # The original list should not be modified
    assert nonzero_constraints == original_nonzero, (
        f"nonzero_constraints was modified from {original_nonzero} to {nonzero_constraints}"
    )


def test_lower_bound_early_exit():
    """
    Test that when the lower bound is immediately satisfiable,
    we get the correct solution without additional searching.
    """
    Z1, Z2 = sym.symbols('Z1 Z2')
    
    # Simple constraints where lower bound should be achievable
    integer_constraints = [Z1, Z2]
    nonzero_constraints = [Z1 * Z2]
    
    results = find_rational_vars_integer_results(
        integer_constraints=integer_constraints,
        nonzero_constraints=nonzero_constraints,
        zero_constraints=[],
        variables=[Z1, Z2],
        max_result_range=4
    )
    
    assert len(results) > 0
    
    # Minimum non-zero solution should have cost 2 (e.g., Z1=1, Z2=1)
    cost = sum(abs(x) for x in results[0]["results"])
    assert cost == 2


def test_enumerate_solutions_sorted_deterministically():
    """
    Test that enumerated solutions are sorted deterministically,
    ensuring consistent ordering across runs.
    """
    Z01, Z10 = sym.symbols('Z01 Z10')
    
    integer_constraints = [Z01, Z10]
    nonzero_constraints = [Z01 * Z10]
    
    # Run twice and compare ordering
    results1 = find_rational_vars_integer_results(
        integer_constraints=integer_constraints,
        nonzero_constraints=nonzero_constraints,
        zero_constraints=[],
        variables=[Z01, Z10],
        max_result_range=2,
        enumerate_sols=True
    )
    
    results2 = find_rational_vars_integer_results(
        integer_constraints=integer_constraints,
        nonzero_constraints=nonzero_constraints,
        zero_constraints=[],
        variables=[Z01, Z10],
        max_result_range=2,
        enumerate_sols=True
    )
    
    assert len(results1) == len(results2)
    for r1, r2 in zip(results1, results2):
        assert r1["results"] == r2["results"], "Solutions not in same order"


def test_symmetry_map_blocks_equivalent_solutions():
    """
    Test that the symmetry_map correctly blocks equivalent solutions.
    With negation symmetry, we should get half as many solutions.
    """
    Z1, Z2 = sym.symbols('Z1 Z2')
    
    integer_constraints = [Z1, Z2]
    nonzero_constraints = [Z1 * Z2]
    
    # Without negation symmetry
    results_no_sym = find_rational_vars_integer_results(
        integer_constraints=integer_constraints,
        nonzero_constraints=nonzero_constraints,
        zero_constraints=[],
        variables=[Z1, Z2],
        max_result_range=2,
        symmetry_map=lambda x: [x]  # Identity only
    )
    
    # With negation symmetry (default)
    results_with_sym = find_rational_vars_integer_results(
        integer_constraints=integer_constraints,
        nonzero_constraints=nonzero_constraints,
        zero_constraints=[],
        variables=[Z1, Z2],
        max_result_range=2,
        symmetry_map=lambda x: [x, [-v for v in x]]
    )
    
    # With symmetry, we should have roughly half the solutions
    # (some solutions may be self-symmetric)
    assert len(results_with_sym) <= len(results_no_sym)
    assert len(results_with_sym) >= len(results_no_sym) // 2


if __name__ == "__main__":
    # test_as_long_error()
    # test_basic_rational_constraints(vars2())
    # test_calc_min_cost()
    test_rational_vars_zero()