import pytest
import sympy as sym
from sircuitenum.z3_interface import find_rational_vars_integer_results, _calc_min_cost

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
        block_negative_equivalents=True
    )
    assert len(solutions) == 2
    assert sum(abs(s) for s in solutions[0]['results']) == 4

    solutions = find_rational_vars_integer_results(
        integer_constraints=integer_constraints,
        nonzero_constraints=nonzero_constraints,
        variables=variables,
        zero_constraints=[],
        max_result_range=max_range,
        block_negative_equivalents=False
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
if __name__ == "__main__":
    # test_as_long_error()
    # test_basic_rational_constraints(vars2())
    # test_calc_min_cost()
    test_rational_vars_zero()