__doc__ = "z3_interface.py: Interface to Z3 SMT solver for finding rational variable assignments satisfying integer constraints."
__author__ = "Eli Weissler"
__version__ = "0.1.0"


__all__ = [
    "find_rational_vars_integer_results",
]
import sympy as sym
from z3 import Solver, Real, RealVal, Int, Sum, If, sat, unsat, unknown, Optimize, Or, IsInt, set_param, Abs

from sircuitenum.singular_interface import _eq_as_numer_denom

# Set seeds for reproducibility
set_param('smt.random_seed', 7)
set_param('sat.random_seed', 7)


def find_rational_vars_integer_results(integer_constraints, nonzero_constraints,
                                       variables, max_result_range=4, timeout_ms=10000,
                                       block_negative_equivalents=True):
    """
    Finds rational variable assignments that satisfy integer and nonzero constraints with minimal total cost.
    Uses the Z3 SMT solver to find solutions that meet the following criteria:

    1. All expressions in ``integer_constraints`` evaluate to integers within ``[-max_result_range, max_result_range]``.
    2. All expressions in ``nonzero_constraints`` evaluate to non-zero values.
    3. The :math:`L_1` norm (sum of absolute values) of the ``integer_constraints`` results is minimized.

    **Algorithm:**
    The solver employs a "Linear Cost Sweep" strategy combined with Domain Reduction. It first infers 
    a minimum possible cost by checking if specific terms are forced to be non-zero. It then rigorously 
    checks for solutions with Total Cost = :math:`C_{min}`, :math:`C_{min}+1`, etc. 
    
    This guarantees that the first set of solutions found is globally optimal with respect to the 
    :math:`L_1` cost

    :param integer_constraints: A list of SymPy expressions that must evaluate to integer values.
    :type integer_constraints: list[sympy.Expr]
    :param nonzero_constraints: A list of SymPy expressions that must evaluate to a non-zero value.
                                Used to enforce validity (e.g., determinant != 0).
    :type nonzero_constraints: list[sympy.Expr]
    :param variables: A list of SymPy symbols representing the unknowns to solve for.
    :type variables: list[sympy.Symbol]
    :param max_result_range: The inclusive maximum absolute value allowed for the result of 
                             any expression in ``integer_constraints``. Defaults to 5.
    :type max_result_range: int
    :param timeout_ms: Optional timeout in milliseconds for the solver operations. If None, no timeout is applied.
    :type timeout_ms: int | None
    :return: A list of unique solutions found at the globally minimal cost. Each solution is a dictionary containing:
    
             * **'results'** (*list[int]*): The integer values of the ``integer_constraints``.
             * **'variables'** (*dict[sympy.Symbol, sympy.Rational]*): Mapping of input variables to their resolved rational values.
             
    :rtype: list[dict]
    """

    # print("integer_constraints = ", integer_constraints)
    # print("nonzero_constraints = ", nonzero_constraints)
    # print("variables = ", variables)

    min_possible_cost, solv, cost_terms, z3_vars, cost = _calc_min_cost(integer_constraints, nonzero_constraints,
                                                variables, max_result_range,
                                                timeout_ms=200)
    
    solv.set("timeout", timeout_ms)



    # Theoretical Max Cost = (Number of Exprs) * (Max Value per Expr)
    # We start at 0 and work up. The first SAT we hit is the proven global minimum.
    max_possible_cost = len(cost_terms) * max_result_range

    for target_cost in range(min_possible_cost, max_possible_cost + 1):

        # Push a temporary context to check "Can Cost == k?"
        solv.push()
        solv.add(cost == target_cost)
        
        check_result = solv.check()
        
        if check_result == sat:
            # SUCCESS! We found the lowest possible cost.
            # No verification loop needed because we checked 0, 1, 2... in order.
            solutions = _enumerate_solutions(solv, cost_terms, z3_vars,
                                             block_negative_equivalents=block_negative_equivalents)
            solv.pop() 
            return solutions
        
        elif check_result == unknown:
            raise TimeoutError("Z3 Solver timed out during cost sweep.")

        solv.pop() # Remove "cost == k", continue to k+1

    return []


def _calc_min_cost(integer_constraints, nonzero_constraints,
                    variables, max_result_range=4, timeout_ms=200):
    
    # --- STEP 1: PRE-COMPILATION ---
    z3_vars = {var: Real(str(var)) for var in variables}
    solv = Solver()
    solv.set("timeout", timeout_ms)

    # 1a. Integer Constraints
    # We simplify ONCE here to avoid overhead inside loops.
    cost_terms = []
    integer_expr = {}
    for expr in integer_constraints:
        rat_expr = sym.simplify(expr, rational=True)
        num, den = _eq_as_numer_denom(rat_expr)
        if (num, den) in integer_expr:
            cost_terms.append(integer_expr[(num, den)])
            continue
        # Simple linear/poly constraint: Z01, Z02-Z22, etc.
        # Or one with rational coefficients: Z01/2 etc.
        # Bound it directly
        if len(den.free_symbols) == 0:    
            z = _sympy_to_z3(sym.simplify(num/den), z3_vars)
            if not isinstance(z, int):
                solv.add(IsInt(z))
                solv.add(z >= -max_result_range)
                solv.add(z <= max_result_range)
                integer_expr[(num, den)] = z
            cost_terms.append(z)
        # Rational constraint: N/D
        # Transform: N == k * D
        # with auxiliary variable k
        # (bound k)
        else:
            k = Int(f"aux_{len(cost_terms)}")
            z_num = _sympy_to_z3(num, z3_vars)
            z_den = _sympy_to_z3(den, z3_vars)
            solv.add(z_num == k * z_den)
            solv.add(k >= -max_result_range)
            solv.add(k <= max_result_range)
            integer_expr[(num, den)] = k
            cost_terms.append(k)

    # 1b. Nonzero Constraints
    # Consider numerator and denominator seperately
    nz_factors = set()
    for expr in nonzero_constraints:
        num, den = _eq_as_numer_denom(sym.nsimplify(expr, rational=True))
        all_expr = [num]
        if den != 1:
            all_expr.append(den)
        for nz_expr in all_expr:
            factored_expr = sym.factor(nz_expr)
            if factored_expr.is_Mul:
                factors = factored_expr.args
            else:
                factors = [factored_expr]
            for sub_expr in factors:
                if sub_expr not in nz_factors:
                    if len(sub_expr.free_symbols) == 0 and expr != 0:
                        continue
                    z_expr = _sympy_to_z3(sub_expr, z3_vars)
                    solv.add(z_expr != 0)
                    nz_factors.add(z_expr)

    # 1c. Cost Function (Sum of Absolute Values)
    cost = Int('cost')
    
    
    # Theoretical Min Cost = sum(min cost per term)
    min_possible_cost = 0
    cost_term_dict = {}
    total_cost = None
    for term in cost_terms:
        if total_cost is None:
            total_cost = Abs(term)
        else:
            total_cost += Abs(term)
        if term in cost_term_dict:
            min_possible_cost += cost_term_dict[term]
            continue
        min_val = 0
        for val in range(0, max_result_range + 1):
            solv.push()
            solv.add(term == val)
            result = solv.check()
            solv.pop()
            if result != unsat:
                break
            min_val += 1
        min_possible_cost += min_val
        cost_term_dict[term] = min_val
    solv.add(cost == total_cost)

    # Verify
    solv.push()
    solv.add(cost < min_possible_cost)
    check_result = solv.check()
    if check_result == sat:
        raise ValueError("Bad minimum value")
    solv.pop()

    return min_possible_cost, solv, cost_terms, z3_vars, cost


def _enumerate_solutions(solver_with_state, tracked_exprs, z3_vars, max_solutions=1000,
                         block_negative_equivalents=True):
    """
    Enumerates all unique solutions for the current solver state.
    """
    results = []
    n_res = 0
    
    # We iterate while the solver remains Satisfiable.
    # Each time we find a solution, we block it and ask for another.
    while solver_with_state.check() == sat:
        m = solver_with_state.model()
        
        # 1. Extract Results
        res_vals = [m.eval(e).as_long() for e in tracked_exprs]
        
        # 2. Extract Variables (as SymPy Rationals)
        var_vals = {}
        for var, z_var in z3_vars.items():
            val = m[z_var]
            if hasattr(val, 'numerator_as_long'):
                val_sym = sym.Rational(val.numerator_as_long(), val.denominator_as_long())
            else:
                val_sym = sym.Integer(val.as_long())
            var_vals[var] = val_sym
            
        results.append({"results": res_vals, "variables": var_vals})
        n_res += 1
        if n_res >= max_solutions:
            print("Reached max_solutions limit during enumeration.")
            break

        # 3. Block this specific result vector, as well as its negative equivalent if desired.
        block_clause = [tracked_exprs[i] != res_vals[i] for i in range(len(tracked_exprs))]
        solver_with_state.add(Or(block_clause))
        if block_negative_equivalents:
            block_clause = [tracked_exprs[i] != -res_vals[i] for i in range(len(tracked_exprs))]
            solver_with_state.add(Or(block_clause))


        
    return results

def _sympy_to_z3(e, z3_vars):
    e = sym.sympify(e)
    if e.is_Integer: 
        return int(e)
    if e.is_Symbol: 
        return z3_vars[e]
    if e.is_Add: 
        return Sum([_sympy_to_z3(arg, z3_vars) for arg in e.args])
    if e.is_Mul:
        res = 1
        for arg in e.args: 
            res = res * _sympy_to_z3(arg, z3_vars)
        return res
    if e.is_Pow:
        base, exp = e.args
        return _sympy_to_z3(base, z3_vars) ** int(exp)
    if e.is_Rational:
        return RealVal(e.p) / RealVal(e.q)
    raise ValueError(f"Unsupported expression type: {type(e)}")


if __name__ == "__main__":

    # --- Example Usage ---
    Z1, Z2 = sym.symbols('Z1 Z2')

    # We want these expressions to be integers:
    # 1. Z1/2
    # 2. Z2/3
    # 3. Z1 + Z2
    
    integer_exprs = [
        Z1 / 2,         
        Z2 / 3,         
        Z1 + Z2         
    ]
    
    # Example Nonzero Constraint:
    # The product Z1*Z2 must not be zero (implies neither is zero)
    nonzero_exprs = [
        Z1 * Z2
    ]

    print("Solving...")
    all_sols = find_rational_vars_integer_results(
        integer_constraints=integer_exprs, 
        nonzero_constraints=nonzero_exprs, 
        variables=[Z1, Z2], 
        max_result_range=5
    )

    print(f"Found {len(all_sols)} unique integer profiles.")
    for i, s in enumerate(all_sols):
        print(f"Solution {i+1}:")
        print(f"  Integer Results: {s['results']}")
        print(f"  Variable Ex:     {s['variables']}")