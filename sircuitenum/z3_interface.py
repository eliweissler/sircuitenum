__doc__ = "z3_interface.py: Interface to Z3 SMT solver for finding rational variable assignments satisfying integer constraints."
__author__ = "Eli Weissler"
__version__ = "0.1.0"


__all__ = [
    "find_rational_vars_integer_results",
]
import sympy as sym
from z3 import Solver, Real, RealVal, Int, Sum, If, sat, unsat, unknown, Optimize, Or, IsInt, set_param

# Set seeds for reproducibility
set_param('smt.random_seed', 7)
set_param('sat.random_seed', 7)


def find_rational_vars_integer_results(integer_constraints, nonzero_constraints,
                                       variables, max_result_range=5, timeout_ms=10000,
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
    
    # --- STEP 1: PRE-COMPILATION ---
    z3_vars = {var: Real(str(var)) for var in variables}
    solv = Solver()
    
    if timeout_ms:
        solv.set("timeout", timeout_ms)

    tracked_exprs = []
    
    # 1a. Integer Constraints
    # We simplify ONCE here to avoid overhead inside loops.
    for expr in integer_constraints:
        expr = sym.simplify(expr, rational=True)
        if isinstance(expr, sym.Eq):
            lhs = _sympy_to_z3(expr.lhs, z3_vars)
            rhs = _sympy_to_z3(expr.rhs, z3_vars)
            solv.add(lhs == rhs)
        else:
            z = _sympy_to_z3(expr, z3_vars)
            solv.add(IsInt(z))
            solv.add(z >= -max_result_range)
            solv.add(z <= max_result_range)
            tracked_exprs.append(z)

    # 1b. Nonzero Constraints
    tracked_expr_map = {str(z): i for i, z in enumerate(tracked_exprs)}
    forced_nonzero_indices = set()
    for expr in nonzero_constraints:
        # Factor expression to handle products and add 
        # Each factor as a separate non-zero constraint.
        simple_expr = sym.simplify(expr, rational=True)
        factored_expr = sym.factor(simple_expr)
        if factored_expr.is_Mul:
            factors = factored_expr.args
        else:
            factors = [factored_expr]
        for sub_expr in factors:
            z_expr = _sympy_to_z3(sub_expr, z3_vars)
            solv.add(z_expr != 0)

    # 1c. Cost Function (Sum of Absolute Values)
    cost = Int('cost')
    solv.add(cost == Sum([If(e >= 0, e, -e) for e in tracked_exprs]))

    # --- STEP 2: LINEAR COST SWEEP ---
    # Theoretical Max Cost = (Number of Exprs) * (Max Value per Expr)
    # We start at 0 and work up. The first SAT we hit is the proven global minimum.
    max_possible_cost = len(tracked_exprs) * max_result_range
    
    # Theoretical Min Cost = sum(min cost per term)
    min_possible_cost = 0
    # Very fast timeout (50ms) per check to infer minimum cost from forced non-zeros
    solv.set("timeout", 50)
    for i, z in enumerate(tracked_exprs):
        min_val = 0
        for val in range(0, max_result_range + 1):
            solv.push()
            solv.add(z == val)
            result = solv.check()
            solv.pop()
            if result != unsat:
                break
            min_val += 1
        min_possible_cost += min_val
    # Reset timeout to the user's main timeout (or default)
    if timeout_ms:
        solv.set("timeout", timeout_ms)
    else:
        solv.set("timeout", 10**8) # Reset to Large Timeout if None specified

    for target_cost in range(min_possible_cost, max_possible_cost + 1):
        # Push a temporary context to check "Can Cost == k?"
        solv.push()
        solv.add(cost == target_cost)
        
        check_result = solv.check()
        
        if check_result == sat:
            # SUCCESS! We found the lowest possible cost.
            # No verification loop needed because we checked 0, 1, 2... in order.
            solutions = _enumerate_solutions(solv, tracked_exprs, z3_vars,
                                             block_negative_equivalents=block_negative_equivalents)
            solv.pop() 
            return solutions
        
        elif check_result == unknown:
            raise TimeoutError("Z3 Solver timed out during cost sweep.")

        solv.pop() # Remove "cost == k", continue to k+1

    return []

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
    if e.is_Integer:
        return int(e)
    elif e.is_Rational:
        return RealVal(e.p) / RealVal(e.q)
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