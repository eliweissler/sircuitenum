__doc__ = "z3_interface.py: Interface to Z3 SMT solver for finding rational variable assignments satisfying integer constraints."
__author__ = "Eli Weissler"
__version__ = "0.1.0"


__all__ = [
    "find_rational_vars_integer_results",
]

import itertools
import sympy as sym
from z3 import Solver, Real, RealVal, Int, Sum, If, sat, unsat
from z3 import unknown, Optimize, Or, IsInt, set_param, Abs
from z3 import Tactic, Then

from sircuitenum.singular_interface import _eq_as_numer_denom
from sircuitenum.equationset import maximally_compatible_sol

# Set seeds for reproducibility
set_param('smt.random_seed', 7)
set_param('sat.random_seed', 7)


def find_rational_vars_integer_results(integer_constraints, nonzero_constraints, zero_constraints,
                                       variables, max_result_range=4, timeout_ms=int(6*1e04),
                                       block_negative_equivalents=True, heuristic_upper=True):
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
    
    # Consider individual terms for lower bound
    lower_bound, solv, cost_terms, z3_vars, cost = _calc_min_cost(integer_constraints, nonzero_constraints,
                                                                        zero_constraints, variables, max_result_range)
    solv.set("rlimit", 0)
    solv.set("timeout", timeout_ms)
    solv.set(logic='QF_NIA')
    
    # Get a heuristic upper bound by setting the maximum number of variables to zero
    if any(nz == 0 for nz in nonzero_constraints):
        raise ValueError("Provided Already Zero Nonzero Constraint")
    nonzero_constraints = [nz for nz in nonzero_constraints if len(nz.free_symbols) > 0]
    # Are there any variables we could freely set to zero?
    if nonzero_constraints and heuristic_upper:
        upper_bound, res = _heuristic_upper_bound(integer_constraints, nonzero_constraints,
                                                zero_constraints,
                                             variables, max_result_range=max_result_range,
                                             timeout_ms=timeout_ms)
        target_costs = list(range(upper_bound, lower_bound-1,-1))
        iteration_order = "down"

        # Check to see if we're already done
        solv.push()
        solv.add(cost < upper_bound)
        check_result = solv.check()
        solv.pop()
        # print("Checking upper bound", upper_bound, check_result)
        if check_result == unsat:
            solv.add(cost == upper_bound)
            check_result = solv.check()
            return _enumerate_solutions(solv, cost_terms, z3_vars,
                                block_negative_equivalents=block_negative_equivalents)
            # return res
    else:
        upper_bound = len(cost_terms) * max_result_range
        target_costs = list(range(lower_bound, upper_bound+1))
        iteration_order = "up"

    for target_cost in target_costs:
    # for target_cost in range(max_possible_cost, lower_bound-1, -1):
        # print("Checking", target_cost)

        # Push a temporary context to check "Can Cost == k?"
        solv.push()
        solv.add(cost == target_cost)
        
        check_result = solv.check()
        # print("check result", check_result)
        
        if check_result == sat:
            # Verify
            solv.push()
            solv.add(cost < target_cost)
            check_result = solv.check()
            solv.pop()
            if check_result != unsat and iteration_order == "down":
                continue
            elif check_result != unsat and iteration_order == "up":
                raise ValueError("Invalid Min Found")
            # SUCCESS! We found the lowest possible cost.
            # No verification loop needed because we checked 0, 1, 2... in order.
            solutions = _enumerate_solutions(solv, cost_terms, z3_vars,
                                             block_negative_equivalents=block_negative_equivalents)
            return solutions
        
        elif check_result == unknown:
            raise TimeoutError(f"  > Z3 gave up! Reason: {solv.reason_unknown()}")

        solv.pop() # Remove "cost == k", continue to k+1

    return []




def _calc_min_cost(integer_constraints, nonzero_constraints, zero_constraints,
                    variables, max_result_range=4, timeout_ms=200):
    

    # --- STEP 1: PRE-COMPILATION ---
    z3_vars = {var: Real(str(var)) for var in variables}
    solver_strategy = Then('simplify', 'propagate-values', 'smt')
    solv = solver_strategy.solver()
    solv.set("timeout", timeout_ms)

    # 1a. Expressions that are zero
    zero_factors = set()
    for expr in zero_constraints:
        num, den = _eq_as_numer_denom(sym.nsimplify(expr, rational=True))
        if (num, den) not in zero_factors:
            solv.add(num == 0)
            if len(den.free_symbols) > 0:
                solv.add(den != 0)

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

    # 1c. Integer Constraints
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
            # Thorough Estimate -- is it possible
            # to be this value, considering all
            # integer conditions?
            solv.push()
            solv.add(term == val)
            result = solv.check()
            solv.pop()
            # It is solvable, or timed out
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
            # val not present
            if val is None:
                continue
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


def _heuristic_upper_bound(integer_constraints, nonzero_constraints, zero_constraints, variables, max_result_range=4, timeout_ms=int(1e03)):

    # Heuristic solver for upper bound -- assume maximal number of variables are zero
    nz_prod = sym.sympify(1)
    for nz in nonzero_constraints:
        nz_prod *= nz
    nz_numer, nz_denom = _eq_as_numer_denom(nz_prod)
    if isinstance(nz_numer, sym.Add):
        nz_terms = nz_numer.args
    else:
        nz_terms = [nz_numer]
    if isinstance(nz_denom, sym.Expr):
        nonzero_constraints.append(nz_denom)
    best_val = len(integer_constraints)*max_result_range
    res = []
    for i in range(len(nz_terms)):
        is_nonzero = [nz_terms[i]]
        if isinstance(nz_denom, sym.Expr):
            is_nonzero.append(nz_denom)
        zero_subs = {}
        for var in variables:
            if var not in nz_terms[i].free_symbols:
                zero_subs[var] = 0

        new_constraints = [c.subs(zero_subs) for c in integer_constraints if c.subs(zero_subs) != 0]
        new_zero_constraints = [c.subs(zero_subs) for c in zero_constraints if c.subs(zero_subs) != 0]
        new_variables = sorted(set(itertools.chain.from_iterable(c.free_symbols for c in new_constraints+zero_constraints+list(nz_terms))), key=str)
        this_res = find_rational_vars_integer_results(new_constraints, nonzero_constraints=is_nonzero,
                                                        zero_constraints=new_zero_constraints,
                                                        variables=new_variables,
                                                    max_result_range=max_result_range,
                                                    timeout_ms=timeout_ms, heuristic_upper=False)
        if this_res:
            val = sum(abs(x) for x in this_res[0]["results"])
            if val <= best_val:
                this_res_proc = []
                for r in this_res:
                    this_res_proc.append({"results": r["results"], "variables": r["variables"] | zero_subs})
            if val < best_val:
                res = this_res_proc
                best_val = val
            elif val == best_val:
                # Explicitly add the zero substitutions to the results
                res.extend(this_res_proc)
    return best_val, res


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