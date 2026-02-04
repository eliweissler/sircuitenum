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


def find_rational_vars_integer_results(integer_constraints, nonzero_constraints, zero_constraints,
                                       variables, max_result_range=5, timeout_ms=int(1e06),
                                       heuristic_upper=True, apriori_sol=None,
                                       symmetry_map=lambda x: [x, [-i for i in x]],
                                       enumerate_sols=True, debug=True):
    """
    Find rational variable assignments such that integer constraints evaluate to integers
    with minimal L1 norm.

    Uses the Z3 SMT solver to find rational values for the given variables such that:

    1. All expressions in ``integer_constraints`` evaluate to integers within
       ``[-max_result_range, max_result_range]``.
    2. All expressions in ``nonzero_constraints`` evaluate to non-zero values.
    3. All expressions in ``zero_constraints`` evaluate to zero.
    4. The L1 norm (sum of absolute values) of the ``integer_constraints`` results
       is minimized.

    The algorithm uses a "Linear Cost Sweep" strategy:
    
    1. Compute a lower bound on the cost by minimizing individual integer constraints independently.
    2. Optionally use a heuristic to find an upper bound by greedily setting variables to zero.
    3. Sweep through possible costs to find the minimum achievable cost.
    4. Enumerate all solutions at that minimum cost.

    :param integer_constraints: Expressions that must evaluate to integer values.
    :type integer_constraints: list[sympy.Expr]
    :param nonzero_constraints: Expressions that must evaluate to non-zero values.
    :type nonzero_constraints: list[sympy.Expr]
    :param zero_constraints: Expressions that must evaluate to zero.
    :type zero_constraints: list[sympy.Expr]
    :param variables: Symbols representing the unknowns to solve for.
    :type variables: list[sympy.Symbol]
    :param max_result_range: Maximum absolute value allowed for integer constraint results.
        Defaults to 4.
    :type max_result_range: int
    :param timeout_ms: Timeout in milliseconds for solver operations. Defaults to 100000.
    :type timeout_ms: int
    :param heuristic_upper: Whether to use heuristic search to find an upper bound
        on the optimal cost. Defaults to True.
    :type heuristic_upper: bool
    :param apriori_sol: Known upper bound on the cost from a previous solution.
        If provided, only searches for solutions with cost <= this value.
        Defaults to None.
    :type apriori_sol: int | None
    :param symmetry_map: Function mapping a result vector to equivalent vectors under
        symmetry. Used to block symmetric solutions during enumeration.
        Defaults to identity and negation.
    :type symmetry_map: Callable[[list[int]], list[list[int]]]
    :param enumerate_sols: Whether to enumerate all solutions at the minimum cost,
        or just return one. Defaults to True.
    :type enumerate_sols: bool

    :returns: List of solutions at the globally minimal cost. Each solution is a
        dictionary with keys:

        - ``'results'`` (*list[int]*): Integer values of the ``integer_constraints``.
        - ``'variables'`` (*dict[sympy.Symbol, sympy.Rational]*): Variable assignments.

        Returns an empty list if no solution exists.
    :rtype: list[dict]

    :raises ValueError: If an invalid minimum is found or constraints are inconsistent.
    :raises TimeoutError: If the solver times out during critical operations.

    Example
    -------
    >>> Z1, Z2 = sym.symbols('Z1 Z2')
    >>> solutions = find_rational_vars_integer_results(
    ...     integer_constraints=[Z1/2, Z2/3],
    ...     nonzero_constraints=[Z1 * Z2],
    ...     zero_constraints=[],
    ...     variables=[Z1, Z2]
    ... )
    """

    # Set random seed for reproducibility
    set_param('smt.random_seed', 7)
    set_param('sat.random_seed', 7)
    
    # Consider individual terms for lower bound
    lower_bound, solv, cost_terms, z3_vars, cost = _calc_min_cost(integer_constraints, nonzero_constraints,
                                                                        zero_constraints, variables, max_result_range)
    solv.set("rlimit", 0)
    solv.set("timeout", timeout_ms)
    solv.set(logic='QF_NIA')

    # If we have an apriori solution that we need to beat, check it now
    upper_bound = len(cost_terms) * max_result_range
    if not apriori_sol is None:
        solv.push()
        # If apriori solution is already lower than lower bound, give up
        if apriori_sol < lower_bound:
            return []
        try:
            if debug:
                print(f"  > Using apriori solution cost = {apriori_sol} to limit search.")
            solv.add(cost <= apriori_sol)
            check_result = solv.check()
            if debug:
                print(f"    > Check result: {check_result}")
            # Nothing can achieve the apriori lower bound
            if check_result == unsat:
                return []
            elif check_result == sat:
                upper_bound = apriori_sol
        # Don't use apriori if it times out
        # switch to heuristic upper bound search
        except TimeoutError:
            apriori_sol = None
            heuristic_upper = True
        solv.pop()
    
    # Check for whether the lower bound is achievable, if it is then we're done
    solv.push()
    try:
        if debug:
            print(f"  > Checking if lower bound cost = {lower_bound} is achievable...")
        solv.add(cost == lower_bound)
        check_result = solv.check()
        if debug:
            print(f"    > Check result: {check_result}")
        if check_result == sat:
            if enumerate_sols:
                return _enumerate_solutions(solv, cost_terms, z3_vars,
                                symmetry_map=symmetry_map)
            else:
                return _enumerate_solutions(solv, cost_terms, z3_vars,
                                max_solutions=1)
    except TimeoutError:
        pass
    solv.pop()

    # Get a heuristic upper bound by setting the maximum number of variables to zero
    if any(nz == 0 for nz in nonzero_constraints):
        raise ValueError("Provided Already Zero Nonzero Constraint")
    nonzero_constraints = [nz for nz in nonzero_constraints if len(nz.free_symbols) > 0]
    # Are there any variables we could freely set to zero?
    if nonzero_constraints and heuristic_upper:
        try:
            upper_other, res = _heuristic_upper_bound(integer_constraints, nonzero_constraints,
                                                    zero_constraints,
                                                variables, max_result_range=max_result_range,
                                                timeout_ms=timeout_ms)
            upper_bound = min(upper_bound, upper_other)
            
        # Don't use heuristic upper if it times out
        except TimeoutError:
            heuristic_upper = False
    # Do we have an upper bound already?
    # If so search downwards from it
    if heuristic_upper or upper_bound < len(cost_terms) * max_result_range:
        target_costs = list(range(upper_bound, lower_bound-1,-1))
        iteration_order = "down"

        # Check to see if we're already done
        done = False
        if upper_bound == lower_bound:
            done = True
        else:
            try:
                solv.push()
                solv.add(cost < upper_bound)
                check_result = solv.check()
                solv.pop()
                if check_result == unsat:
                    done = True
            except TimeoutError:
                pass
        if done:
            solv.push()
            solv.add(cost == upper_bound)
            check_result = solv.check()
            if enumerate_sols:
                solutions = _enumerate_solutions(solv, cost_terms, z3_vars,
                                symmetry_map=symmetry_map)
            else:
                solutions = _enumerate_solutions(solv, cost_terms, z3_vars,
                                symmetry_map=symmetry_map, max_solutions=1)
            # If no solutions found, check if it's because lower==upper but system is infeasible
            if not solutions:
                if check_result == unsat:
                    print("Warning: Lower and Upper Bound Equal but System Infeasible")
                else:
                    raise ValueError("Lower and Upper Bound Equal but No Solutions Found")
            return solutions
    else:
        target_costs = list(range(lower_bound, upper_bound+1))
        iteration_order = "up"

    for target_cost in target_costs:

        print(f"  > Trying Cost = {target_cost}...")

        # Push a temporary context to check "Can Cost == k?"
        solv.push()
        solv.add(cost == target_cost)
        check_result = solv.check()
        
        if check_result == sat:
            # Verify
            solv.pop()
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
            solv.add(cost == target_cost)
            if enumerate_sols:
                solutions = _enumerate_solutions(solv, cost_terms, z3_vars,
                                             symmetry_map=symmetry_map)
            else:
                solutions = _enumerate_solutions(solv, cost_terms, z3_vars,
                                             max_solutions=1)
            return solutions
        
        elif check_result == unknown:
            raise TimeoutError(f"  > Z3 gave up! Reason: {solv.reason_unknown()}")
        
        solv.pop()

        if not apriori_sol is None and target_cost >= apriori_sol:
            # We've already established nothing can beat apriori_sol
            break


    return []




def _calc_min_cost(integer_constraints, nonzero_constraints, zero_constraints,
                    variables, max_result_range=5, timeout_ms=200):
    """
    Calculate a lower bound on the L1 cost and initialize the Z3 solver.

    Sets up the Z3 solver with all constraints and computes a theoretical minimum
    cost by minimizing each integer constraint independently, subject to the nonzero
    and zero constraints.

    :param integer_constraints: Expressions that must evaluate to integer values.
    :type integer_constraints: list[sympy.Expr]
    :param nonzero_constraints: Expressions that must evaluate to non-zero values.
    :type nonzero_constraints: list[sympy.Expr]
    :param zero_constraints: Expressions that must evaluate to zero.
    :type zero_constraints: list[sympy.Expr]
    :param variables: Symbols representing the unknowns to solve for.
    :type variables: list[sympy.Symbol]
    :param max_result_range: Maximum absolute value allowed for integer constraint
        results. Defaults to 4.
    :type max_result_range: int
    :param timeout_ms: Timeout in milliseconds for individual satisfiability checks
        during lower bound computation. Defaults to 200.
    :type timeout_ms: int

    :returns: A tuple containing:

        - ``min_possible_cost`` (*int*): Lower bound on the achievable L1 cost.
        - ``solv`` (*z3.Solver*): Configured Z3 solver with all constraints added.
        - ``cost_terms`` (*list*): Z3 expressions for each integer constraint result.
        - ``z3_vars`` (*dict[sympy.Symbol, z3.Real]*): Mapping from SymPy symbols
          to Z3 variables.
        - ``cost`` (*z3.Int*): Z3 integer variable representing the total L1 cost.

    :rtype: tuple[int, z3.Solver, list, dict, z3.Int]

    :raises ValueError: If the computed lower bound is incorrect (sanity check fails).
    """

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
            # Add nonzero constraints for denominator
            factored_den = sym.factor(den)
            if factored_den.is_Mul:
                factors = factored_den.args
            else:
                factors = [factored_den]
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
            if val < max_result_range:
                min_val += 1
        min_possible_cost += min_val
        cost_term_dict[term] = min_val
    solv.add(cost == total_cost)

    # Add bounds on cost
    for z in integer_expr.values():
        solv.add(z >= -max_result_range)
        solv.add(z <= max_result_range)


    # Verify
    solv.push()
    solv.add(cost < min_possible_cost)
    check_result = solv.check()
    if check_result == sat:
        raise ValueError("Bad minimum value")
    solv.pop()

    return min_possible_cost, solv, cost_terms, z3_vars, cost


def _enumerate_solutions(solver_with_state, tracked_exprs, z3_vars, max_solutions=1000, symmetry_map=lambda x: [x],
                         verify_same_cost=True):
    """
    Enumerate all unique solutions for the current solver state.

    Iteratively finds satisfying assignments, extracts the solution, blocks it
    (along with symmetric equivalents), and repeats until no more solutions exist
    or the maximum is reached.

    The solver must already have appropriate cost constraints added before calling
    this function. Solutions are blocked by adding constraints that exclude the
    found result vector and all its symmetric equivalents.

    :param solver_with_state: Z3 solver with constraints already configured,
        including any cost equality constraints.
    :type solver_with_state: z3.Solver
    :param tracked_exprs: Z3 expressions whose values define a solution
        (typically the integer constraint terms).
    :type tracked_exprs: list[z3.ExprRef]
    :param z3_vars: Mapping from SymPy symbols to their corresponding Z3 variables.
    :type z3_vars: dict[sympy.Symbol, z3.Real]
    :param max_solutions: Maximum number of solutions to enumerate. Defaults to 1000.
    :type max_solutions: int
    :param symmetry_map: Function mapping a result vector to all equivalent vectors
        under symmetry. All symmetric equivalents are blocked together.
        Defaults to identity only.
    :type symmetry_map: Callable[[list[int]], list[list[int]]]
    :param verify_same_cost: Whether to verify that all enumerated solutions have
        the same cost. Defaults to True.
    :type verify_same_cost: bool

    :returns: List of solutions, sorted deterministically. Each solution is a
        dictionary with keys:

        - ``'results'`` (*list[int]*): Integer values of the tracked expressions.
        - ``'variables'`` (*dict[sympy.Symbol, sympy.Rational]*): Variable assignments.

    :rtype: list[dict]

    .. note::
        This function modifies the solver state by adding blocking clauses.
        The solver should not be reused for other purposes after calling this.
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
            if max_solutions != 1:
                print(f"  > Reached max solutions limit of {max_solutions}, stopping enumeration.")
            break

        # 3. Block this specific result vector, as well as any equivalents under symmetry
        n_expr = len(tracked_exprs)
        for res_perm in symmetry_map(res_vals):
            block_clause = [tracked_exprs[i] != res_perm[i] for i in range(n_expr)]
            solver_with_state.add(Or(block_clause))

    # Sort results deterministically to ensure consistent ordering across runs
    # Sort by: results tuple first, then by variable assignments
    def _sort_key(sol):
        res_val = sum(abs(x) for x in sol["results"])
        res_key = tuple([-x for x in sol["results"]])
        var_key = tuple(sorted((str(k), str(v)) for k, v in sol["variables"].items()))
        return (res_val, res_key, var_key)
    results.sort(key=_sort_key)

    # Verify that all results have the same cost
    if len(results) > 1 and verify_same_cost:
        first_cost = sum(abs(x) for x in results[0]["results"])
        for i,r in enumerate(results[1:]):
            this_cost = sum(abs(x) for x in r["results"])
            if this_cost != first_cost:
                raise ValueError(
                    f"Inconsistent costs in enumerated solutions: "
                    f"solution 0 has cost {first_cost} with results {results[0]['results']}, "
                    f"but solution {i+1} has cost {this_cost} with results {r['results']}"
                )

    return results


def _heuristic_upper_bound(integer_constraints, nonzero_constraints, zero_constraints, variables, max_result_range=5, timeout_ms=int(1e03)):
    """
    Find a heuristic upper bound on the optimal cost by maximizing zero variables.

    Attempts to find a valid solution by setting as many variables to zero as
    possible while still satisfying the nonzero constraints. This provides an
    upper bound that can speed up the main optimization by allowing a top-down
    search strategy.

    The algorithm:
    
    1. Expands the product of all nonzero constraints into terms.
    2. For each term, identifies which variables must be nonzero for that term.
    3. Sets all other variables to zero and solves the reduced problem.
    4. Returns the best (lowest cost) solution found.

    :param integer_constraints: Expressions that must evaluate to integer values.
    :type integer_constraints: list[sympy.Expr]
    :param nonzero_constraints: Expressions that must evaluate to non-zero values.
    :type nonzero_constraints: list[sympy.Expr]
    :param zero_constraints: Expressions that must evaluate to zero.
    :type zero_constraints: list[sympy.Expr]
    :param variables: Symbols representing the unknowns to solve for.
    :type variables: list[sympy.Symbol]
    :param max_result_range: Maximum absolute value allowed for integer constraint
        results. Defaults to 4.
    :type max_result_range: int
    :param timeout_ms: Timeout in milliseconds for each sub-problem. Defaults to 1000.
    :type timeout_ms: int

    :returns: A tuple containing:

        - ``best_val`` (*int*): The lowest cost found (upper bound on optimal).
        - ``res`` (*list[dict]*): Solutions achieving the best cost, with zero
          substitutions included in the variable assignments.

    :rtype: tuple[int, list[dict]]

    :raises TimeoutError: If solver times out on a sub-problem (propagated from
        recursive call to :func:`find_rational_vars_integer_results`).
    """

    # Heuristic solver for upper bound -- assume maximal number of variables are zero
    nz_prod = sym.sympify(1)
    for nz in nonzero_constraints:
        nz_prod *= nz
    nz_numer, nz_denom = _eq_as_numer_denom(nz_prod)
    nz_numer = sym.expand(nz_numer)
    if isinstance(nz_numer, sym.Add):
        nz_terms = nz_numer.args
    else:
        nz_terms = [nz_numer]
    if isinstance(nz_denom, sym.Expr):
        nonzero_constraints = list(nonzero_constraints) + [nz_denom]
    best_val = len(integer_constraints)*max_result_range
    res = []
    unique_subs = set()
    for nz_term in sorted(nz_terms, key=lambda x: len(x.free_symbols)):
        is_nonzero = [nz_term]
        if isinstance(nz_denom, sym.Expr):
            is_nonzero.append(nz_denom)
        zero_subs = {}
        for var in variables:
            if var not in nz_term.free_symbols:
                zero_subs[var] = 0
        unique_subs_key = tuple(sorted(zero_subs.items(), key=lambda x: str(x[0])))
        if unique_subs_key in unique_subs:
            continue
        unique_subs.add(unique_subs_key)

        new_constraints = [c.subs(zero_subs) for c in integer_constraints if c.subs(zero_subs) != 0]
        new_zero_constraints = [c.subs(zero_subs) for c in zero_constraints if c.subs(zero_subs) != 0]

        # Make sure to only include variables that remain finite
        if any(c.has(sym.core.numbers.ComplexInfinity) or c.has(sym.core.numbers.Infinity) for c in new_constraints+new_zero_constraints):
            continue

        new_variables = sorted(set(itertools.chain.from_iterable(c.free_symbols for c in new_constraints+zero_constraints+list(nz_terms))), key=str)
        this_res = find_rational_vars_integer_results(new_constraints, nonzero_constraints=is_nonzero,
                                                        zero_constraints=new_zero_constraints,
                                                        variables=new_variables,
                                                    max_result_range=max_result_range,
                                                    timeout_ms=timeout_ms, heuristic_upper=False,
                                                    apriori_sol=best_val, enumerate_sols=False)
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