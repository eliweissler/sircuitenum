"""
Singular Interface for Groebner Cover computation.

This module provides a Python/Sage interface to the Singular grobcov library
for computing the Groebner Cover of a parametric polynomial ideal.

Author: Eli Weissler
Version: 0.1.0
"""

__all__ = [
    "solve_with_singular",
]

import re
import itertools
from pathlib import Path
from typing import List, Dict, Any, Optional

# Load the Singular library and define the procedure
from sage.interfaces.singular import singular
lib_path = Path(__file__).with_name("poly_solver.sing")
singular.eval(f'LIB "{str(lib_path)}";')

import sympy as sym


def solve_with_singular(equations, solve_vars=None):
    """
    Solves a system of SymPy equations using minAssGTZ -> indepSet -> grobcov.
    
    This approach uses the Groebner Cover algorithm to find all solution branches
    including singular/degenerate cases where parameters take special values.
    
    The workflow is:
    1. Compute minimal associated primes (irreducible components)
    2. For each component, find the maximal independent set (free variables)
    3. Run grobcov with those as parameters to get the full stratification
    
    Args:
        equations (list): List of SymPy expressions (assumed equal to 0).
        solve_vars (list, optional): List of SymPy symbols to solve for (potential unknowns).
            If provided, only these variables can become dependent variables, and all
            other symbols are treated as parameters from the start.
            If None, the algorithm auto-discovers which variables are independent
            vs dependent for each component.

    Returns:
        list[dict]: A list of branches. Each branch is a dict with:
            - 'id': Branch identifier
            - 'component': Which prime component this came from
            - 'params': List of parameter names (independent variables)
            - 'vars': List of variable names (dependent variables)
            - 'constraints': List of SymPy expressions (parameter constraints, = 0)
            - 'nonnull': List of SymPy expressions (must be != 0)
            - 'basis': The Groebner basis for this segment
            - 'mappings': Dict mapping variables to their solutions or "Free Parameter"
    """
    # Quick inconsistency check: if any equation is a non-zero constant, system is inconsistent
    eq_simplified = []
    for eq in equations:
        simplified = sym.simplify(eq)
        if simplified.is_number and simplified != 0:
            return []  # Inconsistent system
        eq_simplified.append(simplified)
    equations = eq_simplified
    
    # Collect all symbols from equations
    all_symbols = set()
    for eq in equations:
        all_symbols.update(eq.free_symbols)
    
    # Handle edge case: no variables in the system
    if not all_symbols and all(sym.simplify(eq) == 0 for eq in equations):
        # All equations are 0 = 0, trivially satisfied
        return []
    
    # Determine fixed parameters (symbols that can never be solve vars)
    if solve_vars is not None:
        solve_vars_set = set(solve_vars)
        fixed_params = sorted(list(all_symbols - solve_vars_set), key=str)
        potential_vars = sorted(list(solve_vars_set & all_symbols), key=str)
    else:
        fixed_params = []
        potential_vars = sorted(list(all_symbols), key=str)
    
    # Build input strings for the Singular proc (no spaces)
    str_fixed_params = ",".join(str(p) for p in fixed_params)
    str_potential_vars = ",".join(str(v) for v in potential_vars)
    str_eqs = ",".join(str(eq).replace("**", "^") for eq in equations)

    # Run Singular
    raw_output = singular.eval(f'solve_cover("{str_fixed_params}", "{str_potential_vars}", "{str_eqs}");')
    print("Singular Output:\n", raw_output)  # Print first 500 chars for debugging


    # TODO: separate out the parser
    # Parse Output
    if "|||START|||" not in raw_output:
        print("ERROR: Singular script did not produce expected output markers")
        print("Raw output:", raw_output[:500])
        return []
    
    content = raw_output.split("|||START|||")[1].split("|||END|||")[0]
    raw_branches = content.split("|||BRANCH|||")[1:]
    
    # Helper to convert Singular power notation back to SymPy
    def fix_powers(text, var_names):
        for v in var_names:
            pattern = rf'(?<![a-zA-Z])({v})(\d+)'
            text = re.sub(pattern, rf'\1**\2', text)
        return text
    
    # All variable names for power fixing
    all_var_names = [str(v) for v in fixed_params] + [str(v) for v in potential_vars]
    
    parsed_results = []
    
    for b_text in raw_branches:
        lines = b_text.strip().split("\n")
        branch_data = {
            'id': int(lines[0]),
            'component': None,
            'params': list(fixed_params),  # Start with fixed params
            'vars': [],
            'constraints': [],
            'constraint_basis': [],
            'nonnull': [],
            'basis': [],
            'num_solutions': None,
            'mappings': {}
        }
        
        mode = None
        discovered_params = []
        
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            
            if "|||COMPONENT|||" in line:
                branch_data['component'] = int(line.replace("|||COMPONENT|||", "").strip())
                continue
            if "|||PARAMS|||" in line:
                params_str = line.replace("|||PARAMS|||", "").strip()
                params = [v.strip() for v in params_str.split(",") if v.strip()]
                # Add params -- fixed params are already included
                branch_data['params'] = params
                continue
            if "|||VARS|||" in line:
                vars_str = line.replace("|||VARS|||", "").strip()
                branch_data['vars'] = [v.strip() for v in vars_str.split(",") if v.strip()]
                continue
            if "|||CONSTRAINTS|||" in line:
                mode = "constraints"
                continue
            if "|||CONSTRAINT_BASIS|||" in line:
                mode = "constraint_basis"
                continue
            if "|||NONNULL|||" in line:
                mode = "nonnull"
                continue
            if "|||BASIS|||" in line:
                mode = "basis"
                continue
            if "|||NUM_SOLUTIONS|||" in line:
                num_sols_str = line.replace("|||NUM_SOLUTIONS|||", "").strip()
                branch_data['num_solutions'] = int(num_sols_str)
                continue
            if "|||NUM_CONSTRAINT_SOLUTIONS|||" in line:
                num_sols_str = line.replace("|||NUM_CONSTRAINT_SOLUTIONS|||", "").strip()
                if num_sols_str != "":
                    branch_data['num_constraint_solutions'] = int(num_sols_str)
                continue
            
            # Clean and parse
            clean_line = line.replace("^", "**")
            clean_line = fix_powers(clean_line, all_var_names)
            
            if mode in ["constraints", "nonnull", "basis", "constraint_basis"]:
                expr = sym.simplify(sym.sympify(clean_line))
                if expr == 0:
                    continue
                branch_data[mode].append(expr)

        if branch_data['basis'] == [1]:
            # Inconsistent branch, skip
            continue

        # No constraints
        if branch_data["constraints"] == []:
            param_solutions = [{}]
        # Zero-dimensional parameter constraints (-1 means infinite)
        # Exact number of solutions known
        elif branch_data.get('num_constraint_solutions', -1) > 0:
            param_eqs = branch_data['constraint_basis']
            param_vars = set(itertools.chain.from_iterable(eq.free_symbols for eq in param_eqs))
            param_solutions = sym.solve(param_eqs, param_vars, dict=True)
            if len(param_solutions) != branch_data['num_constraint_solutions']:
                raise ValueError(f"Expected {branch_data['num_constraint_solutions']} sols for branch {branch_data['id']}, got {len(param_solutions)}")
        # Infinite solutions, find parameterized solutions
        else:
            param_solutions = []
            param_branches = []
            for param_branch in solve_with_singular(param_eqs):
                param_solutions += param_branch['mappings']
                param_branches.append(param_branch)
            branch_data['param_branches'] = param_branches

        # Extract mappings from basis using sympy.solve on triangular structure
        dep_vars = [sym.symbols(v) for v in branch_data['vars']]
        dep_eqs = branch_data['basis']
        if branch_data["basis"]:
            dep_vars_in_basis = [v for v in dep_vars if any(eq.has(v) for eq in dep_eqs)]
            solutions = sym.solve(dep_eqs, dep_vars_in_basis, dict=True, simplify=True)
            if len(solutions) != branch_data['num_solutions']:
                breakpoint()
                raise ValueError(f"Expected {branch_data['num_solutions']} sols for branch {branch_data['id']}, got {len(solutions)}")
        else:
            solutions = [{}]  # No equations means all dep_vars are free

        # Combine with parameter solutions
        branch_data['mappings'] = []
        for sol_dict, param_sol_dict in itertools.product(solutions, param_solutions):
            branch_data['mappings'].append({**sol_dict, **param_sol_dict})
        
        # Mark unbounded variables as "Free Parameter"
        branch_data["free_vars"] = [v for v in dep_vars if v not in sol_dict]
        
        parsed_results.append(branch_data)
    
    # renumber branch IDs to be sequential
    for idx, br in enumerate(parsed_results):
        br['id'] = idx + 1
    
    return parsed_results


# =========================================
# EXAMPLE USAGE
# =========================================
if __name__ == "__main__":
    a, b, x, y = sym.symbols('a b x y')

    # Test 1: Auto-discover structure (no solve_vars specified)
    print("=== Test 1: ax = b (auto-discover) ===")
    eqs = [a*x - b]
    branches = solve_with_singular(eqs)
    print(f"Found {len(branches)} Branches:")
    for br in branches:
        print(br)
    # Test 2: Explicitly specify solve_vars (a, b are parameters)
    print("\n=== Test 2: ax = b (solve for x, treat a,b as params) ===")
    eqs = [a*x - b]
    branches = solve_with_singular(eqs, solve_vars=[x])
    print(f"Found {len(branches)} Branches:")
    for br in branches:
        print(br)
    # Test 3: Two equations with explicit solve_vars
    print("\n=== Test 3: x^2 + y = a, a*x = b (solve for x,y) ===")
    eqs = [x**2 + y - a, a*x - b]
    branches = solve_with_singular(eqs, solve_vars=[x, y])
    print(f"Found {len(branches)} Branches:")
    for br in branches:
        print(br)
