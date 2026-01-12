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
from typing import List, Dict, Any, Optional

from sage.interfaces.singular import singular

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
    
    # Build variable strings for Singular
    str_fixed_params = ", ".join(str(p) for p in fixed_params)
    str_potential_vars = ", ".join(str(v) for v in potential_vars)
    
    # Format equations (SymPy ** -> Singular ^)
    str_eqs = ",\n  ".join(str(eq).replace("**", "^") for eq in equations)
    
    # Build ring definition
    # If we have fixed params, put them in the coefficient field
    # The potential_vars go into the ring for minAssGTZ to analyze
    if fixed_params:
        ring_def = f'ring r = (0, {str_fixed_params}), ({str_potential_vars}), dp;'
    else:
        ring_def = f'ring r = 0, ({str_potential_vars}), dp;'
    
    script_parts = [
        'LIB "grobcov.lib";',
        'LIB "primdec.lib";',
        ring_def,
        f'ideal i = {str_eqs};',
        '',
        '// Step 1: Compute minimal associated primes',
        'list prime_comps = minAssGTZ(i);',
        '',
        'string out = "";',
        'out = out + "|||START|||";',
        '',
        'int p, k, j, m;',
        'int branch_id = 0;',
        '',
        '// Step 2: For each prime component',
        'for (p=1; p<=size(prime_comps); p++)',
        '{',
        '    ideal comp = prime_comps[p];',
        '    comp = std(comp);',
        '',
        '    // Find maximal independent set within potential_vars',
        '    intvec indep = indepSet(comp);',
        '',
        '    // Build parameter and variable lists',
        '    // Independent vars become additional parameters for grobcov',
        '    // Dependent vars are what we solve for',
        '    string param_str = "";',
        '    string var_str = "";',

        '    for (m=1; m<=nvars(basering); m++)',
        '    {',
        '        if (indep[m] == 1)',
        '        {',
        '            if (size(param_str) > 0) { param_str = param_str + ","; }',
        '            param_str = param_str + string(var(m));',
        '        }',
        '        else',
        '        {',
        '            if (size(var_str) > 0) { var_str = var_str + ","; }',
        '            var_str = var_str + string(var(m));',
        '        }',
        '    }',
        '    string param_str_in = "' + str_fixed_params.replace(" ", "") + '";',
        '    param_str = param_str + param_str_in;',
        'if (param_str == "")',
        '{',
        '    // No independent parameters discovered; single branch with comp basis',
        '    branch_id = branch_id + 1;',
        '    out = out + "|||BRANCH|||" + string(branch_id) + newline;',
        '    out = out + "|||COMPONENT|||" + string(p) + newline;',
        '    out = out + "|||PARAMS|||" + param_str + newline;',
        '    out = out + "|||VARS|||" + var_str + newline;',
        '    out = out + "|||CONSTRAINTS|||" + newline;',
        '    out = out + "|||NUM_CONSTRAINT_SOLUTIONS|||" + newline;',
        '    out = out + "|||NONNULL|||" + newline;',
        '    out = out + "|||BASIS|||" + newline;',
        '    for (j=1; j<=size(comp); j++) { out = out + string(comp[j]) + newline; }',
        '    // Count expected number of solutions',
        '    int num_solutions = vdim(comp);',
        '    out = out + "|||NUM_SOLUTIONS|||" + string(num_solutions) + newline;',
        '}',
        '    else',
        '    {',
        '    // Has parameters: run grobcov',
        '    // Combine fixed_params with discovered independent vars',
        '    string ring_cmd_dp;',
        '    string ring_cmd_lp;',
        '    string ring_cmd_pr;',
    ]
    script_parts.extend([
        'ring_cmd_dp = "ring r_gc_dp = (0," + param_str + "), (" + var_str + "), dp;";',
        'ring_cmd_lp = "ring r_gc_lp = (0," + param_str + "), (" + var_str + "), lp;";',
        'ring_cmd_pr = "ring r_gc_pr = 0, (" + param_str + "), lp;";',
    ])
    script_parts.extend([
        '',
        # '    system("sh", "ring_cmd_param=" + string(ring_cmd_param) + " >> /tmp/sing_debug.log");',
        '    execute(ring_cmd_pr);',
        '    execute(ring_cmd_lp);',
        '    execute(ring_cmd_dp);',
        '    // Use dp for speedy grobcov, lp for fglm conversions',
        '    ideal comp_gc = imap(r, comp);',
        '',
        '    // Run grobcov',
        '    def C = grobcov(comp_gc, "ext", 1);',
        '',
        '    for (k=1; k<=size(C); k++)',
        '    {',
        '        branch_id = branch_id + 1;',
        '',
        '        def seg = C[k];',
        '        // seg[1] = lpp, seg[2] = actual basis, seg[3] = segment info',
        '        // Count expected number of solutions',
        '        ideal basis = seg[2];',
        '        int num_solutions = vdim(basis);',
        '        def seginfo = seg[3];',
        '        def constraints_info = seginfo[1];',
        '        ideal E_null = constraints_info[1];',
        '        list N_nonnull = constraints_info[2];',
        '',
        '        out = out + "|||BRANCH|||" + string(branch_id) + newline;',
        '        out = out + "|||COMPONENT|||" + string(p) + newline;',
        '        out = out + "|||PARAMS|||" + param_str + newline;',
        '        out = out + "|||VARS|||" + var_str + newline;',
        '',
        '        out = out + "|||CONSTRAINTS|||" + newline;',
        '        for (j=1; j<=size(E_null); j++)',
        '        {',
        '            if (E_null[j] != 0) { out = out + string(E_null[j]) + newline; }',
        '        }',
        '        // Count number of solutions to parameter constraints',
        '        '
        '        setring r_gc_pr;'
        '        ideal E_null_pr = imap(r_gc_dp, E_null);'
        '        if (size(variables(E_null_pr)) > 0)',
        '        {',
        '            execute("ring r_temp = 0, (" + string(variables(E_null_pr)) + "), dp;");',
        '            ideal E_null_temp = std(imap(r_gc_pr, E_null_pr));',   
        '            int num_param_sols = vdim(E_null_temp);'
        '            out = out + "|||NUM_CONSTRAINT_SOLUTIONS|||" + string(num_param_sols) + newline;',
        '            if (num_param_sols > 0)',
        '            {',
        '                execute("ring r_temp_lp = 0, (" + string(variables(E_null_temp)) + "), lp;");',
        '                ideal E_null_temp_lp = fglm(r_temp, E_null_temp);',
        '                system("sh", "echo E_null_temp_lp=" + string(E_null_temp_lp) + " >> /tmp/sing_debug.log");',
        '                out = out + "|||CONSTRAINT_BASIS|||" + newline;',
        '                for (j=1; j<=size(E_null_temp_lp); j++)',
        '                {',
        '                    out = out + string(E_null_temp_lp[j]) + newline;',
        '                }',
        '            }',
        '        }',
        '        else',
        '        {',
        '            out = out + "|||NUM_CONSTRAINT_SOLUTIONS|||-1" + newline;',
        '        }',
        '        setring r_gc_dp;',
        '',
        '        out = out + "|||NONNULL|||" + newline;',
        '        for (j=1; j<=size(N_nonnull); j++)',
        '        {',
        '            if (N_nonnull[j] != 0) { out = out + string(N_nonnull[j]) + newline; }',
        '        }',
        '',
    ])
    # Continue emitting basis and wrap up
    script_parts.extend([
        '        out = out + "|||BASIS|||" + newline;',
        '        if (num_solutions > 0) {',
        '           setring r_gc_lp;',
        '           ideal basis = imap(r_gc_dp, basis);',
        '           basis = fglm(r_gc_dp, basis);'
        '        }',
        '',
        '        for (j=1; j<=size(basis); j++)',
        '        {',
        '            out = out + string(basis[j]) + newline;',
        '        }',
        '        setring r_gc_dp;',
        '        out = out + "|||NUM_SOLUTIONS|||" + string(num_solutions) + newline;',
        '        kill seg, constraints_info, E_null, N_nonnull, seginfo;',
        '',
        '        }',
        '',
        '        kill C;',
        '    }',
        '    setring r;',
        '}',
        '',
        'out = out + "|||END|||";',
        'out;',
    ])
    script = '\n'.join(script_parts)

    # Run Singular
    raw_output = singular.eval(script)
    # print(raw_output)

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
            print("Parametric branch detected, solving parameter constraints symbolically...")
            for param_branch in solve_with_singular(param_eqs):
                param_solutions += param_branch['mappings']
                param_branches.append(param_branch)
            branch_data['param_branches'] = param_branches

            # param_sol_dict = {}
            # for eq in param_eqs:
            #     param_sol_dict[eq] = 0  # Keep as constraint

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
