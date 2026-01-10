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
from typing import List, Dict, Any, Optional

from sage.interfaces.singular import singular

import sympy
from sympy import symbols, sympify, simplify


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
    # Collect all symbols from equations
    all_symbols = set()
    for eq in equations:
        all_symbols.update(eq.free_symbols)
    
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
        '',
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
        '',
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
        '',
        '    // Skip if no dependent variables',
        '    if (size(var_str) == 0) { continue; }',
        '',
        '    // Create ring with proper structure for grobcov',
        '    // Combine fixed_params with discovered independent vars',
        '    string ring_cmd;',
    ]
    
    # Handle the ring creation differently based on whether we have fixed params
    if fixed_params:
        script_parts.extend([
            '    if (size(param_str) > 0)',
            '    {',
            f'        ring_cmd = "ring r_gc = (0,{str_fixed_params}," + param_str + "), (" + var_str + "), lp;";',
            '    }',
            '    else',
            '    {',
            f'        ring_cmd = "ring r_gc = (0,{str_fixed_params}), (" + var_str + "), lp;";',
            '    }',
        ])
    else:
        script_parts.extend([
            '    if (size(param_str) > 0)',
            '    {',
            '        ring_cmd = "ring r_gc = (0," + param_str + "), (" + var_str + "), lp;";',
            '    }',
            '    else',
            '    {',
            '        ring_cmd = "ring r_gc = 0, (" + var_str + "), lp;";',
            '    }',
        ])
    
    script_parts.extend([
        '',
        '    execute(ring_cmd);',
        '    ideal comp_gc = imap(r, comp);',
        '',
        '    // Run grobcov',
        '    def C = grobcov(comp_gc);',
        '',
        '    for (k=1; k<=size(C); k++)',
        '    {',
        '        branch_id = branch_id + 1;',
        '',
        '        def seg = C[k];',
        '        // seg[1] = lpp, seg[2] = actual basis, seg[3] = segment info',
        '        ideal basis = seg[2];',
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
        '        out = out + "|||PARAM_CONSTRAINTS|||" + newline;',
        '        for (j=1; j<=size(E_null); j++)',
        '        {',
        '            if (E_null[j] != 0) { out = out + string(E_null[j]) + newline; }',
        '        }',
        '',
        '        out = out + "|||NONNULL|||" + newline;',
        '        for (j=1; j<=size(N_nonnull); j++)',
        '        {',
        '            if (N_nonnull[j] != 0) { out = out + string(N_nonnull[j]) + newline; }',
        '        }',
        '',
        '        out = out + "|||BASIS|||" + newline;',
        '        for (j=1; j<=size(basis); j++)',
        '        {',
        '            out = out + string(basis[j]) + newline;',
        '        }',
        '',
        '        kill seg, constraints_info, E_null, N_nonnull, seginfo;',
        '    }',
        '',
        '    kill C;',
        '    setring r;',
        '}',
        '',
        'out = out + "|||END|||";',
        'out;',
    ])
    script = '\n'.join(script_parts)
    
    # Run Singular
    raw_output = singular.eval(script)
    
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
            'param_constraints': [],
            'nonnull': [],
            'basis': [],
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
                discovered_params = [v.strip() for v in params_str.split(",") if v.strip()]
                # Add discovered params to the fixed params list
                branch_data['params'] = [str(p) for p in fixed_params] + discovered_params
                continue
            if "|||VARS|||" in line:
                vars_str = line.replace("|||VARS|||", "").strip()
                branch_data['vars'] = [v.strip() for v in vars_str.split(",") if v.strip()]
                continue
            if "|||PARAM_CONSTRAINTS|||" in line:
                mode = "param_constraints"
                continue
            if "|||NONNULL|||" in line:
                mode = "nonnull"
                continue
            if "|||BASIS|||" in line:
                mode = "basis"
                continue
            
            # Clean and parse
            clean_line = line.replace("^", "**")
            clean_line = fix_powers(clean_line, all_var_names)
            
            try:
                expr = simplify(sympify(clean_line))
                if expr == 0:
                    continue
                
                if mode == "param_constraints":
                    branch_data['param_constraints'].append(expr)
                elif mode == "nonnull":
                    branch_data['nonnull'].append(expr)
                elif mode == "basis":
                    branch_data['basis'].append(expr)
            except:
                pass
        
        # Extract mappings from basis
        dep_vars = [symbols(v) for v in branch_data['vars']]
        for basis_poly in branch_data['basis']:
            for var in dep_vars:
                try:
                    coeff = basis_poly.coeff(var)
                    if coeff != 0 and basis_poly.as_poly(var).degree() == 1:
                        rest = basis_poly - coeff * var
                        mapping = simplify(-rest / coeff)
                        branch_data['mappings'][var] = mapping
                except:
                    pass
        
        # Mark unmapped vars as Free
        for var in dep_vars:
            if var not in branch_data['mappings']:
                branch_data['mappings'][var] = "Free Parameter"
        
        # Alias for backwards compatibility
        branch_data['constraints'] = branch_data['param_constraints']
        
        # Skip inconsistent branches (basis = [1] means no solutions)
        if branch_data['basis'] == [1]:
            continue
        
        parsed_results.append(branch_data)
    
    return parsed_results


# =========================================
# EXAMPLE USAGE
# =========================================
if __name__ == "__main__":
    a, b, x, y = symbols('a b x y')

    # Test 1: Auto-discover structure (no solve_vars specified)
    print("=== Test 1: ax = b (auto-discover) ===")
    eqs = [a*x - b]
    branches = solve_with_singular(eqs)
    print(f"Found {len(branches)} Branches:")
    for br in branches:
        print(f"  Branch {br['id']}: params={br['params']}, vars={br['vars']}")
        print(f"    constraints={br['constraints']}, nonnull={br.get('nonnull', [])}")
        print(f"    mappings={br['mappings']}")

    # Test 2: Explicitly specify solve_vars (a, b are parameters)
    print("\n=== Test 2: ax = b (solve for x, treat a,b as params) ===")
    eqs = [a*x - b]
    branches = solve_with_singular(eqs, solve_vars=[x])
    print(f"Found {len(branches)} Branches:")
    for br in branches:
        print(f"  Branch {br['id']}: params={br['params']}, vars={br['vars']}")
        print(f"    constraints={br['constraints']}, nonnull={br.get('nonnull', [])}")
        print(f"    mappings={br['mappings']}")

    # Test 3: Two equations with explicit solve_vars
    print("\n=== Test 3: x^2 + y = a, a*x = b (solve for x,y) ===")
    eqs = [x**2 + y - a, a*x - b]
    branches = solve_with_singular(eqs, solve_vars=[x, y])
    print(f"Found {len(branches)} Branches:")
    for br in branches:
        print(f"  Branch {br['id']}: params={br['params']}, vars={br['vars']}")
        print(f"    constraints={br['constraints']}, nonnull={br.get('nonnull', [])}")
        print(f"    mappings={br['mappings']}")
