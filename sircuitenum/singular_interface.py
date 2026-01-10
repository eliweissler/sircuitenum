"""
Singular Interface for Groebner Cover computation.

This module provides a Python/Sage interface to the Singular grobcov library
for computing the Groebner Cover of a parametric polynomial ideal.

Author: Eli Weissler
Version: 0.1.0
"""

__all__ = [
    "groebner_cover",
    "run_grobcov",
    "parse_grobcov_output",
    "format_grobcov_input",
    "solve_with_singular",
]

import re
from typing import List, Dict, Any, Sequence, Union

from sage.interfaces.singular import singular

import subprocess
import sympy
from sympy import symbols, sympify, simplify

def solve_with_singular(equations, solve_vars):
    """
    Solves a system of SymPy equations using Singular's triangular decomposition.
    Returns a provably complete set of solution branches including singular cases.

    Args:
        equations (list): List of SymPy expressions (assumed equal to 0).
        solve_vars (list): List of SymPy symbols to solve for (unknowns).
        singular_bin (str): Path to the Singular executable.

    Returns:
        list[dict]: A list of branches. Each branch is a dict:
            {
                'id': int,
                'constraints': [sympy_expr, ...],  # Conditions on parameters
                'mappings': {sympy_symbol: sympy_expr, ...} # Explicit solutions
            }
    """
    # 1. Identify Parameters (All symbols NOT in solve_vars)
    all_symbols = set()
    for eq in equations:
        all_symbols.update(eq.free_symbols)
    
    # Sort for deterministic behavior
    # Note: Singular relies on variable order for 'lp' mode (Unknowns > Params)
    unknowns = sorted(list(set(solve_vars)), key=str)
    params = sorted(list(all_symbols - set(unknowns)), key=str)
    
    # Format variables for Singular string input
    str_unknowns = ", ".join(str(v) for v in unknowns)
    str_params = ", ".join(str(p) for p in params)
    
    # 2. Construct Ring Definitions
    # Ring 1 (Decomposition): All symbols are variables. Order is critical (lp).
    if params:
        ring_decomp = f"ring r = (0, {str_params}), ({str_unknowns}), lp;"
        ring_solver = f"ring r_solved = (0, {str_params}), ({str_unknowns}), lp;"
    else:
        # 0-dim case: No parameters exists
        ring_decomp = f"ring r = 0, ({str_unknowns}), lp;"
        ring_solver = f"ring r_solved = 0, ({str_unknowns}), lp;"

    # Format equations (SymPy ** -> Singular ^)
    str_eqs = ",\n  ".join(str(eq).replace("**", "^") for eq in equations)

    # 4. GENERATE SCRIPT (Using String Buffer)
    # Workflow: minAssGTZ -> for each component, find independent set -> grobcov
    # This ensures grobcov gets 0-dimensional systems
    
    all_vars = f"{str_params}, {str_unknowns}" if params else str_unknowns
    num_params = len(params)
    
    # Build Singular script - use grobcov directly with params in coefficient field
    if params:
        ring_def = f'ring r = (0, {str_params}), ({str_unknowns}), lp;'
    else:
        ring_def = f'ring r = 0, ({str_unknowns}), lp;'
    
    # grobcov segment structure:
    # seg[1] = reduced Groebner basis for this segment
    # seg[2] = original ideal mapped to this segment (not needed)
    # seg[3] = list with segment info
    #   seg[3][1] = [E, N] where E = ideals that must be 0, N = ideals that must be != 0
    
    script_parts = [
        'LIB "grobcov.lib";',
        '',
        ring_def,
        f'ideal i = {str_eqs};',
        '',
        '// Run grobcov directly - it handles parameter stratification',
        'def C = grobcov(i);',
        '',
        'string out = "";',
        'out = out + "|||START|||";',
        '',
        'int k, j;',
        'int branch_id = 0;',
        '',
        'for (k=1; k<=size(C); k++)',
        '{',
        '    branch_id = branch_id + 1;',
        '',
        '    def seg = C[k];',
        '    // seg[1] = lpp (leading power product), seg[2] = actual ideal, seg[3] = segment info',
        '    ideal basis = seg[2];',
        '    def seginfo = seg[3];',
        '    def constraints_info = seginfo[1];',
        '    ideal E_null = constraints_info[1];',
        '    list N_nonnull = constraints_info[2];',
        '',
        '    out = out + "|||BRANCH|||" + string(branch_id) + newline;',
        '',
        '    out = out + "|||PARAM_CONSTRAINTS|||" + newline;',
        '    for (j=1; j<=size(E_null); j++)',
        '    {',
        '        if (E_null[j] != 0) { out = out + string(E_null[j]) + newline; }',
        '    }',
        '',
        '    out = out + "|||NONNULL|||" + newline;',
        '    for (j=1; j<=size(N_nonnull); j++)',
        '    {',
        '        if (N_nonnull[j] != 0) { out = out + string(N_nonnull[j]) + newline; }',
        '    }',
        '',
        '    out = out + "|||BASIS|||" + newline;',
        '    for (j=1; j<=size(basis); j++)',
        '    {',
        '        out = out + string(basis[j]) + newline;',
        '    }',
        '',
        '    kill seg, constraints_info, E_null, N_nonnull, seginfo;',
        '}',
        '',
        'out = out + "|||END|||";',
        'out;',
    ]
    script = '\n'.join(script_parts)

    # 4. Run Singular
    raw_output = singular.eval(script)

    # 5. Parse Output
    if "|||START|||" not in raw_output:
        print("ERROR: Singular script did not produce expected output markers")
        return []

    # Extract relevant block
    content = raw_output.split("|||START|||")[1].split("|||END|||")[0]
    raw_branches = content.split("|||BRANCH|||")[1:] # First split is empty
    
    # Helper to convert Singular power notation (x2 -> x**2) back to SymPy
    def fix_powers(text, var_names):
        """Convert Singular's compact power notation back to ** notation."""
        for v in var_names:
            # Match variable name followed by digits (e.g., x2 -> x**2, a3 -> a**3)
            # But only if not preceded by another letter (to avoid matching xa2)
            pattern = rf'(?<![a-zA-Z])({v})(\d+)'
            text = re.sub(pattern, rf'\1**\2', text)
        return text
    
    # Get all variable names for power fixing (both params and unknowns)
    all_var_names = [str(p) for p in params] + [str(u) for u in unknowns]
    
    parsed_results = []

    for b_text in raw_branches:
        lines = b_text.strip().split("\n")
        branch_data = {
            'id': int(lines[0]), 
            'component': None,        # Which prime component this came from
            'indep_vars': [],         # Independent variables (free parameters in this branch)
            'dep_vars': [],           # Dependent variables (solved for)
            'param_constraints': [],  # E = 0 conditions on parameters
            'nonnull': [],            # Expressions that must be != 0
            'basis': [],              # Groebner basis for this segment
            'mappings': {}            # Solved mappings (extracted from basis)
        }
        
        mode = None
        for line in lines[1:]:
            line = line.strip()
            if not line: continue
            
            if "|||COMPONENT|||" in line:
                mode = "component"
                continue
            if "|||INDEP|||" in line:
                mode = "indep"
                continue
            if "|||DEP|||" in line:
                mode = "dep"
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
            
            if mode == "component":
                branch_data['component'] = int(line)
                mode = None
                continue
            if mode == "indep":
                branch_data['indep_vars'] = [v.strip() for v in line.split(",") if v.strip()]
                mode = None
                continue
            if mode == "dep":
                branch_data['dep_vars'] = [v.strip() for v in line.split(",") if v.strip()]
                mode = None
                continue
            
            # Clean Singular syntax for SymPy
            clean_line = line.replace("^", "**")
            # Fix power notation (x2 -> x**2, a3 -> a**3)
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
        
        # Extract mappings from the basis
        # Look for linear equations in unknowns: "x - expr" means x = expr
        for basis_poly in branch_data['basis']:
            for unk in unknowns:
                # Check if this polynomial is linear in unk and can give us unk = something
                try:
                    coeff = basis_poly.coeff(unk)
                    if coeff != 0 and basis_poly.as_poly(unk).degree() == 1:
                        # unk appears linearly: coeff*unk + rest = 0  =>  unk = -rest/coeff
                        rest = basis_poly - coeff * unk
                        mapping = simplify(-rest / coeff)
                        branch_data['mappings'][unk] = mapping
                except:
                    pass
        
        # Mark unknowns not in mappings as Free
        for unk in unknowns:
            if unk not in branch_data['mappings']:
                branch_data['mappings'][unk] = "Free Parameter"
        
        # For backwards compatibility, also set 'constraints' 
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

    # Test 1: Simple ax = b (one equation, one unknown)
    print("=== Test 1: ax = b ===")
    eqs = [a*x - b]
    branches = solve_with_singular(eqs, [x])
    print(f"Found {len(branches)} Branches:")
    for br in branches:
        print(f"  Branch {br['id']}: constraints={br['constraints']}, nonnull={br.get('nonnull', [])}, mappings={br['mappings']}")

    # Test 2: Two equations - x^2 + y = a AND a*x = b
    print("\n=== Test 2: x^2 + y = a, a*x = b ===")
    eqs = [x**2 + y - a, a*x - b]
    branches = solve_with_singular(eqs, [x, y])
    print(f"Found {len(branches)} Branches:")
    for br in branches:
        print(f"  Branch {br['id']}: constraints={br['constraints']}, nonnull={br.get('nonnull', [])}")
        print(f"    mappings={br['mappings']}")
