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
import itertools, functools
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

# Load the Singular library and define the procedure
from sage.interfaces.singular import singular
lib_path = Path(__file__).with_name("poly_solver.sing")
singular.eval(f'LIB "{str(lib_path)}";')
singular.eval("ring SUPER_RING = 0, (a,b,c,d,f,g,h,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z), dp;")

import sympy as sym

# Helper to convert Singular power notation back to SymPy
# (e.g. x2 -> x**2)
def fix_powers(text, var_names):
    new_text = ""
    i = 0
    n_chars = len(text)
    while i < n_chars:
        this_char = text[i]
        if this_char in var_names:
            # Check for following digits
            j = i + 1
            power_str = ""
            while j < n_chars and text[j].isdigit():
                power_str += text[j]
                j += 1
            if power_str:
                new_text += this_char + "**" + power_str
                i = j
            else:
                new_text += this_char
                i += 1
        else:
            new_text += this_char
            i += 1
    return new_text


def clean_singular_string(raw_output):
    """
    Removes Singular list markers ([1]:) and comments (//)
    so the output looks exactly like the old string format.
    """
    lines = raw_output.splitlines()
    clean_lines = []
    # Regex to match "[123]: " at the start of a line
    marker_re = re.compile(r'^\s*\[\d+\]:\s*')
    for line in lines:
        # 1. Drop warning/comment lines
        if line.strip().startswith("//"):
            continue
        # 2. Remove the "[N]:" marker, keeping the rest of the line
        # e.g. "[1]:   |||START|||"  -->  "|||START|||"
        # e.g. "[9]:   (a)"          -->  "(a)"
        cleaned = marker_re.sub('', line)
        # 3. Keep non-empty lines
        if cleaned.strip():
            clean_lines.append(cleaned)
    # Join back into the single block of text your parser expects
    return "\n".join(clean_lines)


def _robust_substitute(expr, subs_dict):
    val = expr
    while any(val.has(var) for var in subs_dict.keys()):
        for var, sub_val in subs_dict.items():
            val = val.subs(var, sub_val)
    return sym.simplify(val)

def parse_singular_output(raw_output: str, 
                       all_var_names: list[str], 
                       inv_dummy_map: dict[str, str]) -> List[Dict[str, Any]]:
    """
    Parses Singular output string into a list of raw dictionaries with SymPy expressions.
    Does NOT perform any solving, recursion, or filtering.
    """
    if "[1]:" in raw_output:
        raw_output = clean_singular_string(raw_output)

    if "|||START|||" not in raw_output:
        # print("ERROR: Singular script did not produce expected output markers")
        return []
    
    content = raw_output.split("|||START|||")[1].split("|||END|||")[0]
    raw_branches = content.split("|||BRANCH|||")[1:]
    
    parsed_raw_branches = []
    
    for b_text in raw_branches:
        lines = b_text.strip().split("\n")
        branch_data = {
            'id': int(lines[0]),
            'component': [],
            'params': [],
            'vars': [],
            'constraints': [],
            'constraint_basis': [],
            'nonnull': [],
            'basis': [],
            'num_solutions': -1, # Default to infinite if missing
            'num_constraint_solutions': -1
        }
        
        mode = None
        
        for line in lines[1:]:
            line = line.strip()
            if not line: continue
            
            # --- METADATA PARSING ---
            if "|||PARAMS|||" in line:
                params_str = line.replace("|||PARAMS|||", "").strip()
                params = [inv_dummy_map.get(v.strip(), v.strip()) for v in params_str.split(",") if v.strip()]
                branch_data['params'] = [sym.sympify(p) for p in params]
                continue
            if "|||VARS|||" in line:
                vars_str = line.replace("|||VARS|||", "").strip()
                vars = [inv_dummy_map.get(v.strip(), v.strip()) for v in vars_str.split(",") if v.strip()]
                branch_data['vars'] = [sym.sympify(v) for v in vars]
                continue
            if "|||NUM_SOLUTIONS|||" in line:
                val = line.replace("|||NUM_SOLUTIONS|||", "").strip()
                if val: branch_data['num_solutions'] = int(val)
                continue
            if "|||NUM_CONSTRAINT_SOLUTIONS|||" in line:
                val = line.replace("|||NUM_CONSTRAINT_SOLUTIONS|||", "").strip()
                if val: branch_data['num_constraint_solutions'] = int(val)
                continue
            
            # --- MODE SWITCHING ---
            if "|||CONSTRAINTS|||" in line:      mode = "constraints"; continue
            if "|||CONSTRAINT_BASIS|||" in line: mode = "constraint_basis"; continue
            if "|||NONNULL|||" in line:          mode = "nonnull"; continue
            if "|||BASIS|||" in line:            mode = "basis"; continue
            if "|||COMPONENT|||" in line:        mode = "component"; continue # Already handled
            
            # --- EXPRESSION PARSING ---
            # 1. Fix Powers (^ -> **)
            clean_line = fix_powers(line, all_var_names)
            # 2. Fix variable spacing (ac -> a*c)
            # (Assumes fix_powers logic is similar or you can use your existing fix_powers)
            sym_line = ""
            for i in range(len(clean_line)):
                ch = clean_line[i]
                sym_line += inv_dummy_map.get(ch, ch)
                # Check for implicit multiplication necessity
                if ch in all_var_names or ch.isnumeric():
                    if i + 1 < len(clean_line):
                        next_ch = clean_line[i + 1]
                        if next_ch in all_var_names or next_ch == '(':
                             sym_line += '*'
            
            try:
                expr = sym.simplify(sym.sympify(sym_line))
                if expr != 0:
                    branch_data[mode].append(expr)
            except:
                pass # logging error

        parsed_raw_branches.append(branch_data)
        
    return parsed_raw_branches


def resolve_branch_logic(branch_data: dict, original_equations: list) -> List[Dict]:
    """
    Takes a raw branch dict, handles ghost constraints, solves sub-systems,
    filters invalid roots, and refines mappings.
    Returns a list of resolved branch dictionaries.
    """
    
    # ---------------------------------------------------------------------
    # 1. GHOST CONSTRAINT SEPARATION
    # ---------------------------------------------------------------------
    dep_vars_set = set(branch_data['vars'])
    basis_eqs = branch_data['basis']
    
    true_dep_eqs = []
    ghost_constraints = []
    
    for eq in basis_eqs:
        if eq.free_symbols.intersection(dep_vars_set):
            true_dep_eqs.append(eq)
        else:
            ghost_constraints.append(eq)
            
    branch_data['basis'] = true_dep_eqs
    # Append ghosts to constraints list
    branch_data['constraints'] = branch_data['constraints'] + ghost_constraints

    # ---------------------------------------------------------------------
    # 2. SOLVE CONSTRAINTS (Recursive & Filtered)
    # ---------------------------------------------------------------------
    param_solutions = []
    
    # A. Trivial Case
    if not branch_data['constraints']:
        param_solutions = [{}]
        
    # B. Finite Case
    elif branch_data.get('num_constraint_solutions', -1) > 0:
        # Use constraint_basis to solve for parameters
        eqs_to_solve = branch_data['constraint_basis']
        vars_to_solve = [v for v in branch_data['params'] if any(v in eq.free_symbols for eq in eqs_to_solve)]
        
        raw_sols = sym.solve(eqs_to_solve, vars_to_solve, dict=True)
        
        # NONNULL FILTER
        nonnull_exprs = branch_data.get('nonnull', [])
        for sol in raw_sols:
            is_valid = True
            for nk in nonnull_exprs:
                if nk.subs(sol).simplify() == 0:
                    is_valid = False
                    break
            if is_valid:
                param_solutions.append(sol)
                
    # C. Infinite Case (Recursion)
    else:
        # TODO: Expand branches so that constraints are consistent
        sub_branches = solve_with_singular(branch_data['constraints'])
        for sub in sub_branches:
            # We assume recursive solve_with_singular returns valid/filtered mappings
            param_solutions.append(sub['mapping'])

    # ---------------------------------------------------------------------
    # 3. SOLVE DEPENDENT VARIABLES (Triangular)
    # ---------------------------------------------------------------------
    dep_solutions = []
    dep_eqs = branch_data['basis']
    dep_vars = branch_data['vars']
    
    if not dep_eqs:
        dep_solutions = [{}]
    else:
        # B. Finite Case
        if branch_data['num_solutions'] != -1:
            # Strictly solve for dependent variables (treat params as constants)
            dep_solutions = sym.solve(dep_eqs, dep_vars, dict=True, simplify=True)

        # C. Infinite Case (Recursion)
        else:
            # TODO: Expand branches so that constraints are consistent
            sub_branches = solve_with_singular(dep_eqs, solve_vars=dep_vars)
            for sub in sub_branches:
                dep_solutions.append(sub['mapping'])

    # ---------------------------------------------------------------------
    # 4. COMBINE & REFINE
    # ---------------------------------------------------------------------
    resolved_branches = []
    
    # Cartesian Product of Parameter solutions * Dependent solutions
    raw_mappings = []
    for p_sol, d_sol in itertools.product(param_solutions, dep_solutions):
        raw_mappings.append({**p_sol, **d_sol})
        
    # Convert each mapping into a distinct branch object (flat structure)
    for idx, mapping in enumerate(raw_mappings):
        new_br = branch_data.copy()
        new_br['id'] = f"{branch_data['id']}.{idx}" # Sub-ID
        new_br['mapping'] = mapping
        new_br['free_vars'] = [v for v in dep_vars if v not in mapping]
        new_br['free_params'] = [v for v in branch_data['params'] if v not in mapping]
        resolved_branches.append(new_br)
        
    return resolved_branches


def make_dummy_map(var_list: List[sym.Symbol]):
    # Optional: Dummy substitutions to avoid Singular parsing issues
    dummy_map = {}
    inv_dummy_map = {}
    ord_val = 97  # ASCII 'a'
    for s in var_list:
        # skip e
        if chr(ord_val) in ['e', 'i']:
            ord_val += 1
        if ord_val > 122:  # ASCII 'z'
            raise ValueError("Too many variables for dummy substitution (max 24)")
        dummy_map[str(s)] = chr(ord_val)
        inv_dummy_map[chr(ord_val)] = str(s)
        ord_val += 1
    return dummy_map, inv_dummy_map


def is_compatible(equations):

    equations = list(equations)
    
    if not equations:
        return True  # Empty system is trivially compatible

    # Convert equations to SymPy expressions if needed
    if isinstance(list(equations)[0], sym.Equality):
        equations = [eq.lhs - eq.rhs for eq in equations]
    
    # Quick inconsistency check: if any equation is a non-zero constant, system is inconsistent
    for eq in equations:
        if eq.is_Number and eq != 0:
            return False
        
    # Collect all symbols from equations
    all_symbols = set()
    for eq in equations:
        all_symbols.update(str(s) for s in eq.free_symbols)
    
    # No variables in the system
    if not all_symbols:
        if all(sym.simplify(eq) == 0 for eq in equations):
            # All equations are 0 = 0, trivially satisfied
            return []
    
    # Not obviously true or false, use Singular to check

    dummy_map, _ = make_dummy_map(all_symbols)

    # Build input strings for the Singular proc (no spaces)
    str_vars = ",".join(dummy_map[str(v)] for v in all_symbols)
    str_eqs = ",".join(str(sym.nsimplify(eq)).replace("**", "^") for eq in equations)
    for orig, dummy in dummy_map.items():
        str_eqs = str_eqs.replace(orig, dummy)
    
    # 1. Enforce the ring context FIRST
    # 2. Then define the ideal and call the proc
    cmd = f"setring SUPER_RING; check_solvability(ideal({str_eqs}));"
    
    return _cached_compatible(cmd)


@functools.cache
def _cached_compatible(cmd: str):
    # print(cmd.split(";")[1])
    cleaned = clean_singular_string(singular.eval(cmd))
    return "1" in cleaned


def solve_with_singular(equations: list[sym.Expr], solve_vars=None, check_solvability=True) -> List[Dict[str, Any]]:
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
        check_solvability (bool): If True, performs a quick compatibility check
            before running the full Groebner Cover. Defaults to True.

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
    if not isinstance(equations, list):
        equations = list(equations)
    # Convert equations to SymPy expressions if needed
    if isinstance(equations[0], sym.Equality):
        equations = [eq.lhs - eq.rhs for eq in equations]
    
    
    all_symbols = set()
    for eq in equations:
        all_symbols.update(str(s) for s in eq.free_symbols)
    
    # Quick compatibility check
    if not equations or not all_symbols:
        return []  # No equations or no variables means no branches
    if check_solvability:
        if not is_compatible(equations):
            return []
    
    # Determine Fixed vs Potential variables
    if solve_vars is not None:
        solve_vars_set = set(str(s) for s in solve_vars)
        fixed_params = sorted(all_symbols - solve_vars_set)
        potential_vars = sorted(set(solve_vars_set & all_symbols))
    else:
        fixed_params = []
        potential_vars = sorted(set(str(s) for s in all_symbols))

    dummy_map, inv_dummy_map = make_dummy_map(fixed_params + potential_vars)


    # Build input strings for the Singular proc (no spaces)
    str_fixed_params = ",".join(dummy_map[str(p)] for p in fixed_params)
    str_potential_vars = ",".join(dummy_map[str(v)] for v in potential_vars)
    str_eqs = ",".join(str(sym.nsimplify(eq)).replace("**", "^") for eq in equations)
    for orig, dummy in dummy_map.items():
        str_eqs = str_eqs.replace(orig, dummy)

    # Run Singular
    # print("Calling Singular grobcov solver...")
    # print(f'solve_cover("{str_fixed_params}", "{str_potential_vars}", "{str_eqs}");')
    singular_call = f'solve_cover("{str_fixed_params}", "{str_potential_vars}", "{str_eqs}");'
    raw_output = _cached_from_singular_call(singular_call)
    # print("Singular call complete.")
    # print(raw_output)
    
    # 1. Parse Raw Output
    all_var_names = list(dummy_map.values())
    raw_branches = parse_singular_output(raw_output, all_var_names, inv_dummy_map=inv_dummy_map)
    final_branches = []
    # 2. Resolve Each Branch (Logic moved "inside solve")
    for raw_br in raw_branches:
        
        if raw_br['basis'] == [1]: continue # Inconsistent

        # Resolve (Handles recursion, filtering, refinement)
        resolved_list = resolve_branch_logic(raw_br, equations)
        
        final_branches.extend(resolved_list)

    # 3. Final Validity Check
    valid_branches = []
    for br in final_branches:
        if check_branch_validity(br, equations):
            valid_branches.append(br)
    
    # 4. Filter Redundant Branches
    valid_branches = filter_redundant_branches(valid_branches)

    return valid_branches

# @functools.cache
def _cached_from_singular_call(singular_call: str):
    # print("Executing Singular command:", singular_call)
    return singular.eval(singular_call)

def check_branch_validity(branch, original_eqs):
    # Simple check to ensure we don't return garbage
    mapping = branch['mapping']
    for eq in original_eqs:
        if sym.simplify(_robust_substitute(eq, mapping)) != 0:
            return False
    return True


def filter_redundant_branches(branches, verbose=False):
    """
    Filters branches assuming they have been flattened (1 mapping per branch).
    """
    
    def is_subset(branch_spec, branch_gen):
        # 1. SETUP MAP
        if 'mapping' not in branch_spec: return False
        spec_map = branch_spec['mapping'].copy()
        
        # Augment with basis
        for eq in branch_spec.get('basis', []):
            if isinstance(eq, sym.Symbol): spec_map[eq] = 0
                
        # 2. BASIS CHECK
        for eq in branch_gen.get('basis', []):
            if eq.subs(spec_map).simplify() != 0: return False 

        # 3. HOLE CHECK
        for constr in branch_gen.get('nonnull', []):
            if constr == 1: continue
            if constr.subs(spec_map).simplify() == 0: return False

        # 4. MAPPING CHECK
        if branch_gen.get('mapping'):
            gen_map = branch_gen['mapping']
            for lhs, rhs in gen_map.items():
                if (lhs - rhs).subs(spec_map).simplify() != 0:
                    return False 
        return True

    to_remove_ids = set()
    for i, b_spec in enumerate(branches):
        if b_spec['id'] in to_remove_ids: continue
        for j, b_gen in enumerate(branches):
            if i == j: continue
            if b_gen['id'] in to_remove_ids: continue

            if is_subset(b_spec, b_gen):
                if verbose: print(f"Removing {b_spec['id']} (Subset of {b_gen['id']})")
                to_remove_ids.add(b_spec['id'])
                break
    
    to_keep = [b for b in branches if b['id'] not in to_remove_ids]
    # sort by ID
    to_keep.sort(key=lambda b: float(b['id']))
    return to_keep


def extract_mappings(branches: List[Dict[str, Any]], real_only: bool = False) -> List[Dict[str, Any]]:
    """
    Extracts the variable mappings from each branch into a simplified format.
    
    Args:
        branches (list): List of branch dicts as returned by solve_with_singular.
    
    Returns:
        list: List of dicts mapping variable names to their solutions or "Free Parameter".
    """

    simplified_mappings = []
    for br in branches:
        mapping = br['mapping']
        if real_only:
            # Check if mapping has an explicit imaginary part
            if all(not sym.sympify(val).has(sym.I) for val in mapping.values()):
                simplified_mappings.append(mapping)
        else:
            simplified_mappings.append(mapping)
    return simplified_mappings


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

