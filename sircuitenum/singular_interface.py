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
    val = sym.simplify(expr)
    while any(val.has(var) for var in subs_dict.keys()):
        for var, sub_val in subs_dict.items():
            val = sym.simplify(val.subs(var, sub_val))
    return val

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


def refine_branch_mappings(branch, original_eqs):
    """
    Checks if the branch mappings fully satisfy the original equations.
    If 'residuals' remain (non-zero terms after substitution), it solves 
    them to find the hidden sub-branches.

    Example:
        Mapping: {Z22: 0}
        Original Eq: Z12 * (Z00 - Z10) - Z22
        Residual: Z12 * (Z00 - Z10)  (Not zero!)
        Refinement Solves: [Z12=0] OR [Z00=Z10]
    """
    if 'mappings' not in branch or not branch['mappings']:
        return branch

    refined_mappings = []
    
    for candidate in branch['mappings']:
        # 1. IDENTIFY RESIDUALS
        # Substitute the candidate mapping into all original equations
        residuals = []
        for eq in original_eqs:
            val = eq
            
            # Iterative substitution to resolve chains (a->b, b->c)
            # Loop limit prevents infinite recursion on circular deps
            for _ in range(len(candidate) + 5):
                new_val = val.subs(candidate)
                if new_val == val:
                    break
                val = new_val
            
            # Simplify to handle complex cancellation
            res = val.simplify()
            if res != 0:
                residuals.append(res)
        
        if not residuals:
            # Case A: Perfect fit. The generic mapping works.
            refined_mappings.append(candidate)
        else:
            # Case B: The mapping was too "loose".
            # The residuals represent constraints we missed.
            
            # Identify which variables appear in the residuals
            resid_syms = set().union(*[r.free_symbols for r in residuals])
            
            # Attempt to solve the residuals for these variables
            try:
                refinements = sym.solve(residuals, list(resid_syms), dict=True)
            except NotImplementedError:
                # If SymPy can't solve it, we can't refine it. 
                # This branch might be truly invalid or too complex.
                continue
            
            if not refinements:
                # Contradiction: The residuals cannot be solved.
                # This means the generic branch is invalid in this context.
                continue
                
            for ref in refinements:
                # 3. MERGE REFINEMENT
                # Combine original mapping with new refinement
                new_map = candidate.copy()
                new_map.update(ref)
                
                # 4. RESOLVE DEPENDENCIES AGAIN
                # The refinement might have defined a variable that was previously 
                # on the RHS of a mapping (e.g. Z00 -> Z10, and now Z10 -> 0)
                final_map = {}
                for k, v in new_map.items():
                    val = v
                    for _ in range(len(new_map) + 5):
                        new_val = val.subs(new_map)
                        if new_val == val:
                            break
                        val = new_val
                    final_map[k] = val.simplify()
                    
                refined_mappings.append(final_map)
    
    branch['mappings'] = refined_mappings
    return branch


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
        # Use constraint_basis if available, else constraints
        eqs_to_solve = branch_data['constraint_basis'] if branch_data['constraint_basis'] else branch_data['constraints']
        vars_to_solve = set(itertools.chain.from_iterable(eq.free_symbols for eq in eqs_to_solve))
        
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
        # Recurse!
        sub_branches = solve_with_singular(branch_data['constraints'])
        for sub in sub_branches:
            # We assume recursive solve_with_singular returns valid/filtered mappings
            param_solutions.append(sub['mapping'])
            # Optionally store structure: branch_data['param_branches'].append(sub)

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
            
            # VDIM Check
            reduction = 0
            for eq in dep_eqs:
                if isinstance(eq, sym.Pow):
                    if eq.args[1].is_integer and eq.args[1] > 1:
                        reduction += eq.args[1] - 1
            
            # Note: We rely on Ghost Constraints removal to make this check accurate
            # if len(dep_solutions) != branch_data['num_solutions'] - reduction:
            #     print(f"Warning: Branch {branch_data['id']} solution count mismatch.")

        # C. Infinite Case (Recursion)
        else:
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
        
    # Refine Mappings (Fixes recursive splitting issues like User 10.3)
    if raw_mappings:
        temp_branch = {'mappings': raw_mappings}
        temp_branch = refine_branch_mappings(temp_branch, original_equations)
        final_mappings = temp_branch['mappings']
    else:
        final_mappings = []
        
    # Convert each mapping into a distinct branch object (flat structure)
    for idx, mapping in enumerate(final_mappings):
        new_br = branch_data.copy()
        new_br['id'] = f"{branch_data['id']}.{idx}" # Sub-ID
        new_br['mapping'] = mapping
        new_br['free_vars'] = [v for v in dep_vars if v not in mapping]
        new_br['free_params'] = [v for v in branch_data['params'] if v not in mapping]
        resolved_branches.append(new_br)
        
    return resolved_branches


def solve_with_singular(equations, solve_vars=None, dummy_subs=True) -> List[Dict[str, Any]]:
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
        dummy_subs (bool, optional): Whether to perform dummy substitutions to avoid
            Singular parsing issues with certain symbols. Default is True.

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
    # Convert equations to SymPy expressions if needed
    if isinstance(equations[0], sym.Equality):
        equations = [eq.lhs - eq.rhs for eq in equations]

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
        all_symbols.update(str(s) for s in eq.free_symbols)
    
    # Handle edge case: no variables in the system
    if not all_symbols and all(sym.simplify(eq) == 0 for eq in equations):
        # All equations are 0 = 0, trivially satisfied
        return []

    # Determine Fixed vs Potential variables
    if solve_vars is not None:
        solve_vars_set = set(str(s) for s in solve_vars)
        fixed_params = sorted(all_symbols - solve_vars_set)
        potential_vars = sorted(set(solve_vars_set & all_symbols))
    else:
        fixed_params = []
        potential_vars = sorted(set(str(s) for s in all_symbols))

    # Optional: Dummy substitutions to avoid Singular parsing issues
    dummy_map = {}
    inv_dummy_map = {}
    if dummy_subs:
        ord_val = 97  # ASCII 'a'
        for s in fixed_params + potential_vars:
            # skip e
            if chr(ord_val) == 'e':
                ord_val += 1
            if ord_val > 122:  # ASCII 'z'
                raise ValueError("Too many variables for dummy substitution (max 25)")
            dummy_map[str(s)] = chr(ord_val)
            inv_dummy_map[chr(ord_val)] = str(s)
            ord_val += 1
    else:
        for s in fixed_params + potential_vars:
            dummy_map[str(s)] = str(s)
            inv_dummy_map[str(s)] = str(s)
    
    # Build input strings for the Singular proc (no spaces)
    str_fixed_params = ",".join(dummy_map[str(p)] for p in fixed_params)
    str_potential_vars = ",".join(dummy_map[str(v)] for v in potential_vars)
    str_eqs = ",".join(str(sym.nsimplify(eq)).replace("**", "^") for eq in equations)
    for orig, dummy in dummy_map.items():
        str_eqs = str_eqs.replace(orig, dummy)

    # Run Singular
    print("Calling Singular grobcov solver...")
    print(f'solve_cover("{str_fixed_params}", "{str_potential_vars}", "{str_eqs}");')
    singular_call = f'solve_cover("{str_fixed_params}", "{str_potential_vars}", "{str_eqs}");'
    raw_output = _cached_from_singular_call(singular_call)
    
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

@functools.cache
def _cached_from_singular_call(singular_call: str):
    return singular.eval(singular_call)

def check_branch_validity(branch, original_eqs):
    # Simple check to ensure we don't return garbage
    mapping = branch['mapping']
    for eq in original_eqs:
        if _robust_substitute(eq, mapping) != 0:
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

def _symbols2real(expr):
        d = {var: sym.Symbol(var.name, real=True) for var in expr.free_symbols}
        return expr.subs(d)

def extract_mappings(branches: List[Dict[str, Any]], real_only: bool = False) -> List[Dict[str, Any]]:
    """
    Extracts the variable mappings from each branch into a simplified format.
    
    Args:
        branches (list): List of branch dicts as returned by solve_with_singular.
    
    Returns:
        list: List of dicts mapping variable names to their solutions or "Free Parameter".
    """

    # Flatten branches if needed
    if any('mappings' in br for br in branches):
        branches = flatten_branches(branches)
    simplified_mappings = []
    for br in branches:
        mapping = br['mapping']
        if real_only:
            # Check if mapping has an explicit imaginary part
            if all(not sym.sympify(val).has(sym.I) for val in mapping.values()):
                # Convert to all real variables
                real_mapping = {}
                for var, val in mapping.items():
                    real_mapping[_symbols2real(var)] = _symbols2real(val)
                simplified_mappings.append(real_mapping)
        else:
            simplified_mappings.append(mapping)
    return simplified_mappings


def flatten_branches(branches):
    """
    Expands branches with multiple mappings into separate, distinct branches.
    Retains the original ID but adds a suffix (e.g., 10 -> 10.0, 10.1).
    """
    flat_list = []
    
    for b in branches:
        # If no mappings, may be just a basis constraint (keep as is)
        if not b.get('mappings') and not b.get('basis'):
            new_b = b.copy()
            new_b['mapping'] = {}
            del new_b['mappings']
            flat_list.append(new_b)
            continue
            
        # Explode mappings
        for i, mapping in enumerate(b['mappings']):
            # Create a shallow copy of the branch info
            new_b = b.copy()
            
            # OVERWRITE 'mappings' with just THIS single mapping
            new_b['mapping'] = mapping
            if "free_vars" in b:
                new_b["free_vars"] = b["free_vars"][i]
            if "free_params" in b:
                new_b["free_params"] = b["free_params"][i]
            # if "dep_branches" in b:
            #     new_b["dep_branches"] = b["dep_branches"][i]
            del new_b['mappings']
            
            #  Update ID to track lineage
            # (Using string IDs temporarily for clarity)
            new_b['id'] = f"{b['id']}.{i}" 
            
            flat_list.append(new_b)
            
    return flat_list


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

