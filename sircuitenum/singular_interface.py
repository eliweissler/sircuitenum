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

def parse_singular_output(raw_output: str, potential_vars: Union[list[sym.Symbol], list[str], str],
                          fixed_params: Union[list[sym.Symbol], list[str], str] = [],
                          inv_dummy_map: dict[str, sym.Symbol] = {}) -> List[Dict[str, Any]]:

    if "[1]:" in raw_output:
        raw_output = clean_singular_string(raw_output)

    # Parse Output
    if "|||START|||" not in raw_output:
        print("ERROR: Singular script did not produce expected output markers")
        print("Raw output:", raw_output[:500])
        return []
    
    content = raw_output.split("|||START|||")[1].split("|||END|||")[0]
    raw_branches = content.split("|||BRANCH|||")[1:]
    
    # All variable names for power fixing
    all_var_names = [str(v) for v in fixed_params if str(v) != ","] + [str(v) for v in potential_vars if str(v) != ","]
    
    parsed_results = []
    
    for b_text in raw_branches:
        lines = b_text.strip().split("\n")
        branch_data = {
            'id': int(lines[0]),
            'component': None,
            'params': [],
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
                params = [inv_dummy_map.get(v.strip(), v.strip()) for v in params_str.split(",") if v.strip()]
                # Add params -- fixed params are already included
                branch_data['params'] = [sym.sympify(p) for p in params]
                continue
            if "|||VARS|||" in line:
                vars_str = line.replace("|||VARS|||", "").strip()
                vars = [inv_dummy_map.get(v.strip(), v.strip()) for v in vars_str.split(",") if v.strip()]
                branch_data['vars'] = [sym.sympify(v) for v in vars]
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
                # ac -> a*c
                sym_line = ""
                for i in range(len(clean_line)):
                    ch = clean_line[i]
                    sym_line += inv_dummy_map.get(ch, ch)
                    if ch in all_var_names or ch.isnumeric():
                        if i + 1 < len(clean_line):
                            next_ch = clean_line[i + 1]
                            if next_ch in all_var_names:
                                sym_line += '*'
                expr = sym.simplify(sym.sympify(sym_line))
                if expr == 0:
                    continue
                branch_data[mode].append(expr)

        if branch_data['basis'] == [1]:
            # Inconsistent branch, skip
            continue

        # Separate any ghost constraints that
        # snuck into basis
        dep_vars = set(branch_data['vars'])
        basis_eqs = branch_data['basis']
        
        true_dep_eqs = []
        ghost_constraints = []
        
        for eq in basis_eqs:
            # If equation contains ANY dependent variable, it's a variable definition
            if eq.free_symbols.intersection(dep_vars):
                true_dep_eqs.append(eq)
            else:
                # Otherwise, it's a constraint on parameters that leaked into the basis
                ghost_constraints.append(eq)
        # Update branch data
        branch_data['basis'] = true_dep_eqs
        branch_data['constraints'] += ghost_constraints


        # No constraints
        raw_param_solutions = []
        if branch_data["constraints"] == []:
            raw_param_solutions += [{}]
        # Zero-dimensional parameter constraints (-1 means infinite)
        # Exact number of solutions known
        elif branch_data.get('num_constraint_solutions', -1) > 0:
            param_eqs = branch_data['constraint_basis']
            param_vars = set(itertools.chain.from_iterable(eq.free_symbols for eq in param_eqs))
            raw_param_solutions = sym.solve(param_eqs, param_vars, dict=True)
            if len(raw_param_solutions) != branch_data['num_constraint_solutions']:
                raise ValueError(f"Expected {branch_data['num_constraint_solutions']} sols for branch {branch_data['id']}, got {len(raw_param_solutions)}")
        # Infinite solutions, find parameterized solutions
        else:
            param_branches = []
            for param_branch in solve_with_singular(branch_data["constraints"]):
                this_sol = param_branch["mapping"]
                raw_param_solutions += [this_sol]
                param_branches.append(param_branch)
            branch_data['param_branches'] = param_branches

        # Extract mappings from basis using sympy.solve on triangular structure
        dep_vars = branch_data['vars']
        param_vars = branch_data['params']
        dep_eqs = branch_data['basis']
        if branch_data["basis"]:
            if branch_data['num_solutions'] == -1:
                dep_solutions = []
                dep_branches = []
                for dep_branch in solve_with_singular(dep_eqs):
                    solutions += [dep_branch['mapping']]
                    dep_branches.append(dep_branch)
                branch_data['dep_branches'] = dep_branches
            else:
                # Take multiplicity of zero roots into account
                # for equations like x^3 = 0, count as 3 solutions
                solutions = sym.solve(dep_eqs, dep_vars, dict=True, simplify=True)
                reduction = 0
                for eq in dep_eqs:
                    if isinstance(eq, sym.Pow):
                        base = eq.args[0]
                        exponent = eq.args[1]
                        if exponent.is_integer and exponent > 1:
                            reduction += exponent - 1
                if len(solutions) != branch_data['num_solutions'] - reduction:
                    raise ValueError(f"Expected {branch_data['num_solutions']} sols for branch {branch_data['id']}, got {len(solutions)}")
        else:
            solutions = [{}]  # No equations means all dep_vars are free

        # Combine with parameter solutions
        branch_data['mappings'] = []
        branch_data["free_vars"] = []
        branch_data["free_params"] = []
        for sol_dict, param_sol_dict in itertools.product(solutions, param_solutions):
            this_sol = {**sol_dict, **param_sol_dict}
            branch_data['mappings'].append(this_sol)
            branch_data["free_vars"].append([v for v in dep_vars if v not in branch_data['mappings'][-1]])
            branch_data["free_params"].append([v for v in param_vars if v not in branch_data['mappings'][-1]])
        
        parsed_results.append(branch_data)
    
    # renumber branch IDs to be sequential
    for idx, br in enumerate(parsed_results):
        br['id'] = idx + 1

    return parsed_results

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
    raw_output = singular.eval(f'solve_cover("{str_fixed_params}", "{str_potential_vars}", "{str_eqs}");')
    print("Singular call complete.")
    # print("output snippet:", raw_output)

    branches = parse_singular_output(raw_output, str_potential_vars, str_fixed_params, inv_dummy_map=inv_dummy_map)
    branches_flat = []
    # Filter branches based on nonnull constraints
    for br in flatten_branches(branches):
        # if "10" in br['id']:
        #     print("branch 10 detected", br["id"])
        #     print(br)
        this_sol = br['mapping']
        this_non_null = br['nonnull']
        if all(_robust_substitute(constr, this_sol) != 0 for constr in this_non_null):
            branches_flat.append(br)

    # print(f"Filtering {len(branches_flat)} branches for redundancy...")
    # print("Before filtering:")
    # for br in branches_flat:
        # print(br)
    # branches_filtered = filter_redundant_branches(branches_flat)
    # print(f"Reduced to {len(branches_filtered)} unique branches after filtering.")
    # Check each branch for validity
    branches_filtered = branches_flat
    for br in branches_filtered:
        is_valid = True
        for eq in equations:
            lhs_val = _robust_substitute(eq, br['mapping'])
            if lhs_val != 0:
                is_valid = False
                # print(f"Warning: Branch {br['id']} does not satisfy equation {eq} (got {lhs_val})")
                # print(f"  Mapping: {br['mapping']}")
                # print(f"  Basis: {br['basis']}")
                # print(f"  Non-null: {br['nonnull']}")
    return branches_filtered


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
    # Flatten branches if needed
    if any('mappings' in br for br in branches):
        branches = flatten_branches(branches)
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

