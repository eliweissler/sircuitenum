__doc__ = "singular_interface.py: Interface to Singular for solving underdetermined polynomial systems using Groebner Covers."
__author__ = "Eli Weissler"
__version__ = "0.1.0"


__all__ = [
    "solve_with_singular", "is_compatible", "extract_mappings", "get_sage_groebner_basis", "solve_with_sage_0D"
]

import subprocess
import os
import signal
import time
import sys
import select

import sympy as sym
import numpy as np
from sympy import S
import re
import itertools, functools
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Iterable

# Singular Interface -- Initialize once per process
from sage.all import singular, PolynomialRing, QQ, ideal, SR, var, QQbar, lcm
from sage.features import Executable
try:
    SINGULAR_PATH = Executable("Singular","Singular").absolute_filename()
except:
    raise ValueError("Cannot find Singular Binary")
WORKER_SINGULAR = None

# Sircuitenum
from sircuitenum.rational_check import has_at_least_one_clean_root, get_clean_roots, _has_rational_coeffs



def get_sage_groebner_basis(sympy_eqs, sympy_vars):
    """
    Calculates the Groebner Basis using SageMath (Singular backend) 
    but accepts and returns SymPy objects.
    
    Args:
        sympy_eqs: List of SymPy equations or expressions (assumed = 0)
        sympy_vars: List of SymPy symbols
    
    Returns:
        List of SymPy expressions representing the Groebner Basis
    """
    var_names = [str(v) for v in sympy_vars]
    R = PolynomialRing(QQ, names=var_names, order='lex')
    
    sage_eqs = []
    for eq in sympy_eqs:
        if isinstance(eq, sym.Eq):
            expr = eq.lhs - eq.rhs
        else:
            expr = eq
        sage_eqs.append(R(str(expr)))
        
    I = ideal(sage_eqs)
    B = I.groebner_basis()
    
    sympy_basis = []
    for poly in B:
        sympy_basis.append(SR(poly)._sympy_())
    return sympy_basis


def solve_0D_backsub(sympy_eqs, sympy_vars, sympy_params=None, rational_only=False):
    """
    Finds ALL rational solutions for a system with parameters.
    Includes a final substitution pass to ensure all variables are fully numeric.
    """
    if sympy_params is None: sympy_params = []

    # Get triangular groebner basis using sage
    # Order: vars > params (lex is mandatory for triangular form)
    all_vars = sympy_vars + sympy_params
    gb = get_sage_groebner_basis(sympy_eqs, all_vars)
    # Inconsistent system
    if len(gb) == 1 and gb[0] == 1:
        return []
    # Identify any variables that don't actually appear
    sympy_vars =[v for v in sympy_vars if any(v in eq.free_symbols for eq in gb)]
    if len(sympy_vars) == 0:
        raise ValueError(f"Unconstrained variables: {sympy_vars}")

    # Backsubstitute, with branching for multiple roots
    solve_order = sympy_vars[::-1]
    current_branches = [{}] 
    param_set = set(sympy_params)
    for target_var in solve_order:
        next_branches = []
        for partial_sol in current_branches:
            # 1. Substitute knowns
            specialized_basis = [poly.subs(partial_sol) for poly in gb]
            
            # 2. Select Polynomial that contains only target_var + params
            # Pick the one with the lowest degree in target_var
            candidate_polys = []
            allowed_symbols = param_set | {target_var}
            for poly in specialized_basis:
                if poly == 0:
                    continue
                if target_var not in poly.free_symbols:
                    continue
                if poly.free_symbols.issubset(allowed_symbols):
                    candidate_polys.append(poly)
            if not candidate_polys:
                raise ValueError(f"Unconstrained variable: {target_var}")
            poly_to_solve = min(candidate_polys, key=lambda p: sym.degree(p, target_var)).as_poly(target_var)
            # 3. Find roots via factorization -> solve
            roots = []
            if rational_only:
                if has_at_least_one_clean_root(poly_to_solve):
                    for root in get_clean_roots(poly_to_solve):
                        if _has_rational_coeffs(root, sympy_params):
                            roots.append({target_var: root})
            else:
                coeff, factors = sym.factor_list(poly_to_solve)
                for factor, exp in factors:
                    sols = sym.solve(factor, target_var, dict=True, simplify=True)
                    print("sols", sols)
                    roots.extend(sols)

            # 4. Create new branches from roots
            for r in roots:
                new_branch = partial_sol.copy() | r
                next_branches.append(new_branch)
        
        current_branches = next_branches
        if not current_branches:
            return []

    # Final Simplify Pass
    final_results = []
    for sol in current_branches:
        clean_sol = {k: sym.simplify(v) for k, v in sol.items()}
        final_results.append(clean_sol)
        
    return final_results


def is_compatible(equations: Iterable[Union[sym.Expr, sym.Equality]], check_fraction: bool = True) -> bool:
    """
    Checks if a system of SymPy equations is mathematically consistent (has at least one solution).

    This function performs a series of checks ranging from trivial inconsistency detection 
    (e.g., :math:`1 = 0`) to complex ideal membership tests using the Singular algebra system.

    :param equations: A collection of SymPy expressions or equalities. If expressions are provided, 
                      they are assumed to be equal to zero.
    :type equations: Iterable[Union[sym.Expr, sym.Equality]]
    :param check_fraction: If ``True``, detects fractions in the equations. It clears denominators 
                           and applies the Rabinowitsch trick (adding :math:`1 - z \\cdot denom = 0`) 
                           to enforce that denominators cannot be zero. Defaults to ``True``.
    :type check_fraction: bool, optional If True, detects fractions in the equations. It clears denominators

    :return: ``True`` if the system is potentially solvable (compatible), ``False`` if it is 
             proven inconsistent.
    :rtype: bool

    .. note::
        This function uses an external interface to **Singular**. It constructs a Groebner basis 
        computation to determine if the ideal generated by the equations is the whole ring (1).
    """

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

    # Verify fractions are not present and adjust if needed
    if check_fraction:
        new_eqs = []
        denoms = []
        for eq in equations:
            numer, denom = _eq_as_numer_denom(eq)
            if denom != 1:
                denoms.append(denom)
            new_eqs.append(numer)
        equations = new_eqs
        # Rabinowitsch Trick to enforce nonzero conditions
        nz_term = []
        if len(denoms) > 0:
            nz_eq = sym.sympify(1)
            for d in denoms:
                nz_eq *= d
            nz_var = sym.symbols('denomNZVar')
            nz_term = [1 - nz_var * nz_eq]
        equations += nz_term
        
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
    for orig in sorted(dummy_map.keys(), key=lambda x: (-len(x), str(x))):
        str_eqs = str_eqs.replace(orig, dummy_map[orig])

    # 1. Enforce the ring context FIRST
    # 2. Then define the ideal and call the proc
    cmd = f"setring SUPER_RING; check_solvability(ideal({str_eqs}));"
    
    return _cached_compatible(cmd)


@functools.cache
def _cached_compatible(cmd: str):
    if WORKER_SINGULAR is None:
        initialize_singular()
    cleaned = clean_singular_string(WORKER_SINGULAR.eval(cmd))
    return "1" in cleaned


def solve_with_singular(equations: list[sym.Expr], solve_vars=None, check_solvability=True,
                        check_fraction: bool = True, rational_only: bool = False) -> List[Dict[str, Any]]:
    """
    Solves a system of SymPy equations using the Groebner Cover algorithm via Singular.

    This approach utilizes a pipeline of ``facstd`` (factorized Groebner Basis) ->
    ``minAssGTZ`` (minimal associated primes) -> ``indepSet`` (independent sets) ->
    ``grobcov`` (Groebner Cover) to provably find all solution branches, including singular
    and degenerate cases where parameters take specific values.

    :param equations: A list of SymPy expressions or equalities to solve.
    :type equations: List[Union[sym.Expr, sym.Equality]]
    :param solve_vars: A specific list of symbols to treat as dependent variables (unknowns). 
                       Symbols present in ``equations`` but not in ``solve_vars`` are treated 
                       as free parameters (constants) for the solving process. 
                       If ``None``, the algorithm auto-discovers the independent versus 
                       dependent variables for each component.
    :type solve_vars: Iterable[sym.Symbol], optional
    :param check_solvability: If ``True``, runs a quick :func:`is_compatible` check before 
                              attempting the full Groebner Cover. Defaults to ``True``.
    :type check_solvability: bool, optional
    :param check_fraction: If ``True``, processes equations to clear denominators and enforce 
                           non-zero conditions on them. Defaults to ``True``.
    :param rational_only: If ``True``, only allows for solutions that contain rational numbers
    :type check_fraction: bool, optional

    :return: A list of solution branches. Each branch is a dictionary containing:
    
             * **id** (*int*): Unique branch identifier.
             * **component** (*int*): Leading polynomial powers of the Groebner Basis.
             * **params** (*list[str]*): Names of independent variables (free parameters).
             * **vars** (*list[str]*): Names of dependent variables (solved unknowns).
             * **constraints** (*list[sym.Expr]*): Conditions on parameters for this branch to be valid.
             * **constraint_basis** (*list[sym.Expr]*): A Groebner basis for the constraints if there are finite solutions.
             * **nonnull** (*list[sym.Expr]*): Expressions that must not be zero.
             * **basis** (*list*): The Groebner basis for this specific segment.
             * **mapping** (*dict*): A dictionary mapping variable names to their solution expressions 
               or the string "Free Parameter".
    :rtype: List[Dict[str, Any]]
    """

    # Verify fractions are not present and adjust if needed
    denoms = []
    if check_fraction:
        new_eqs = []
        for eq in equations:
            numer, denom = _eq_as_numer_denom(eq)
            if denom != 1:
                denoms.append(denom)
            new_eqs.append(numer)
        equations = new_eqs

    if not isinstance(equations, list):
        equations = list(equations)
    
    all_symbols = set()
    for eq in equations:
        all_symbols.update(str(s) for s in eq.free_symbols)
    
    # Quick compatibility check
    if not equations or not all_symbols:
        return []  # No equations or no variables means no branches
    # Convert equations to SymPy expressions if needed
    if isinstance(equations[0], sym.Equality):
        equations = [eq.lhs - eq.rhs for eq in equations]
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
    for orig in sorted(dummy_map.keys(), key=lambda x: (-len(x), str(x))):
        str_eqs = str_eqs.replace(orig, dummy_map[orig])

    # Run Singular
    singular_call = f'solve_cover("{str_fixed_params}", "{str_potential_vars}", "{str_eqs}");'
    raw_output = _cached_solve(singular_call)

    
    # 1. Parse Raw Output
    all_var_names = list(dummy_map.values())
    raw_branches = parse_singular_output(raw_output, all_var_names, inv_dummy_map=inv_dummy_map)
    final_branches = []
    # 2. Resolve Each Branch (Logic moved "inside solve")
    for raw_br in raw_branches:
        
        if raw_br['basis'] == [1]: continue # Inconsistent

        # Resolve (Handles recursion, filtering, refinement)
        resolved_list = resolve_branch_logic(raw_br, equations, rational_only=rational_only)
        
        final_branches.extend(resolved_list)
    # 3. Final Validity Check
    valid_branches = []
    for br in final_branches:
        if (check_branch_validity(br, equations) and 
        all(_robust_substitute(denom, br['mapping']) != 0 for denom in denoms)):
            valid_branches.append(br)
    
    # 4. Filter Redundant Branches
    valid_branches = filter_redundant_branches(valid_branches)

    return valid_branches

@functools.cache
def _cached_solve(singular_call: str):
    if WORKER_SINGULAR is None:
        initialize_singular()
    out = WORKER_SINGULAR.eval(singular_call)
    return out

def extract_mappings(branches: List[Dict[str, Any]], real_only: bool = False) -> List[Dict[str, Any]]:
    """
    Extracts and simplifies variable mappings from the complex branch structure returned by the solver.

    :param branches: The list of branch dictionaries returned by :func:`solve_with_singular`.
    :type branches: List[Dict[str, Any]]
    :param real_only: If ``True``, filters out any mappings where the solution explicitly 
                      contains the imaginary unit :math:`i` (SymPy's ``I``). Defaults to ``False``.
    :type real_only: bool, optional

    :return: A list of simplified dictionaries, where each dictionary maps variables to their 
             SymPy expressions representing the solutions. Can be used in expr.subs(mapping).
    :rtype: List[Dict[str, Any]]
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


def resolve_branch_logic(branch_data: dict, original_equations: list, rational_only=False) -> List[Dict]:
    """
    Takes a raw branch dict, handles ghost constraints, solves sub-systems
    (finitely or recursively), and merges the results into fully resolved branches.
    """
    
    # ---------------------------------------------------------------------
    # 1. GHOST CONSTRAINT SEPARATION
    # ---------------------------------------------------------------------
    dep_vars_set = set(branch_data['vars'])
    basis_eqs = branch_data['basis']
    
    true_dep_eqs = []
    ghost_constraints = []
    
    for eq in basis_eqs:
        # If eq involves Dependent Vars, it stays in the Basis.
        if eq.free_symbols.intersection(dep_vars_set):
            true_dep_eqs.append(eq)
        # Otherwise, it's a constraint on the Parameters (Ghost).
        else:
            ghost_constraints.append(eq)
            
    # Update the branch data to reflect this separation
    branch_data['basis'] = true_dep_eqs
    # Append ghosts to existing constraints list
    active_constraints = branch_data['constraints'] + ghost_constraints

    # ---------------------------------------------------------------------
    # 2. SOLVE CONSTRAINTS (Recursive & Filtered)
    # ---------------------------------------------------------------------
    # We will store results as "Branch Objects" to preserve metadata 
    # (constraints, basis) from recursive calls.
    param_branches = [] 
    
    # A. Trivial Case (No constraints)
    if not active_constraints:
        # Returns one "empty" branch context
        param_branches = [{'mapping': {}, 'constraints': [], 'basis': []}]
        
    # B. Finite Case (SymPy Solve)
    elif branch_data.get('num_constraint_solutions', -1) > 0:
        eqs_to_solve = branch_data['constraint_basis']
        # Identify which parameters are actually involved
        vars_to_solve = [v for v in branch_data['params'] 
                         if any(v in eq.free_symbols for eq in eqs_to_solve)]
        
        raw_sols = solve_0D_backsub(eqs_to_solve, vars_to_solve,
                                    rational_only=rational_only)
        
        # NONNULL FILTER
        nonnull_exprs = branch_data.get('nonnull', [])
        for sol in raw_sols:
            is_valid = True
            for nk in nonnull_exprs:
                if nk.subs(sol).simplify() == 0:
                    is_valid = False
                    break
            if is_valid:
                # Wrap the mapping in a standardized Branch Object
                param_branches.append({
                    'mapping': sol,
                    'constraints': branch_data["constraints"],
                    'basis': []
                })
                
    # C. Infinite Case (Recursive Singular Call)
    else:
        # Recursively solve the constraint system.
        # This returns a list of full branch dictionaries.
        # We pass 'params' as the variables to solve for.
        param_branches = solve_with_singular(active_constraints, solve_vars=branch_data['params'], 
                                             check_fraction=False, rational_only=rational_only)

    # ---------------------------------------------------------------------
    # 3. SOLVE DEPENDENT VARIABLES (Triangular)
    # ---------------------------------------------------------------------
    dep_branches = []
    dep_eqs = branch_data['basis']
    dep_vars = branch_data['vars']
    
    if not dep_eqs:
        dep_branches = [{'mapping': {}, 'constraints': [], 'basis': []}]
    else:
        # B. Finite Case
        if branch_data['num_solutions'] != -1:
            raw_dep_sols = solve_0D_backsub(dep_eqs, dep_vars, branch_data["params"],
                                            rational_only=rational_only)
            for sol in raw_dep_sols:
                dep_branches.append({
                    'mapping': sol,
                    'constraints': [],
                    'basis': dep_eqs
                })

        # C. Infinite Case (Recursive Singular Call)
        else:
            # Recursively solve the dependent system.
            dep_branches = solve_with_singular(dep_eqs, solve_vars=dep_vars, check_fraction=False,
                                               rational_only=rational_only)

    # ---------------------------------------------------------------------
    # 4. COMBINE and MERGE BRANCHES
    # ---------------------------------------------------------------------
    resolved_branches = []
    
    # Cartesian Product of Parameter Branches * Dependent Branches
    # Now we are merging Objects, not just mappings.
    count = 0
    for p_br, d_br in itertools.product(param_branches, dep_branches):
        
        # A. Merge Mappings
        combined_mapping = {**p_br.get('mapping', {}), **d_br.get('mapping', {})}
        
        # B. Merge Constraints
        # We accumulate any 'residue' constraints returned by the recursive calls
        combined_constraints = (
            p_br.get('constraints', []) + 
            d_br.get('constraints', [])
        )
        
        # C. Merge Basis
        combined_basis = (
             p_br.get('basis', []) + 
             d_br.get('basis', [])
        )

        new_br = branch_data.copy()
        
        new_br['id'] = f"{branch_data['id']}.{count}"
        count += 1
        new_br['mapping'] = combined_mapping
        new_br['constraints'] = combined_constraints # Preserve recursive constraints
        new_br['basis'] = combined_basis
        
        # Recalculate Free Vars/Params based on what ended up in the mapping
        new_br['free_vars'] = [v for v in dep_vars if v not in combined_mapping]
        new_br['free_params'] = [v for v in branch_data['params'] if v not in combined_mapping]
        
        resolved_branches.append(new_br)
    
    return resolved_branches


def make_dummy_map(var_list: List[sym.Symbol]):
    dummy_map = {}
    inv_dummy_map = {}
    # Identify any variables that are single lowercase letters
    for s in var_list:
        if re.fullmatch(r'[a-z]', str(s)):
            dummy_map[str(s)] = str(s)
            inv_dummy_map[str(s)] = str(s)
    # Dummy substitutions to avoid Singular parsing issues
    ord_val = 97  # ASCII 'a'
    for s in var_list:
        if str(s) in dummy_map:
            continue  # Already assigned
        # skip e
        if chr(ord_val) in ['e', 'i']:
            ord_val += 1
        if ord_val > 122:  # ASCII 'z'
            raise ValueError("Too many variables for dummy substitution (max 24)")
        dummy_map[str(s)] = chr(ord_val)
        inv_dummy_map[chr(ord_val)] = str(s)
        ord_val += 1
    return dummy_map, inv_dummy_map


def check_branch_validity(branch, original_eqs, eps=1e-12):
    # Simple check to ensure we don't return garbage
    mapping = branch['mapping']
    for eq in original_eqs:
        val = sym.simplify(_robust_substitute(eq, mapping))
        failed = True
        if len(val.free_symbols) == 0:
            if abs(val) < eps:
                failed = False
        if failed:
            print(f"Branch {branch['id']} failed validity check on equation {eq} with mapping {mapping}")
            print(f"Result: {_robust_substitute(eq, mapping)}")
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


def _eq_as_numer_denom(eq: Union[sym.Eq, sym.Expr]):
    """Extract numerator and denominator from equation or expression.
    
    Parameters
    ----------
    eq : sym.Eq or sym.Expr
        Equation or expression to decompose.
        
    Returns
    -------
    tuple[sym.Expr, sym.Expr]
        Numerator and denominator after combining fractions.
    """
    if isinstance(eq, sym.Eq):
        eq = (eq.lhs - eq.rhs)
    # Combine fractions -- quick
    numer, denom = eq.as_numer_denom()
    if numer.as_numer_denom()[1] != 1 and denom.as_numer_denom()[1] != 1:
        # More complex fractions present
        eq_new = sym.expand(eq).together(deep=True)
        numer, denom = eq_new.as_numer_denom()
    return numer, denom


def initialize_singular():
    """
    Called once per worker to initialize a fresh singular interface
    """
    # Sage's process spawner tries to flush stdout, but multiprocessing 
    # workers on macOS often have sys.stdout set to None. 
    # Assign it to /dev/null to prevent an AttributeError.
    # if sys.stdout is None:
    #     sys.stdout = open(os.devnull, 'w')
    # if sys.stderr is None:
    #     sys.stderr = open(os.devnull, 'w')
    lib_path = Path(__file__).with_name("poly_solver.sing")
    startup_code = ""
    startup_code += f'LIB "{str(lib_path)}";\n'
    startup_code += "ring SUPER_RING = 0, (a,b,c,d,f,g,h,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z), dp;\n"
    # startup_code += "option(prot);"
    global WORKER_SINGULAR
    WORKER_SINGULAR = SafeSingular(startup_code=startup_code, binary_path=SINGULAR_PATH, timeout=60)



class SafeSingular:
    def __init__(self, startup_code="", binary_path=SINGULAR_PATH, timeout=30):
        self.binary_path = binary_path
        self.startup_code = startup_code
        self.timeout = timeout
        self.process = None

    def start(self):
        """Launches process and runs startup code."""
        self.process = subprocess.Popen(
            [self.binary_path, "--no-tty", "--quiet"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True, # Critical for killing process tree
            bufsize=1
        )
        
        # Immediately run the startup code if it exists
        if self.startup_code:
            try:
                self.eval(self.startup_code)
            except (TimeoutError, RuntimeError):
                print("Startup code failed! Killing process.")
                self.kill()
                raise

    def eval(self, cmd, timeout=None):
        """
        Public method: Ensures process exists, then runs command.
        """
        # Auto-restart if dead
        if self.process is None:
            self.start()

            
        return self._execute_raw(cmd, timeout)
    
    def _execute_raw(self, cmd, timeout):

        if timeout is None:
            timeout = self.timeout
        # Unique sentinel to mark end of command
        sentinel = "___CMD_DONE___"
        
        # Ensure command ends with newline and forces a flush
        if not cmd.strip().endswith(";"):
            cmd += ";"
        full_cmd = f"{cmd};\nprint('{sentinel}');\nprint('');\n"

        if self.process.poll() is not None:
            self.process = None
            raise RuntimeError("Singular process is dead.")

        # Write command
        try:
            self.process.stdin.write(full_cmd)
            self.process.stdin.flush()
        except (BrokenPipeError, AttributeError):
            self.kill()
            raise RuntimeError("Singular process died unexpectedly.")

        output_buffer = ""
        start_time = time.time()
        
        # Get file descriptor for low-level read
        fd = self.process.stdout.fileno()

        while True:
            # Calculate timeout
            if timeout:
                remaining = timeout - (time.time() - start_time)
                if remaining <= 0:
                    self.kill()
                    raise TimeoutError(f"Singular timed out after {timeout} seconds")
            else:
                remaining = None

            # Select waits for DATA, not LINES. 
            ready, _, _ = select.select([self.process.stdout], [], [], remaining)

            if ready:
                # RAW READ: Read up to 1024 bytes. 
                # This returns '1' or '.' immediately without waiting for \n
                chunk = os.read(fd, 1024).decode('utf-8', errors='replace')
                
                if not chunk: # EOF
                    self.kill()
                    raise RuntimeError("Singular process closed connection.")
                
                output_buffer += chunk

                # Check for Soft Errors
                if "?" in chunk and "error" in chunk.lower():
                     # Simple heuristic: if we see "? ... error", abort
                     pass 

                # Check for sentinel
                if sentinel in output_buffer:
                    break
            else:
                self.kill()
                raise TimeoutError("Singular timed out")

        return output_buffer.replace(sentinel, "").strip()

    def _wait_for_prompt(self, timeout):
        """Consumes initial banner/prompt output."""
        start = time.time()
        fd = self.process.stdout.fileno()
        buf = ""
        while (time.time() - start) < timeout:
            r, _, _ = select.select([self.process.stdout], [], [], 0.1)
            if r:
                chunk = os.read(fd, 1024).decode('utf-8')
                buf += chunk
                # Singular prompt is usually ">"
                if ">" in buf:
                    return
        # If we time out, we assume it's silent and ready
        return

    def kill(self):
        """Force kills the process group and resets state."""
        if self.process:
            pgid = os.getpgid(self.process.pid)
            os.killpg(pgid, signal.SIGKILL)
            self.process.wait()
            self.process = None
